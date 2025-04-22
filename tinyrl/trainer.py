import torch
import json
import numpy as np
import torch.distributed as dist
import threading
import requests
import os
import re
import re
import sys
import time
import datetime
import wandb
import random
import hydra
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from rollout.sos_utils import rollout_sos
from rollout.hsp_utils import rollout_hsp
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import popen_launch_server
from sglang.srt.utils import init_custom_process_group
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm.auto import tqdm
from copy import deepcopy

CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

class PrefixDataset(Dataset):
    def __init__(self, data_path, num_samples=None):
        """
        Initialize dataset with option to load only first N samples
        Args:
            data_path: Path to json data file
            num_samples: If set, only load this many samples from the dataset
        """
        with open(data_path, "r") as f:
            data = json.load(f)
            if num_samples is not None:
                self.data = data[:num_samples]
            else:
                self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GRPOTrainer:
    """
    Assumes 2 GPU, cuda:0 for training, cuda:1 for rollout.
    """
    def __init__(self, config: DictConfig):
        if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            assert len(cuda_visible_devices.split(",")) >= 2, f"CUDA_VISIBLE_DEVICES must be set to at least 2 GPUs (we will use the first 2 GPUs), got {cuda_visible_devices}"
        # Environment setup
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        torch.cuda.set_device(0)  # Use GPU 0 for HF model
        mp.set_start_method("spawn", force=True)

        self.config = config
        self.verbose = config.logging.verbose

        # Initialize wandb
        if config.logging.wandb_name is None:
            wandb_name = f"grpo_{config.model.name.split('/')[-1]}"
            wandb_name += f"_bs{config.data.train_batch_size}"
            wandb_name += f"_rollout{config.rollout.group_size}"
        else:
            wandb_name = config.logging.wandb_name
        
        if config.logging.use_current_time:
            wandb_name += f"-{CURRENT_TIME}"
        self.wandb_name = wandb_name
        print(colored(f"Wandb name: {self.wandb_name}", "blue"))
        wandb.init(
            project=config.logging.wandb_project,
            name=self.wandb_name,
            config=OmegaConf.to_container(config, resolve=True)
        )

        # Data
        print(colored("Loading dataset", "blue"))
        train_dataset = PrefixDataset(config.data.train_data_path)
        val_dataset = PrefixDataset(
            config.data.val_data_path,
            num_samples=config.training.val_samples
        )
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.data.train_batch_size,
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.data.eval_batch_size,
            shuffle=False
        )

        # Rollout 
        condition_prefix = config.rollout.condition_prefix
        if config.rollout.mode == "sos":
            # sos and hs both use sos rollout function
            self.rollout_fn = rollout_sos
        elif config.rollout.mode == "hsp":
            # hsp uses hsp rollout function
            self.rollout_fn = rollout_hsp
        else:
            raise ValueError(f"Rollout mode {config.rollout.mode} not supported")
        
        if condition_prefix is not None:
            from functools import partial
            print(colored(f"Using condition prefix: {condition_prefix}", "blue"))
            self.rollout_fn = partial(self.rollout_fn, condition_prefix=condition_prefix)
        print(colored(f"Loading model {config.model.name}", "blue"))
        # Model - Train
        self.model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype="bfloat16").to("cuda:0")
        if self.config.training.kl_beta > 0:
            self.ref_model = deepcopy(self.model)
        
        # Enable gradient checkpointing if configured
        if self.config.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.config.training.kl_beta > 0:
                self.ref_model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        self.bos_token = self.tokenizer.bos_token
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.training.learning_rate)

        # Get model state dict shapes for weight sync
        self.state_dict_shapes = {
            key: value.shape 
            for key, value in self.model.state_dict().items()
        }

        print(colored(f"Launch serving server as sub-process", "blue"))

        if config.model.server_url.rstrip("/").endswith(":0") or config.model.master_port == 0:
            # if either is 0, we will randomly pick a port between 20000 and 50000 for the server and use the next port for the master
            port = random.randint(20000, 50000)
            config.model.server_url = config.model.server_url.rstrip("0") + f"{port}"
            config.model.master_port = port + 1
        
        # Model - Serve
        mp.set_start_method("spawn", force=True)
        self.server_process = popen_launch_server(
            config.model.name,
            config.model.server_url,
            timeout=60,
            other_args=("--base-gpu-id", "1", "--tp-size", "1", "--mem-fraction-static", str(config.model.mem_fraction_static)),
            # This is used to print the server logs to the console, enable verbose level > 1 to print
            return_stdout_stderr=(sys.stdout, sys.stderr) if self.verbose > 1 else None
        )

        print(colored(f"Initialize weight transfer groups", "blue"))
        def init_weight_group():
            time.sleep(2)
            requests.post(
                f"{self.config.model.server_url}/init_weights_update_group",
                json={
                    "master_address": "localhost",
                    "master_port": f"{self.config.model.master_port}",
                    "rank_offset": 1,
                    "world_size": 2,
                    "group_name": "weight_update_group",
                    "backend": "nccl",
                },
            )
        threading.Thread(target=init_weight_group).start()
        # Initialize distributed process groups
        self.weight_group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{self.config.model.master_port}",
            world_size=2,  # HF model and server
            rank=0,
            group_name="weight_update_group"
        )
        dist.barrier(group=self.weight_group, device_ids=[0])
        torch.cuda.synchronize()
            
    def sync_weights_to_server(self):
        """Sync model weights from training model to server"""
        # Barrier to ensure both processes are ready
        # dist.barrier(group=self.weight_group, device_ids=[0])
        # torch.cuda.synchronize()

        # Broadcast each parameter
        for name, param in self.model.named_parameters():
            dist.broadcast(
                param.data,
                src=0,
                group=self.weight_group
            )
            
            # Update weights on server
            requests.post(
                f"{self.config.model.server_url}/update_weights_from_distributed",
                json={
                    "name": name,
                    "dtype": str(param.dtype).split('.')[-1],
                    "shape": self.state_dict_shapes[name],
                },
            )

        torch.cuda.synchronize()

    def compute_log_prob(self, seqs, model, use_chunk=False):
        # Prepare batches
        all_prefixes = []
        all_full_texts = []
        for seq in seqs:
            all_prefixes.append(seq['prefix'])
            all_full_texts.append(seq['prefix'] + seq['output'])
        
        batch_prefixes = all_prefixes
        batch_full_texts = all_full_texts
        
        # Tokenize prefixes and full texts
        prefix_tokens = self.tokenizer(batch_prefixes, padding=True, return_tensors="pt").to("cuda:0")
        full_tokens = self.tokenizer(batch_full_texts, padding=True, return_tensors="pt").to("cuda:0")
        targets = full_tokens.input_ids[:, 1:]  # Shift right by 1
        seq_token_lengths = full_tokens.input_ids.ne(self.tokenizer.pad_token_id).sum(1)
        if self.verbose:
            print(colored("🔢 [Compute Log Prob] Token shape for full sequence batch: " + str(full_tokens.input_ids.shape), "cyan"))
        
        # Get prefix lengths for each sequence in batch
        prefix_lengths = prefix_tokens.input_ids.ne(self.tokenizer.pad_token_id).sum(1)
        
        if use_chunk:
            # Process in chunks to save memory (only used for old log probs computation)
            chunk_size = self.config.training.get(
                "log_probs_chunk_size", self.config.training.grad_accum_chunk_size)
            num_seqs = full_tokens.input_ids.shape[0]
            seq_length = full_tokens.input_ids.shape[1]
            
            # Initialize tensor to store token log probs
            token_log_probs_all = torch.zeros(
                (num_seqs, seq_length - 1),
                dtype=torch.float32,
                device="cuda:0"
            )
            
            # Process each chunk
            for i in range(0, num_seqs, chunk_size):
                chunk_end = min(i + chunk_size, num_seqs)
                chunk_tokens = {
                    k: v[i:chunk_end] for k, v in full_tokens.items()
                }
                chunk_targets = targets[i:chunk_end].unsqueeze(2)
                
                outputs = model(**chunk_tokens)
                logits = outputs.logits[:, :-1, :]
                chunk_log_probs = torch.log_softmax(logits, dim=-1)
                
                # Compute token log probs for this chunk directly
                chunk_token_log_probs = chunk_log_probs.gather(
                    dim=2, 
                    index=chunk_targets
                ).squeeze(2)
                
                token_log_probs_all[i:chunk_end] = chunk_token_log_probs
                
                # Clean up
                # del outputs
                # del logits
                # del chunk_log_probs
                # del chunk_token_log_probs
                torch.cuda.empty_cache()
        else:
            # Original non-chunked processing
            outputs = model(**full_tokens)
            logits = outputs.logits[:, :-1, :]  # Remove last position
            
            if self.verbose:
                print(colored("🔢 [Compute Log Prob] Logits shape: " + str(logits.shape), "cyan"))
            log_probs_batch = torch.log_softmax(logits, dim=-1)
            
            token_log_probs_all = log_probs_batch.gather(dim=2, index=targets.unsqueeze(2)).squeeze(2)
            
            # del outputs
            # del logits
            # del log_probs_batch
        # Create a mask for valid positions
        batch_seq_len = targets.shape[1]
        range_tensor = torch.arange(batch_seq_len, device=targets.device).unsqueeze(0).expand(prefix_lengths.size(0), -1)
        valid_mask = (range_tensor >= (prefix_lengths - 1).unsqueeze(1)) & (targets.ne(self.tokenizer.pad_token_id))

        assert token_log_probs_all.shape == valid_mask.shape, f"{token_log_probs_all.shape} != {valid_mask.shape}"
        token_log_probs_all = token_log_probs_all * valid_mask
        
        # Clean up intermediate tensors
        # del prefix_tokens
        # del full_tokens
        # del range_tensor
        
        return token_log_probs_all, valid_mask, seq_token_lengths

    def masked_mean(self, values, valid_mask, dim=None):
        """Compute mean of tensor with a masked values."""
        return (values * valid_mask).sum(dim=dim) / (valid_mask.sum(dim=dim) + 1e-10)
    
    def masked_sum(self, values, valid_mask, dim=None):
        """Compute sum of tensor with a masked values."""
        return (values * valid_mask).sum(dim=dim)

    def compute_kl_loss(self, log_prob, ref_log_prob, valid_mask, loss_type='low_var_kl'):
        """Compute KL divergence loss between current and reference policy.
        
        Args:
            log_prob: Current policy log probabilities
            ref_log_prob: Reference policy log probabilities
            valid_mask: Attention mask for valid tokens
            loss_type: Type of KL loss calculation ('low_var_kl' or 'full_kl')
        """
        if loss_type == 'low_var_kl':
            # Low variance estimator from GRPO paper (DeepSeekMath: https://arxiv.org/abs/2402.03300)
            kl_div = torch.exp(ref_log_prob - log_prob) - (ref_log_prob - log_prob) - 1
        else:  # 'full_kl'
            raise NotImplementedError(f"KL Loss type {loss_type} not implemented")
        
        # We will divide by the number of rollouts after loss accumulation
        return self.masked_mean(kl_div, valid_mask, dim=1).sum(dim=0)

    def has_nan(self, tensor, name=None):
        """Utility function to check for NaN values in a tensor."""
        if torch.is_tensor(tensor):
            is_nan = torch.isnan(tensor).any().item()
            if is_nan and name:
                nan_indices = torch.where(torch.isnan(tensor))
                print(colored(f"❌ NaN detected in {name} at indices {nan_indices}", "red"))
                if tensor.numel() < 100:  # Only print full tensor if it's small
                    print(colored(f"Tensor values: {tensor}", "red"))
                print(colored(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}", "red"))
            return is_nan
        return False

    def check_model_nans(self, model, stage=""):
        """Check for NaNs in model parameters and gradients."""
        has_nan_params = False
        has_nan_grads = False
        nan_param_names = []
        nan_grad_names = []
        
        for name, param in model.named_parameters():
            if self.has_nan(param.data, f"{stage} - Parameter {name}"):
                has_nan_params = True
                nan_param_names.append(name)
            if param.grad is not None and self.has_nan(param.grad, f"{stage} - Gradient {name}"):
                has_nan_grads = True
                nan_grad_names.append(name)
        
        if has_nan_params or has_nan_grads:
            print(colored(f"❌ NaN detected in model at stage {stage}", "red"))
            if has_nan_params:
                print(colored(f"Parameters with NaN: {nan_param_names}", "red"))
            if has_nan_grads:
                print(colored(f"Gradients with NaN: {nan_grad_names}", "red"))
        
        return has_nan_params or has_nan_grads

    def compute_grpo_loss(
        self, log_probs, old_log_probs, advantages, valid_mask, seq_per_rollout, clip_ratio=0.2, ref_log_probs=None):
        """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
        Reference: https://rlhfbook.com/c/11-policy-gradients.html#policy-gradient-algorithms

        Args:
            old_log_probs: `(torch.Tensor)`
                shape: (bs, response_length)
            log_probs: `(torch.Tensor)`
                shape: (bs, response_length)
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            valid_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            clip_ratio: (float)
                The clip range used in PPO. See https://arxiv.org/abs/1707.06347

        Returns:
            pg_loss: `a scalar torch.Tensor`
                policy gradient loss computed via PPO
            pg_clipfrac: (float)
                a float number indicating the fraction of policy gradient loss being clipped
            approx_kl: (float)
                approximate KL divergence between old and new policy
        """
        
        # Check for NaNs in inputs
        if self.has_nan(log_probs, "log_probs") or self.has_nan(old_log_probs, "old_log_probs") or \
           self.has_nan(advantages, "advantages") or self.has_nan(valid_mask, "valid_mask"):
            print(colored("❌ NaN detected in GRPO loss inputs", "red"))
            print(colored(f"log_probs stats: min={log_probs.min()}, max={log_probs.max()}, mean={log_probs.mean()}", "red"))
            print(colored(f"old_log_probs stats: min={old_log_probs.min()}, max={old_log_probs.max()}, mean={old_log_probs.mean()}", "red"))
            print(colored(f"advantages stats: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}", "red"))
            return None, None, None
        
        # Compute probability ratio between new and old policies
        ratio = torch.exp(log_probs - old_log_probs)
        
        if self.has_nan(ratio, "policy_ratio"):
            print(colored("❌ NaN detected in policy ratio", "red"))
            print(colored(f"log_probs - old_log_probs stats: min={(log_probs - old_log_probs).min()}, max={(log_probs - old_log_probs).max()}", "red"))
            return None, None, None
        
        # PPO clipping objective
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        if self.has_nan(pg_losses1, "pg_losses1") or self.has_nan(pg_losses2, "pg_losses2"):
            print(colored("❌ NaN detected in PPO losses", "red"))
            return None, None, None
        
        pg_loss_max = torch.max(pg_losses1, pg_losses2)
        
        # Apply completion mask and compute final loss
        pg_loss = self.masked_mean(pg_loss_max, valid_mask, dim=1) / (seq_per_rollout + 1e-10)
        pg_loss = pg_loss.sum(dim=0)
        
        if self.has_nan(pg_loss, "final_pg_loss"):
            print(colored("❌ NaN detected in final PG loss", "red"))
            return None, None, None
        
        if ref_log_probs is not None:
            if self.has_nan(ref_log_probs, "ref_log_probs"):
                print(colored("❌ NaN detected in reference log probs", "red"))
                return None, None, None
            pg_low_var_kl = self.compute_kl_loss(log_probs, ref_log_probs, valid_mask, loss_type='low_var_kl')
        else:
            pg_low_var_kl = 0.

        # Compute metrics for logging
        with torch.no_grad():
            # Compute clipping fraction
            pg_clipfrac = ((pg_losses2 > pg_losses1).float() * valid_mask).sum() / valid_mask.sum()
            
        return pg_loss, pg_low_var_kl, pg_clipfrac

    def drop_zero_adv_seqs(self, seqs):
        return [seq for seq in seqs if seq['advantage'] != 0]

    def compute_grpo_advantage(self, data, group_size, eps=1e-10):
        rewards = np.array([int(dp['is_correct']) for dp in data])
        n_groups = len(rewards) // group_size
        assert n_groups * group_size == len(rewards)
        rewards = rewards.reshape(-1, group_size)
        
        means = rewards.mean(axis=1, keepdims=True)
        stds = rewards.std(axis=1, keepdims=True)
        advantages = (rewards - means) / (stds + eps)
        advantages = advantages.flatten().tolist()
        for idx, item in enumerate(data):
            for seq in item['seqs']:
                seq['advantage'] = advantages[idx]
                seq['seq_per_rollout'] = len(item['seqs'])
        return data

    def compute_threads_penalty(self, seq_token_lengths, seq_strs):
        # if the current string is to instantiate new threads and if the number of threads exceeds the max allowed threads, we will penalize it
        if "max_allowed_threads" not in self.config.training:
            return torch.zeros_like(seq_token_lengths)
        
        max_allowed_threads = self.config.training.max_allowed_threads
        thread_exceeded_nums = []

        n_threads = []
        
        # Ensure we process all prefixes and return a value for each
        for seq_str in seq_strs:
            sub_searches = re.findall(r'<Start Sub Search (\d+) at level \d+>', seq_str)
            if sub_searches:
                n_threads.append(max(int(num) for num in sub_searches))
                # Get the highest sub-search number which indicates total threads
                max_thread_num = max(int(num) for num in sub_searches)
                thread_exceeded_nums.append(max_thread_num - max_allowed_threads if max_thread_num > max_allowed_threads else 0)
            else:
                # If no sub-searches found, append 0 penalty
                thread_exceeded_nums.append(0)
            
        # Convert to tensor with same device as seq_token_lengths
        thread_counts_tensor = torch.tensor(thread_exceeded_nums, device=seq_token_lengths.device)
        # Apply penalty based on thread count
        return -self.config.training.get("thread_penalty", 0.1) * thread_counts_tensor, sum(n_threads) / len(n_threads) if len(n_threads) > 0 else 0.

    def compute_length_penalty(self, seq_token_lengths, prefixes=None):
        """Compute length penalty for sequences.
        Args:
            seq_token_lengths: `(torch.Tensor)`
                shape: (bs, )
            prefixes: `(list)`
                length: (bs, )
                """
        
        if self.config.training.length_penalty_type == "one-side":
            # If the length is too short, we will not penalize it (clamp will set the value to 0).
            # If the length is too long, we will penalize it.
            return -self.config.training.length_penalty * (torch.clamp(seq_token_lengths - self.config.training.target_length, min=0) / self.config.training.target_length) ** self.config.training.length_penalty_power
        elif self.config.training.length_penalty_type == "two-side":
            # If the length is too short, we will penalize it.
            # If the length is too long, we will penalize it.
            return -self.config.training.length_penalty * (torch.abs(seq_token_lengths - self.config.training.target_length) / self.config.training.target_length) ** self.config.training.length_penalty_power
        elif self.config.training.length_penalty_type == "two-side-token-cond":
            # Here we assume bos is <s>.
            assert all([prefix.startswith("<s>Token Budget: ") for prefix in prefixes]), f"All prefixes must start with '<s>Token Budget: ', but got {prefixes}"
            target_lengths = [int(prefix.removeprefix("<s>Token Budget: ").split(" ")[0]) for prefix in prefixes] 
            target_lengths = torch.tensor(target_lengths, device=seq_token_lengths.device)
            # print(f"target_lengths: {target_lengths}")
            # print(f"seq_token_lengths: {seq_token_lengths}")
            assert getattr(self.config.training, "length_penalty_target_length_scale", 1.0) == 1.0, f"length_penalty_target_length_scale must be 1.0 for two-side-token-cond (not implemented for scaling)"
            return -self.config.training.length_penalty * (torch.abs(seq_token_lengths - target_lengths) / target_lengths) ** self.config.training.length_penalty_power
        elif self.config.training.length_penalty_type == "one-side-token-cond":
            # Here we assume bos is <s>.
            assert all([prefix.startswith("<s>Token Budget: ") for prefix in prefixes]), f"All prefixes must start with '<s>Token Budget: ', but got {prefixes}"
            target_lengths = [int(prefix.removeprefix("<s>Token Budget: ").split(" ")[0]) for prefix in prefixes] 
            target_lengths = torch.tensor(target_lengths, device=seq_token_lengths.device)
            target_lengths = target_lengths * getattr(self.config.training, "length_penalty_target_length_scale", 1.0)
            # print(f"target_lengths: {target_lengths}")
            # print(f"seq_token_lengths: {seq_token_lengths}")
            return -self.config.training.length_penalty * (torch.clamp(seq_token_lengths - target_lengths, min=0) / target_lengths) ** self.config.training.length_penalty_power
        else:
            raise NotImplementedError(f"Length penalty type {self.config.training.length_penalty_type} not implemented")

    # def compute_valid_mask(self, seqs):
    #     """Compute valid mask for sequences without computing log probabilities."""
    #     # Prepare batches
    #     all_prefixes = []
    #     all_full_texts = []
    #     for seq in seqs:
    #         all_prefixes.append(seq['prefix'])
    #         all_full_texts.append(seq['prefix'] + seq['output'])
        
    #     # Tokenize prefixes and full texts
    #     prefix_tokens = self.tokenizer(all_prefixes, padding=True, return_tensors="pt").to("cuda:0")
    #     full_tokens = self.tokenizer(all_full_texts, padding=True, return_tensors="pt").to("cuda:0")
    #     targets = full_tokens.input_ids[:, 1:]  # Shift right by 1
        
    #     # Get prefix lengths for each sequence in batch
    #     prefix_lengths = prefix_tokens.input_ids.ne(self.tokenizer.pad_token_id).sum(1)
        
    #     # Create a mask for valid positions
    #     batch_seq_len = targets.shape[1]
    #     range_tensor = torch.arange(batch_seq_len, device=targets.device).unsqueeze(0).expand(prefix_lengths.size(0), -1)
    #     valid_mask = (range_tensor >= (prefix_lengths - 1).unsqueeze(1)) & (targets.ne(self.tokenizer.pad_token_id))
        
    #     # Clean up
    #     # del prefix_tokens
    #     # del full_tokens
    #     # del range_tensor
        
    #     return valid_mask

    def _run_inner_training_loop(self, train_seqs, old_log_probs, ref_log_probs, advantages, seq_per_rollout, batch_metrics):
        """Run multiple PPO/GRPO updates on the same batch of sequences."""
        for inner_step in range(self.config.training.inner_steps):
            if self.verbose:
                print(colored("📊 [Inner Training] Number of sequences in training batch: " + str(len(train_seqs)), "cyan"))
                print(colored("💾 [CUDA Memory] Pre-compute log prob memory usage: " + 
                      f"{torch.cuda.max_memory_allocated() / 1024**2:.2f}MB", "cyan"))
            
            # Check for NaNs in model before training step
            if self.check_model_nans(self.model, f"before_inner_step_{inner_step}"):
                print(colored(f"❌ NaNs detected in model before inner step {inner_step}", "red"))
                return
            
            chunk_size = self.config.training.grad_accum_chunk_size
            num_seqs = len(train_seqs)
            
            self.optimizer.zero_grad()
            accumulated_loss = 0
            accumulated_pg_loss = 0
            accumulated_pg_low_var_kl = 0
            accumulated_pg_clipfrac = 0
            policy_ratios = []
            seq_token_lengths = []
            length_penalty_values = []
            thread_penalty_values = []
            
            for start_idx in range(0, num_seqs, chunk_size):
                end_idx = min(start_idx + chunk_size, num_seqs)
                seq_chunk = train_seqs[start_idx:end_idx]
                log_probs_chunk, log_probs_valid_mask, seq_token_lengths_chunk = self.compute_log_prob(seq_chunk, self.model)
                
                if self.has_nan(log_probs_chunk, f"log_probs_chunk_{start_idx}"):
                    print(colored(f"❌ NaN detected in log probs chunk starting at index {start_idx}", "red"))
                    return
                
                seq_token_lengths.append(seq_token_lengths_chunk)
                
                old_log_probs_chunk = old_log_probs[start_idx:end_idx, :log_probs_chunk.shape[1]]
                if ref_log_probs is not None:
                    ref_log_probs_chunk = ref_log_probs[start_idx:end_idx, :log_probs_chunk.shape[1]]
                else:
                    ref_log_probs_chunk = None

                advantages_chunk = advantages[start_idx:end_idx, None]
                seq_per_rollout_chunk = seq_per_rollout[start_idx:end_idx]
                
                pg_loss, pg_low_var_kl, pg_clipfrac = self.compute_grpo_loss(
                    log_probs_chunk,
                    old_log_probs_chunk,
                    advantages_chunk,
                    log_probs_valid_mask,
                    seq_per_rollout_chunk,
                    clip_ratio=self.config.training.ppo_clip_ratio,
                    ref_log_probs=ref_log_probs_chunk
                )

                if pg_loss is None:  # NaN detected in loss computation
                    return

                loss = pg_loss + self.config.training.kl_beta * pg_low_var_kl
                loss = loss / len(train_seqs)
                
                if self.has_nan(loss, f"loss_chunk_{start_idx}"):
                    print(colored(f"❌ NaN detected in loss for chunk starting at index {start_idx}", "red"))
                    return
                
                loss.backward()
                
                # Check for NaNs in gradients after backward pass
                if self.check_model_nans(self.model, f"after_backward_chunk_{start_idx}"):
                    print(colored(f"❌ NaNs detected in gradients after backward pass for chunk {start_idx}", "red"))
                    return
                
                accumulated_loss += loss.item()
                accumulated_pg_loss += pg_loss.item()
                accumulated_pg_low_var_kl += pg_low_var_kl.item() if isinstance(pg_low_var_kl, torch.Tensor) else pg_low_var_kl
                accumulated_pg_clipfrac += pg_clipfrac.item()
                
                with torch.no_grad():
                    ratio_chunk = torch.exp(log_probs_chunk - old_log_probs_chunk).mean(dim=1)
                    policy_ratios.append(ratio_chunk)

            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            if torch.isnan(grad_norm):
                print(colored("❌ NaN detected in gradient norm", "red"))
                return
            
            self.optimizer.step()
            
            # Check for NaNs after optimizer step
            if self.check_model_nans(self.model, f"after_optimizer_step_{inner_step}"):
                print(colored(f"❌ NaNs detected in model after optimizer step {inner_step}", "red"))
                return

            # Compute mean policy ratio across all chunks
            policy_ratios = torch.cat(policy_ratios)
            mean_policy_ratio = policy_ratios.mean().item()
            seq_token_lengths = torch.cat(seq_token_lengths).float()
            mean_seq_token_lengths = seq_token_lengths.mean().item()
            
            if len(length_penalty_values) > 0:
                length_penalty_values = torch.cat(length_penalty_values)
                mean_length_penalty_values = length_penalty_values.mean().item()
            else:
                mean_length_penalty_values = 0.
            
            if len(thread_penalty_values) > 0:
                thread_penalty_values = torch.cat(thread_penalty_values)
                mean_thread_penalty_values = thread_penalty_values.mean().item()
            else:
                mean_thread_penalty_values = 0.
            
            # Log metrics for this inner step
            batch_metrics.update({
                f"loss_step_{inner_step}": accumulated_loss,
                f"policy_ratio_mean_step_{inner_step}": mean_policy_ratio,
                f"pg_loss_step_{inner_step}": accumulated_pg_loss,
                f"pg_low_var_kl_step_{inner_step}": accumulated_pg_low_var_kl,
                f"pg_clipfrac_step_{inner_step}": accumulated_pg_clipfrac,
                f"seq_token_lengths_mean_step_{inner_step}": mean_seq_token_lengths,
                f"length_penalty_values_mean_step_{inner_step}": mean_length_penalty_values,
                f"thread_penalty_values_mean_step_{inner_step}": mean_thread_penalty_values,
                f"grad_norm_step_{inner_step}": grad_norm.item(),
            })
            print(colored(f"   ↳ Inner step {inner_step}: loss = {accumulated_loss:.4f}, grad_norm = {grad_norm:.4f}", "green"))

    def validate(self):
        """Run validation on the validation set"""
        print(colored(f"\n📊 Running validation with temperature {self.config.rollout.eval_temperature}...", "blue"))
        
        # Sync weights to server for validation inference
        self.sync_weights_to_server()
        
        total_correct = 0
        total_samples = 0
        
        for val_batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            # Rollout for validation batch
            val_results = self.rollout_fn(
                self.config.model.server_url + '/v1',
                val_batch,
                self.bos_token,
                self.config.rollout.eval_temperature
            )
            
            # Count correct predictions
            batch_correct = sum(int(result['is_correct']) for result in val_results)
            total_correct += batch_correct
            total_samples += len(val_results)
        val_accuracy = total_correct / total_samples
        
        # Add number of validation samples to the log message
        print(colored(f"🎯 Validation Accuracy: {val_accuracy:.4f} (on {total_samples} samples)", "cyan"))
        return val_accuracy

    def train(self):
        step = 0
        train_iter = iter(self.train_dataloader)
        print(colored(f"\n🚀 Starting training with {self.config.training.num_steps} steps...", "cyan", attrs=["bold"]))
        
        # Initial validation
        val_accuracy = self.validate()
        wandb.log({"validation/accuracy": val_accuracy}, step=step)
        
        # Create progress bar for overall training
        pbar = tqdm(total=self.config.training.num_steps, desc="Training", 
                    unit="step", dynamic_ncols=True, leave=True)
        
        while step < self.config.training.num_steps:
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                train_batch = next(train_iter)
            
            # Update progress bar description with current step
            pbar.set_description(f"Training (Step {step+1}/{self.config.training.num_steps})")
            
            # Clear CUDA cache at the start of each step
            torch.cuda.empty_cache()
            
            # Sync model weights to server
            print(colored("\n🔄 Syncing weights to server...", "blue"))
            self.sync_weights_to_server()
            
            # Rollout Stage
            print(colored("🎲 Starting rollout phase...", "magenta"))
            # Repeat each prefix by rollout_group_size
            repeated_prefix = []
            for prefix in train_batch:
                repeated_prefix += [prefix] * self.config.rollout.group_size
            
            if self.verbose:
                print(colored(f"📝 [Rollout] Number of prefixes after repetition: {len(repeated_prefix)}", "cyan"))

            rollout_results = self.rollout_fn(
                self.config.model.server_url + '/v1',
                repeated_prefix,
                self.bos_token,
                self.config.rollout.sample_temperature
            )

            # Train Stage
            print(colored("🔨 Starting training phase...", "green"))
            # Compute GRPO advantages
            try:
                rollout_results = self.compute_grpo_advantage(rollout_results, self.config.rollout.group_size)
            except Exception as e:
                print(colored(f"❌ Error: {e}", "red"))
                from IPython import embed; embed()
                continue
            
            # Calculate average reward for logging
            rewards = [int(dp['is_correct']) for dp in rollout_results]
            avg_reward = sum(rewards) / len(rewards)
            batch_metrics = {"avg_reward": avg_reward}
            print(colored(f"💫 Average reward: {avg_reward:.4f}", "cyan"))

            train_seqs = []
            for item in rollout_results:
                train_seqs.extend(item['seqs'])
            
            if self.verbose:
                print(colored(f"📊 [Training Data] Sequences: {len(train_seqs)}, Rollout results: {len(rollout_results)}", "cyan"))

            # Drop sequences with zero advantage
            if not self.config.training.kl_beta > 0:
                # if no kl penalty + zero advantage sequences, there's no learning signal and we just skip this batch
                train_seqs = self.drop_zero_adv_seqs(train_seqs)
            if len(train_seqs) == 0:
                print(colored("‼️ No sequences with non-zero advantage, skipping batch", "red"))
                continue

            # Get old log probabilities
            with torch.no_grad():
                old_log_probs, _, _ = self.compute_log_prob(train_seqs, self.model, use_chunk=True)
                
                if self.config.training.kl_beta > 0:
                    ref_log_probs, _, _ = self.compute_log_prob(train_seqs, self.ref_model, use_chunk=True)
                else:
                    ref_log_probs = None

            # Get advantages and move to CUDA
            advantages = torch.tensor([seq['advantage'] for seq in train_seqs], device="cuda:0")
            seq_per_rollout = torch.tensor([seq['seq_per_rollout'] for seq in train_seqs], device="cuda:0")
            
            # Track metrics for this batch
            batch_metrics.update({
                "num_train_seqs": len(train_seqs),
                "mean_advantage": advantages.mean().item(),
                "max_advantage": advantages.max().item(),
                "min_advantage": advantages.min().item(),
            })

            # Inner training loop - multiple GRPO updates on same batch
            self._run_inner_training_loop(train_seqs, old_log_probs, ref_log_probs, advantages, seq_per_rollout, batch_metrics)

            # Clean up tensors after inner loop
            # del old_log_probs
            # del advantages
            # del seq_per_rollout
            # Log all metrics to wandb
            if step % self.config.logging.log_interval == 0:
                wandb.log(batch_metrics, step=step)
            
            # Run validation
            if (step+1) % self.config.training.validate_per_steps == 0:
                val_accuracy = self.validate()
                wandb.log({"validation/accuracy": val_accuracy}, step=step)
            
            # Update progress bar
            pbar.update(1)
            
            step += 1
            
            if self.config.training.save_steps > 0 and step % self.config.training.save_steps == 0:
                assert self.config.training.output_dir is not None, "output_dir must be specified if save_steps is set"
                output_dir = os.path.join(self.config.training.output_dir, self.wandb_name)
                save_dir = f"{output_dir}/global_step_{step}"
                os.makedirs(save_dir, exist_ok=True)
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                
        # Close progress bar
        pbar.close()
        
        if self.config.training.output_dir is not None:
            save_dir = os.path.join(self.config.training.output_dir, self.wandb_name)
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            
            # Push to hub
            if os.environ.get("HF_TOKEN", None) is not None:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=os.environ["HF_TOKEN"])
                    repo_type = "model"
                    repo_name = f"LM-Parallel/{self.wandb_name}"

                    api.create_repo(
                        repo_name, repo_type=repo_type, private=True, exist_ok=True)
                    api.upload_folder(
                        folder_path=save_dir,
                        repo_id=repo_name,
                        repo_type=repo_type,
                        ignore_patterns="global_step_*",
                    )
                    print(colored("✅ Successfully pushed checkpoint to hub", "green"))
                except Exception as e:
                    print(colored(f"❌ Error pushing to hub: {e}", "red"))

    def __del__(self, exit_code=0):
        """Cleanup distributed resources"""
        # Cleanup local weight group
        if hasattr(self, 'weight_group'):
            try:
                dist.destroy_process_group(self.weight_group)
                del self.weight_group
            except KeyboardInterrupt as e:
                raise e
            except Exception:
                pass

        # Terminate the server process
        if hasattr(self, 'server_process'):
            try:
                print(colored("🛑 Terminating server process...", "red"))
                self.server_process.terminate()
                self.server_process.wait()
                del self.server_process
            except KeyboardInterrupt as e:
                raise e
            except Exception:
                pass
        
        # Cleanup wandb
        wandb.finish(exit_code=exit_code)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None: 
    exit_code = 0
    try:
        trainer = GRPOTrainer(cfg)
        trainer.train()
    except Exception as e:
        print(colored(f"❌ Error: {e}", "red"))
        exit_code = 1
        raise e
    finally:
        trainer.__del__(exit_code=exit_code)

if __name__ == "__main__":
    main()
