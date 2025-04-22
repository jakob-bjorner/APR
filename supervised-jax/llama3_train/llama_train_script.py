from typing import List, Dict, Any, Optional
import json
from functools import partial
import os
import collections
import tempfile

import tyro
import jax
from scalax.sharding import MeshShardingHelper, TreePathShardingRule
from flax.training.train_state import TrainState
import numpy as np
import itertools
import pickle as pkl
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import optax
from jax.sharding import PartitionSpec as PS

from llama_train.utils import (
    load_checkpoint, open_with_bucket, save_checkpoint,
    get_float_dtype_by_name, get_weight_decay_mask,
    cross_entropy_loss_and_accuracy, global_norm,
    average_metrics, WandbLogger, delete_with_bucket,
    jax_distributed_initalize, jax_distributed_barrier
)
from llama_train.llama3 import (
    LLaMAConfig, LLAMA_STANDARD_CONFIGS,
    FlaxLLaMAForCausalLM, download_openllama_easylm,
)
from llama_train.optimizer import load_adamw_optimizer, load_palm_optimizer

def process_pretrain_example(
    seq: str,
    max_length: int,
    tokenizer: AutoTokenizer,
):
    tokenization = tokenizer(
        [tokenizer.bos_token+seq+tokenizer.eos_token],
        padding='max_length',
        truncation=True,
        max_length=max_length+1,
        return_tensors='np',
    )

    input_ids = tokenization.input_ids[:, :-1]
    target_ids = tokenization.input_ids[:, 1:]
    attention_mask = tokenization.attention_mask[:, :-1]
    position_ids = np.maximum(np.cumsum(attention_mask, axis=-1) - 1, 0)
    loss_masks = attention_mask

    batch_items = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        target_ids=target_ids,
        loss_masks=loss_masks,
    )
    return batch_items


def checkpointer(
    path: str,
    train_state: Any,
    config: Any,
    gather_fns: Any,
    metadata: Any=None,
    save_optimizer_state: bool=False,
    save_float_dtype: str='bf16',
    active=True,
):
    if not path.startswith('gcs://'):
        os.makedirs(path, exist_ok=True)
    if save_optimizer_state:
        checkpoint_state = train_state
        if not active:
            checkpoint_path = '/dev/null'
        else:
            checkpoint_path = os.path.join(path, 'train_state.msgpack')
        checkpoint_gather_fns = gather_fns
    else:
        checkpoint_state = train_state.params
        if not active:
            checkpoint_path = '/dev/null'
        else:
            checkpoint_path = os.path.join(path, 'params.msgpack')
        checkpoint_gather_fns = gather_fns.params
    metadata_path = os.path.join(path, 'metadata.pkl')
    config_path = os.path.join(path, 'config.json')
    
    save_checkpoint(
        checkpoint_state,
        checkpoint_path,
        gather_fns=checkpoint_gather_fns,
        float_dtype=save_float_dtype,
    )
    if active:
        with open_with_bucket(metadata_path, 'wb') as f:
            pkl.dump(metadata, f)
        with open_with_bucket(config_path, 'w') as f:
            json.dump(config, f)

def main(
    load_model: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: Optional[str],
    sharding: str,
    num_train_steps: int,
    max_length: int,
    bsize: int,
    log_freq: int,
    num_eval_steps: int,
    save_model_freq: int,
    wandb_project: str,
    param_dtype: str='fp32',
    activation_dtype: str='fp32',
    optim_config: str='adamw:{}',
    logger_config: str='{}',
    checkpointer_config: str='{}',
    model_config_override: str='{}',
    inputs_tokenizer_override: str='{}',
    outputs_tokenizer_override: str='{}',
    jax_distributed_initalize_config: str='{}',
    save_initial_checkpoint: bool=False,
    log_initial_step: bool=True,
    max_checkpoints: Optional[int]=None,
    eval_bsize: Optional[int]=None,
    physical_axis_splitting: bool=False,
    shuffle_train_data: bool=True,
    hf_repo_id: str='LM-Parallel/sample',
):
    args_dict = dict(locals())
    print(args_dict)
    sharding: List[int] = list(map(lambda x: int(x.strip()), sharding.split(',')))
    if eval_bsize is None:
        eval_bsize = bsize

    param_dtype = get_float_dtype_by_name(param_dtype)
    activation_dtype = get_float_dtype_by_name(activation_dtype)
    
    logger_config: Dict[str, Any] = json.loads(logger_config)
    checkpointer_config: Dict[str, Any] = json.loads(checkpointer_config)
    model_config_override: Dict[str, Any] = json.loads(model_config_override)
    inputs_tokenizer_override: Dict[str, Any] = json.loads(inputs_tokenizer_override)
    outputs_tokenizer_override: Dict[str, Any] = json.loads(outputs_tokenizer_override)
    jax_distributed_initalize_config: Dict[str, Any] = json.loads(jax_distributed_initalize_config)

    jax_distributed_initalize(**jax_distributed_initalize_config)
    jax_distributed_barrier()

    if optim_config.startswith('adamw:'):
        optim_config = json.loads(optim_config[len('adamw:'):])
        optim_config['weight_decay_mask'] = get_weight_decay_mask(optim_config.pop('weight_decay_exclusions', tuple()))
        grad_accum_steps = optim_config.pop('grad_accum_steps', 1)
        optimizer, optimizer_info = load_adamw_optimizer(**optim_config)
    elif optim_config.startswith('palm:'):
        optim_config = json.loads(optim_config[len('palm:'):])
        optim_config['weight_decay_mask'] = get_weight_decay_mask(optim_config.pop('weight_decay_exclusions', tuple()))
        grad_accum_steps = optim_config.pop('grad_accum_steps', 1)
        optimizer, optimizer_info = load_palm_optimizer(**optim_config)
    else:
        raise ValueError(f'Unknown optimizer config: {optim_config}')
    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer,
            grad_accum_steps,
        )

    mesh = MeshShardingHelper(sharding, ['dp', 'fsdp', 'mp'], mesh_axis_splitting=physical_axis_splitting)  # Create a 3D mesh with data, fsdp, and model parallelism axes
    with mesh.get_context():
        print('mesh:', mesh.mesh)
        print('loading model ...')

        if load_model.startswith('paths:'):
            model_paths = json.loads(load_model[len('paths:'):])
            if not ('remove_dict_prefix' in model_paths):
                model_paths['remove_dict_prefix'] = None
        else:
            raise ValueError(f'Unknown model info type: {load_model}')
        
        config_is_temp = False
        if 'config' in model_paths and model_paths['config'].startswith('gcs://'):
            temp_file = tempfile.NamedTemporaryFile('wb', delete=False)
            with open_with_bucket(model_paths['config'], 'rb') as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths['config'] = temp_file.name
            config_is_temp = True
        
        if 'config' in model_paths:
            config = LLaMAConfig.from_pretrained(model_paths['config'], **model_config_override)
        elif 'default_config_name' in model_paths:
            config = LLaMAConfig(**LLAMA_STANDARD_CONFIGS[model_paths['default_config_name']], **model_config_override)
        else:
            config = LLaMAConfig(**model_config_override)
        
        if config_is_temp:
            os.remove(model_paths['config'])

        model = FlaxLLaMAForCausalLM(config, dtype=activation_dtype, _do_init=False, param_dtype=param_dtype, input_shape=(bsize, 1024))
        # TODO: embedding dim is hardcoded to 1024, it's
        
        tokenizer_is_temp = False
        if model_paths['tokenizer'].startswith('gcs://'):
            temp_file = tempfile.NamedTemporaryFile('wb', delete=False)
            with open_with_bucket(model_paths['tokenizer'], 'rb') as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths['tokenizer'] = temp_file.name
            tokenizer_is_temp = True
        
        tokenizer_kwargs = dict(
            truncation_side='right',
            padding_side='right',
        )
        tokenizer_kwargs.update(outputs_tokenizer_override)
        tokenizer = AutoTokenizer.from_pretrained(model_paths['tokenizer'], **tokenizer_kwargs)
        tokenizer.add_special_tokens({'pad_token': tokenizer.convert_ids_to_tokens(config.pad_token_id)})
        
        if tokenizer_is_temp:
            os.remove(model_paths['tokenizer'])

        sharding_rules = TreePathShardingRule(*config.get_partition_rules())

        @partial(
            mesh.sjit,
            in_shardings=(sharding_rules,),
            out_shardings=sharding_rules,
        )
        def create_train_state_from_params(params):
            return TrainState.create(params=params, tx=optimizer, apply_fn=None)
        
        @partial(
            mesh.sjit,
            in_shardings=(PS(),),
            out_shardings=sharding_rules,
        )
        def init_fn(rng):
            params = model.init_weights(rng, (bsize, 1024))
            return create_train_state_from_params(params)

        train_state_shape = jax.eval_shape(lambda: init_fn(jax.random.PRNGKey(0)))
        shard_train_state_fns, gather_train_state_fns = mesh.make_shard_and_gather_fns(train_state_shape, sharding_rules)

        if 'params' in model_paths:
            train_state = create_train_state_from_params(load_checkpoint(
                model_paths['params'],
                shard_fns=shard_train_state_fns.params,
                remove_dict_prefix=model_paths['remove_dict_prefix'],
                convert_to_dtypes=jax.tree_util.tree_map(lambda x: x.dtype, train_state_shape.params),
            ))
        elif 'train_state' in model_paths:
            train_state = load_checkpoint(
                model_paths['train_state'],
                shard_fns=shard_train_state_fns,
                remove_dict_prefix=model_paths['remove_dict_prefix'],
                convert_to_dtypes=jax.tree_util.tree_map(lambda x: x.dtype, train_state_shape),
            )
        else:
            train_state = init_fn(jax.random.PRNGKey(0))
        
        print(model)
        print('model loaded.')

        @partial(
            mesh.sjit,
            in_shardings=(sharding_rules, PS(),PS()),
            out_shardings=(sharding_rules, PS()),
            args_sharding_constraint=(sharding_rules, None, PS(('dp', 'fsdp'))),
            donate_argnums=(0,),
        )
        def train_step(train_state, rng, batch):
            def loss_and_accuracy(params):
                logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch['position_ids'],
                    params=params,
                    dropout_rng=rng,
                    train=True,
                ).logits
                return cross_entropy_loss_and_accuracy(
                    logits, batch['target_ids'], batch['loss_masks'],
                )
            # print("start training...")
            grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (loss, accuracy), grads = grad_fn(train_state.params)
            # print(f"loss: {loss}, accuracy: {accuracy}")
            train_state = train_state.apply_gradients(grads=grads)
            # print("gradients applied.")
            metrics = dict(
                loss=loss,
                accuracy=accuracy,
                learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
                gradient_norm=global_norm(grads),
                param_norm=global_norm(train_state.params),
            )
            return train_state, metrics
        
        @partial(
            mesh.sjit,
            in_shardings=(sharding_rules, PS()),
            out_shardings=PS(),
            args_sharding_constraint=(sharding_rules, PS(('dp', 'fsdp'))),
        )
        def eval_step(params, batch):
            logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'],
                params=params,
                train=False,
            ).logits
            loss, accuracy = cross_entropy_loss_and_accuracy(
                logits, batch['target_ids'], batch['loss_masks'],
            )
            metrics = dict(
                eval_loss=loss,
                eval_accuracy=accuracy,
            )
            return metrics
        
        print('loading data ...')
        if train_data_path.endswith('jsonl'):
            train_examples = []
            with open_with_bucket(train_data_path, 'r') as f:
                for line in f:
                    train_examples.append(json.loads(line.strip()))
        else:
            with open_with_bucket(train_data_path, 'r') as f:
                train_examples = json.load(f)
        if eval_data_path.endswith('jsonl'):
            eval_examples = []
            with open_with_bucket(eval_data_path, 'r') as f:
                for line in f:
                    eval_examples.append(json.loads(line.strip()))
        else:
            with open_with_bucket(eval_data_path, 'r') as f:
                eval_examples = json.load(f)
        print('done.')

        def data_iterable(data_items, rng, bsize, shuffle=True, loop=True):
            while True:
                with jax.default_device(jax.devices('cpu')[0]):
                    idxs = []
                    for _ in range((bsize + (len(data_items) - 1)) // len(data_items)):
                        if shuffle:
                            rng, subrng = jax.random.split(rng)
                            curr_idxs = jax.random.permutation(subrng, np.arange(len(data_items)))
                            idxs.extend(curr_idxs.tolist())
                        else:
                            curr_idxs = np.arange(len(data_items))
                            idxs.extend(curr_idxs.tolist())
                    idxs = np.asarray(idxs)
                for batch_idx in range(len(idxs) // bsize):
                    batch_idxs = idxs[batch_idx*bsize:(batch_idx+1)*bsize]
                    batch_examples = [data_items[idx] for idx in batch_idxs]
                    processed_batch_examples = []
                    for example in batch_examples:
                        processed_batch_examples.append(process_pretrain_example(
                            example,
                            max_length,
                            tokenizer,
                        ))
                    batch = dict()
                    for key in processed_batch_examples[0]:
                        batch[key] = np.concatenate([example[key] for example in processed_batch_examples], axis=0)
                    yield batch
                if not loop:
                    break

        if 'enable' not in logger_config:
            logger_config['enable'] = (jax.process_index() == 0)
        if 'config_to_log' in logger_config:
            logger_config['config_to_log'].update(args_dict)
        else:
            logger_config['config_to_log'] = args_dict
        logger = WandbLogger(
            wandb_project,
            output_dir=output_dir,
            **logger_config,
        )
        print('wandb logger initialized.')

        checkpoint_queue = collections.deque()
        
        def _save_checkpoint(
            train_state,
            step,
        ):
            old_step = None
            if (max_checkpoints is not None) and (len(checkpoint_queue) >= max_checkpoints):
                old_step = checkpoint_queue.popleft()
            if logger.can_save():
                print(f'saving checkpoint at step {step} ...')
                # delete old checkpoint if max checkpoints is reached
                if old_step is not None:
                    old_path = os.path.join(logger.output_dir, 'checkpoints', f'step_{old_step}')
                    delete_with_bucket(old_path, recursive=True)
            
            metadata = dict(
                step=step,
                args_dict=args_dict,
            )

            checkpointer(
                path=os.path.join(logger.output_dir, 'checkpoints', f'step_{step}'),
                train_state=train_state,
                config=config.to_dict(),
                gather_fns=gather_train_state_fns,
                metadata=metadata,
                active=logger.can_save(),
                **checkpointer_config,
            )

            checkpoint_queue.append(step)

            if logger.can_save():
                print('saved.')
        
        if save_initial_checkpoint:
            _save_checkpoint(train_state, 0)
        
        rng = jax.random.PRNGKey(0)
        
        rng, eval_iterable_rng = jax.random.split(rng)
        rng, subrng = jax.random.split(rng)
        train_iterable = data_iterable(train_examples, subrng, bsize, shuffle=shuffle_train_data, loop=True)
        for step, train_batch in tqdm(itertools.islice(enumerate(train_iterable), num_train_steps), total=num_train_steps):
            rng, subrng = jax.random.split(rng)
            train_state, metrics = train_step(train_state, subrng, train_batch)
            # print(f"step {step} metrics: {metrics}")
            if log_freq > 0 and ((step+1) % log_freq == 0 or (log_initial_step and step == 0)):
                if num_eval_steps > 0:
                    eval_metric_list = []
                    eval_iterable = data_iterable(eval_examples, eval_iterable_rng, eval_bsize, shuffle=True, loop=False)
                    for eval_batch in itertools.islice(eval_iterable, num_eval_steps):
                        eval_metric_list.append(eval_step(train_state.params, eval_batch))
                    metrics.update(average_metrics(jax.device_get(eval_metric_list)))
                log_metrics = {"step": step+1}
                log_metrics.update(metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                print(log_metrics)
            
            if save_model_freq > 0 and (step+1) % save_model_freq == 0:
                _save_checkpoint(train_state, step+1)
        
        if save_model_freq > 0 and (num_train_steps not in checkpoint_queue):
            _save_checkpoint(train_state, num_train_steps)
        
        jax_distributed_barrier()
        logger.finish()
        jax_distributed_barrier()

        # Only have the first worker push to hub to avoid conflicts
        if jax.process_index() == 0 and logger.can_save():
            import shutil
            print("First worker copying final checkpoint to hub...")
            
            # Create temp directory for checkpoint
            temp_dir = tempfile.mkdtemp()
            final_ckpt_path = os.path.join(logger.output_dir, 'checkpoints', f'step_{num_train_steps}')
            
            # Copy checkpoint files to temp dir
            if final_ckpt_path.startswith('gcs://'):
                with open_with_bucket(os.path.join(final_ckpt_path, 'params.msgpack'), 'rb') as f:
                    with open(os.path.join(temp_dir, 'params.msgpack'), 'wb') as f_out:
                        f_out.write(f.read())
                with open_with_bucket(os.path.join(final_ckpt_path, 'config.json'), 'rb') as f:
                    with open(os.path.join(temp_dir, 'config.json'), 'wb') as f_out:
                        f_out.write(f.read())
            else:
                shutil.copy2(os.path.join(final_ckpt_path, 'params.msgpack'), temp_dir)
                shutil.copy2(os.path.join(final_ckpt_path, 'config.json'), temp_dir)

            # Push to hub
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_type = "model"
                repo_name = hf_repo_id

                api.create_repo(repo_name, repo_type=repo_type, private=False, exist_ok=True)
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_name,
                    repo_type=repo_type
                )
                print("Successfully pushed checkpoint to hub")
            except Exception as e:
                print(f"Error pushing to hub: {e}")
            finally:
                # Cleanup temp directory
                shutil.rmtree(temp_dir)
if __name__ == "__main__":
    tyro.cli(main)
