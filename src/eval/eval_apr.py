import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import colored

from transformers import AutoTokenizer
from litellm import text_completion, APIError
from src.eval.parallel_infernce_utils import (
    get_search_result, 
    get_main_trace_after_sub_search, 
    get_subsearch_info, 
    check_solution
)

def parse_bool(value):
    if value.lower() in ['true', '1', 'yes', 'y']:
        return True
    elif value.lower() in ['false', '0', 'no', 'n']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")

# Example command for sglang:
# python -m sglang.launch_server  --served-model-name model --model-path Parallel-Reasoning/llama-apr_cond10_grpo --port 2346 --dp-size 8

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run APR inference experiments')
parser.add_argument('--model_name', type=str, default="llama-apr", 
                    help='Model name')
parser.add_argument("-d", "--data",type=str, default="data/val.json")
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument('--disable_parallel_inference', action='store_true', default=False,
                    help='Whether to use parallel inference')
parser.add_argument('--use_subcall_cond', action='store_true', default=False,
                    help='Whether to use subcall count conditioning instead of token count')
parser.add_argument('--max_workers', type=int, default=16,
                    help='Maximum number of workers for parallel inference')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum tokens for generation')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for generation')
parser.add_argument('--budget', type=int, default=None,
                    help='Budget for generation')
parser.add_argument('--output_dir', type=str, 
                    default=None,
                    help='Directory to save results. If None, defaults to "../results/{model_name}/val_apr"')

args = parser.parse_args()

# Set global variables from arguments
MODEL_NAME = args.model_name
PARALLEL_INFERENCE = not args.disable_parallel_inference
USE_SUBCALL_COND = args.use_subcall_cond
MAX_WORKERS = args.max_workers
TEMPERATURE = args.temperature
BUDGET = args.budget
SAVE_DIR = args.output_dir

# Load validation data
data_path = args.data
with open(data_path, "r") as f:
    val_data = json.load(f)

# Initialize tokenizer
ckpt = args.ckpt
tokenizer = AutoTokenizer.from_pretrained(ckpt)
bos_token = tokenizer.bos_token

# API configuration
api_base_url = "http://127.0.0.1:2346/v1"
api_key = "api_key"
model_name = "model"
max_tokens = args.max_tokens

# Configuration for text generation
ADD_ANGLE_BRACKETS = False
ADD_BOS = True

# BUDGET and TEMPERATURE will be handled outside in bash script

def add_angle_brackets(text):
    lines = text.split('\n')
    result_lines = []
    for line in lines:
        if '>' in line and '<' not in line:
            line = '<' + line
        result_lines.append(line)
    return '\n'.join(result_lines)

def generate(prefix, stop = [], temperature = 0.0):
    if ADD_BOS:
        prefix = bos_token + prefix
    result = text_completion(
        model=f"openai/{model_name}",
        prompt=prefix,
        api_base=api_base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    text = result['choices'][0]['text']
    complete_text = prefix + text
    complete_text = complete_text.replace(bos_token, ' ')
    if complete_text[0] == ' ':
        complete_text = complete_text[1:]
    if ADD_ANGLE_BRACKETS:
        complete_text = add_angle_brackets(complete_text)
    return complete_text, result

def add_all_calls(trace_dict):
    """Recursively collect all call traces from a trace_dict."""
    all_calls = []
    all_calls += trace_dict['main_calls']
    for sub in trace_dict.get('sub_calls', []):
        for sub_trace in sub:
            all_calls += add_all_calls(sub_trace)
    return all_calls

def calculate_tokens(item, ds_name="apr"):
    token_count = 0
    if 'apr' in ds_name:
        seqs = add_all_calls(item['trace_dict'])
        
        if len(seqs) > 1:
            # Find all sequences that start with "Moving to Node #0"
            root_seqs = [seq for seq in seqs if "Moving to Node #0\n" in seq]
            
            # Sort root sequences by length (shortest first)
            root_seqs = sorted(root_seqs, key=len)
            
            # Calculate total tokens without considering redundancy
            total_tokens = 0
            # Calculate total token count and track longest sequence in one pass
            longest_seq_tokens = 0
            for seq in seqs:
                tokens = tokenizer.encode(tokenizer.bos_token + seq + tokenizer.eos_token)
                total_tokens += len(tokens)
                longest_seq_tokens = max(longest_seq_tokens, len(tokens))
            
            # Calculate redundant tokens between root sequences
            redundant_tokens = 0
            if len(root_seqs) > 1:
                # Find common prefixes between each pair of sequences
                for i in range(len(root_seqs) - 1):
                    j = i + 1
                    seq1 = root_seqs[i]
                    seq2 = root_seqs[j]
                    
                    # Find common prefix
                    prefix_len = 0
                    for k in range(min(len(seq1), len(seq2))):
                        if seq1[k] == seq2[k]:
                            prefix_len += 1
                        else:
                            break
                    
                    if prefix_len > 0:
                        common_prefix = seq1[:prefix_len]
                        # Count tokens in this prefix
                        prefix_tokens = len(tokenizer.encode(common_prefix)) - 2  # Subtract BOS/EOS
                        redundant_tokens += max(0, prefix_tokens)
            
            # Final token count is total minus redundant
            token_count = total_tokens - redundant_tokens
            
            item['longest_seq_token_count'] = longest_seq_tokens
            sub_calls = [seq for seq in seqs if not "Moving to Node #0\n" in seq]
            item['avg_seq_token_count'] = token_count / (len(sub_calls) + 1)
        else:
            tokens = tokenizer.encode(tokenizer.bos_token + seqs[0] + tokenizer.eos_token)
            token_count = len(tokens)
            item['longest_seq_token_count'] = token_count
            item['avg_seq_token_count'] = token_count
    else:
        seq = item['search_path']
        tokens = tokenizer.encode(tokenizer.bos_token + seq + tokenizer.eos_token)
        token_count = len(tokens)

    item['token_count'] = token_count
    return item

    
def decode_trace(prefix, temperature):
    # we should never let the model generate <Sub Searches>
    # whenever it happens, we replace it with <Calling Sub Searches>
    while True:
        trace = generate(prefix, stop = ["<Sub Searches>"], temperature = temperature)
        prefix = trace[0]
        if trace[1].choices[0].matched_stop == "<Sub Searches>":
            prefix += "<Calling Sub Searches>"
        else:
            break
    prefix = trace[0]
    if prefix.split('\n')[-1] == "":
        # TODO: why is this happening?
        prefix = prefix[:-1]
    return prefix

if PARALLEL_INFERENCE:
    def batch_decode_trace(prefix_list, temperature):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks and store futures
            future_to_prefix = {executor.submit(decode_trace, prefix, temperature): prefix for prefix in prefix_list}
            
            # Initialize results list with the same length as prefix_list
            results = [None] * len(prefix_list)
            
            # As futures complete, store results in the correct order
            for future in as_completed(future_to_prefix):
                prefix = future_to_prefix[future]
                try:
                    result = future.result()
                    original_idx = prefix_list.index(prefix)
                    results[original_idx] = result
                except Exception as e:
                    print(f"Error processing prefix: {e}")
                    original_idx = prefix_list.index(prefix)
                    results[original_idx] = None
                    
        return results
else:
    def batch_decode_trace(prefix_list, temperature):
        # TODO make this parallel in the future
        result_list = []
        for prefix in prefix_list:
            result_list.append(decode_trace(prefix, temperature))
        return result_list
    
    
def is_calling_subsearch(trace):
    return "<End Calling Sub Searches>" in trace.split('\n')[-1]
def call_search(prefix, temperature, budget=None, avg_sub_call=False):
    try:
        trace_dict = {"main_calls": [], "sub_calls": []}
        trace = decode_trace(prefix, temperature)
        trace_dict["main_calls"].append(trace)
        while is_calling_subsearch(trace):
            sub_search_prefix_list, sub_search_nodes = get_subsearch_info(trace, budget, avg_sub_call)
            # call sub searchers
            # TODO: this assumes we only nest one level of sub searchers
            # In the future, we need to support nested sub searchers by recursion
            sub_search_traces = batch_decode_trace(sub_search_prefix_list, temperature)
            trace_dict["sub_calls"].append([])
            for sub_search_trace in sub_search_traces:
                trace_dict["sub_calls"][-1].append({"main_calls": [sub_search_trace]})
            sub_search_results = [get_search_result(trace) for trace in sub_search_traces]
            new_prefix = get_main_trace_after_sub_search(trace, sub_search_nodes, sub_search_results)
            trace = decode_trace(new_prefix, temperature)
            trace_dict["main_calls"].append(trace)
        return get_search_result(trace), trace_dict
    except APIError as e:
        print(f"Error at call_search: {e}")
        raise e
    except Exception as e:
        print(f"Error at call_search: {e}")
        return None, trace_dict


def get_prefix(dataset, idx, is_subcall_cond=False, use_budget=None):
    dp = dataset[idx]
    sub_call_budget = None
    prefix = f"Moving to Node #0\nCurrent State: {dp['target']}:{dp['nums']}, Operations: []"
    
    if is_subcall_cond:
        # Just use the provided budget for subcall conditioning
        assert use_budget is not None, "Subcall conditioning requires a budget"
        subcall_count = use_budget
        prefix = f"Sub Call Budget: {subcall_count} " + prefix
        
    if is_subcall_cond:
        # For subcall conditioning, we don't need to set sub_call_budget
        sub_call_budget = None
    elif sub_call_budget is not None:
        sub_call_budget = ((sub_call_budget - 1) // 512 + 1) * 512
    else:
        sub_call_budget = dp['token_count'] if 'token_count' in dp else None
    return prefix, sub_call_budget


def run_inference_experiment(
    model_name,
    val_data,
    n_tasks=1000,
    temperature=0.0,
    budget=None,
    use_subcall_cond=False,
    parallel_inference=True,
    max_workers=16,
    save_dir=None
):
    """
    Run an inference experiment with the specified configuration.
    
    Args:
        model_name (str): Name of the model to use
        val_data (list): Validation data
        n_tasks (int): Number of tasks to run
        temperature (float): Temperature for generation
        budget (int, optional): Token budget. If None, use default from data
        use_subcall_cond (bool): Whether to use subcall count conditioning
        parallel_inference (bool): Whether to use parallel inference
        max_workers (int): Maximum number of workers for parallel inference
        save_dir (str, optional): Directory to save results. If None, use default
    
    Returns:
        dict: Results dictionary with trajectories, trace_dicts, ratings, true_ratings
    """
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"{model_name.replace('/', '_')}_{n_tasks}_0_temp_{str(temperature).replace('.','_')}"
    if budget is not None:
        base_name += f"_budget_{budget}"
    
    save_path = os.path.join(save_dir, f"{base_name}.json")
    
    # Check if results already exist
    if os.path.exists(save_path):
        print(colored(f"Results already exist at {save_path}. Loading...", "green"))
        with open(save_path, 'r') as f:
            results = json.load(f)
        
        # Calculate and print metrics
        true_ratings = results["true_ratings"]
        ratings = results["ratings"]
        true_success_rate = np.mean(true_ratings)
        success_rate = np.mean(ratings)
        
        print(f"model: {model_name}, temperature: {temperature}, budget: {budget}\n"
              f"success_rate: {true_success_rate:.3f}, unverified_success_rate: {success_rate:.3f}")
        
        return results
    
    # If results don't exist, run the experiment
    success_count = []
    true_success_count = []
    logs = []

    pbar = tqdm(range(n_tasks))
    
    def process_sample(i):
        prefix, sub_call_budget = get_prefix(
            dataset=val_data, idx=i, 
            is_subcall_cond=use_subcall_cond, 
            use_budget=budget, 
        )
        out = call_search(prefix, temperature, budget=None, avg_sub_call=False)
        solution = out[0]
        true_success = check_solution(prefix, solution) if solution is not None else False
        return out, solution is not None, true_success

    if parallel_inference:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_sample, i) for i in range(n_tasks)]
            
            for i, future in enumerate(as_completed(futures)):
                out, success, true_success = future.result()
                logs.append(out)
                success_count.append(success)
                true_success_count.append(true_success)
                pbar.update(1)
                pbar.set_postfix({
                    'true_success_rate': f"{np.mean(true_success_count):.3f}", 
                    'success_rate': f"{np.mean(success_count):.3f}"
                })
    else:
        for i in pbar:
            out, success, true_success = process_sample(i)
            logs.append(out)
            success_count.append(success)
            true_success_count.append(true_success)
            pbar.update(1)
            pbar.set_postfix({
                'true_success_rate': f"{np.mean(true_success_count):.3f}", 
                'success_rate': f"{np.mean(success_count):.3f}"
            })
    
    # Prepare results
    trajectories = [x[0] if x[0] is not None else "" for x in logs]
    trace_dicts = [x[1] for x in logs]
    ratings = [1 if x else 0 for x in success_count]
    true_ratings = [1 if x else 0 for x in true_success_count]
    
    print(colored("Results Summary:", "cyan", attrs=["bold"]))
    print(colored(f"Model: ", "yellow") + colored(f"{model_name}", "green") + 
          colored(f", Temperature: ", "yellow") + colored(f"{temperature}", "green") + 
          colored(f", Budget: ", "yellow") + colored(f"{budget}", "green"))
    # print(colored(f"Unverified Success Rate: ", "yellow") + colored(f"{np.mean(success_count):.4f}", "green"))
    print(colored(f"Success Rate: ", "yellow") + colored(f"{np.mean(true_success_count):.4f}", "green"))

    results = {
        "trajectories": trajectories,
        "trace_dicts": trace_dicts,
        "ratings": ratings,
        "true_ratings": true_ratings
    }
    
    
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"{model_name}_{n_tasks}_0_temp_{str(temperature).replace('.','_')}"
    if budget is not None:
        if use_subcall_cond:
            base_name += f"_subcall_budget_{budget}"
        else:
            base_name += f"_budget_{budget}"
    
    # Save results
    print(f"Saving results to {save_path}")
    with open(save_path, 'w') as f:
        json.dump(results, f)
    
    return results


if __name__ == "__main__":    
    print(colored("Model name: ", "cyan", attrs=["bold"]) + colored(MODEL_NAME, "yellow"))
    print(colored("Data path: ", "cyan", attrs=["bold"]) + colored(data_path, "yellow"))
    print(colored("Checkpoint: ", "cyan", attrs=["bold"]) + colored(ckpt, "yellow"))
    
    print(colored("Conditions: ", "cyan", attrs=["bold"]) + 
          colored(f"use_subcall_cond={USE_SUBCALL_COND}", "yellow"))
    print(colored("Inference: ", "cyan", attrs=["bold"]) + 
          colored(f"parallel_inference={PARALLEL_INFERENCE}, "
                 f"max_workers={MAX_WORKERS}, "
                 f"max_tokens={max_tokens}", "yellow"))
    print(colored("Generation: ", "cyan", attrs=["bold"]) + 
          colored(f"temperature={TEMPERATURE}, budget={BUDGET}", "yellow"))

    # Set default save directory if not provided
    save_directory = SAVE_DIR if SAVE_DIR is not None else f"results/{MODEL_NAME}/val_apr"
    print(colored(f"save_dir: {save_directory}", "cyan"))

    run_inference_experiment(
        model_name=MODEL_NAME,
        val_data=val_data,
        temperature=TEMPERATURE,
        budget=BUDGET,
        use_subcall_cond=USE_SUBCALL_COND,
        parallel_inference=PARALLEL_INFERENCE,
        max_workers=MAX_WORKERS,
        save_dir=save_directory
    )
