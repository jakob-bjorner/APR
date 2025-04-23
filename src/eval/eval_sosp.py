import os, sys
import json
import random
import argparse
import tqdm
import numpy as np
from transformers import AutoTokenizer
import sglang as sgl
from src.countdown_utils import *
from src.utils import seed_everything
from termcolor import colored
import re

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument("--tknz", type=str, help="path to tokenizer")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples for best-of-n evaluation")
parser.add_argument("-d", "--data",type=str, default="data/val.json")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_tokens", type=int, default=131072)
parser.add_argument("--gens", type=int, default=1)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="results/")
parser.add_argument("--skip-save-results", action="store_true", help="Skip saving results to file")
parser.add_argument("--num_token_cond", action="store_true", help="Use num token condition")
parser.add_argument("--budget", type=int, default=None, help="Set a fixed budget condition")

# Calculate mean token count
def count_tokens(text, tokenizer):
    cleaned_text = re.sub(r'^.*?Current State:', 'Current State:', text)
    # Add offset (so it matches the token count in the train/val data json)
    return len(tokenizer.encode(cleaned_text)) + 2
    
def eval_ll(args):
    """
    Evaluate the model on the data using sglang
    """
    with open(args.data, "r") as json_file:
        raw_data = json.load(json_file)
        
    if not args.skip_save_results:
        output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.data))[0])
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{args.ckpt.split('outputs/')[-1].replace('/','_')}_temp_{str(args.temperature).replace('.','_')}"
        if args.budget is not None:
            base_name += f"_budget_{args.budget}"
        if args.n_samples > 1:
            base_name += f"_n_samples_{args.n_samples}"
        results_file = os.path.join(output_dir, f"{base_name}.json")
        print(f"Results file: {results_file}")
        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            with open(results_file, 'r') as f:
                results = json.load(f)
            pred_ratings = results['ratings']
            
            # Initialize tokenizer for token counting
            tokenizer = AutoTokenizer.from_pretrained(args.tknz) if args.tknz else AutoTokenizer.from_pretrained(args.ckpt)
            
            token_counts = [count_tokens(pred, tokenizer) for pred in results['trajectories']]
            mean_token_count = sum(token_counts) / len(token_counts)
            
            print(colored("Results Summary:", "cyan", attrs=["bold"]))
            print(colored(f"Mean token count: ", "yellow") + colored(f"{mean_token_count:.2f}", "green"))
            print(colored(f"Model Accuracy: ", "yellow") + colored(f"{np.mean([r > 0 for r in pred_ratings]):.4f}", "green"))
            sys.exit(0)
    
    # Initialize sglang engine
    llm = sgl.Engine(
        model_path=args.ckpt, 
        tokenizer_path=args.tknz if args.tknz else args.ckpt,
        allow_auto_truncate=True,
        # log_level='warning',
        # tp_size=args.num_gpus, 
        # dp is slightly faster than tp due to the small model size; also sgl gpt2 has bug with tp
        dp_size=args.num_gpus,
    )
    
    # Prepare prompts
    tokenizer = AutoTokenizer.from_pretrained(args.tknz) if args.tknz else AutoTokenizer.from_pretrained(args.ckpt)
    if args.num_token_cond or args.budget:
        def get_token_budget(sample):
            if args.budget is not None:
                return args.budget
            # Define token budget bins (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
            budget_bins = list(range(512, 4096+1, 512))
            token_count = sample['token_count']
            # Find the appropriate budget bin
            budget = 4096 if token_count > 4096 else \
                next((bin_value for bin_value in budget_bins if token_count <= bin_value), 4096)
            return budget
        
        test_prompts = [
            f"{tokenizer.bos_token}Token Budget: {get_token_budget(sample)} "
            f"Current State: {sample['target']}:{sample['nums']}, Operations: []" 
            for sample in raw_data
        ]
    else:
        test_prompts = [
            f"{tokenizer.bos_token}Current State: {sample['target']}:{sample['nums']}, Operations: []" 
            for sample in raw_data
        ]
    len_nums = [len(sample['nums']) for sample in raw_data]
    data = [d for d, l in zip(test_prompts, len_nums) if l == 4]
    
    # Set up sampling parameters
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "top_k": 50,
    }

    # Process in batches
    batch_size = args.batch_size * args.num_gpus
    
    if args.n_samples > 1:
        # Initialize data structures for multiple samples
        all_sample_predictions = []
        all_sample_ratings = []
        all_sample_reasons = []
        
        # Run inference n_samples times
        for sample_idx in range(args.n_samples):
            print(colored(f"Running sample {sample_idx+1}/{args.n_samples}", "cyan", attrs=["bold"]))
            predictions = []
            
            for b in tqdm.trange(0, len(data), batch_size):
                batch = data[b:min(b+batch_size, len(data))]
                
                if args.gens == 1:
                    # Generate outputs
                    outputs = llm.generate(batch, sampling_params)
                    # Combine prompts with generated text
                    batch_predictions = [prompt + output['text'] for prompt, output in zip(batch, outputs)]
                    predictions.extend(batch_predictions)
                else:
                    assert args.temperature > 0.0, "Temperature must be greater than 0 for sampling"
                    all_outputs = []
                    all_ratings = []
                    
                    # Generate multiple times for each prompt
                    for _ in range(args.gens):
                        outputs = llm.generate(batch, sampling_params)
                        # Combine prompts with generated text for each generation
                        output_texts = [prompt + output['text'] for prompt, output in zip(batch, outputs)]
                        # Get rating for each output
                        ratings = [metric_fn(ot, mode="sft")[0] for ot in output_texts]
                        all_ratings.append(ratings)
                        all_outputs.append(output_texts)
                    
                    # Convert to numpy array for easier processing
                    all_ratings = np.array(all_ratings)
                    print(all_ratings)
                    print(f"average rating", np.mean(all_ratings))
                    
                    # Get the best output for each prompt
                    max_ratings = np.argmax(all_ratings, axis=0)
                    max_rating_vals = np.max(all_ratings, axis=0)
                    print(f"max ratings", np.mean(max_rating_vals))
                    
                    # Select the best outputs
                    batch_predictions = [all_outputs[max_r][i] for i, max_r in enumerate(max_ratings)]
                    predictions.extend(batch_predictions)
            
            # Rate outputs for this sample
            pred_ratings = []
            pred_reasons = []
            for i, pred in enumerate(predictions):
                rating, reason = metric_fn(pred, mode="sft")
                pred_ratings.append(rating)
                pred_reasons.append(reason)
            
            # Store results for this sample
            all_sample_predictions.append(predictions)
            all_sample_ratings.append(pred_ratings)
            all_sample_reasons.append(pred_reasons)
        
        all_sample_ratings_array = np.array(all_sample_ratings)
        binary_correctness = all_sample_ratings_array > 0

        # Print results
        print(colored("Results Summary:", "cyan", attrs=["bold"]))
        print(colored(f"Number of samples: ", "yellow") + colored(f"{args.n_samples}", "green"))
        print(colored(f"Individual sample accuracies: ", "yellow") + 
              colored(f"{[np.mean(binary_correctness[i]) for i in range(args.n_samples)]}", "green"))
        # TODO: cons@n and pass@n
        
        # Save results
        if not args.skip_save_results:
            with open(results_file, "w") as f:
                json.dump({
                    "n_samples": args.n_samples,
                    "individual_sample_accuracies": [float(np.mean(binary_correctness[i])) for i in range(args.n_samples)],
                    "sample_trajectories": all_sample_predictions,
                    "sample_ratings": all_sample_ratings_array.tolist(),
                    "sample_reasons": all_sample_reasons,
                }, f, indent=4)
    
    else:
        # Original single-sample code
        predictions = []
        pred_ratings = []
        pred_reasons = []
        
        for b in tqdm.trange(0, len(data), batch_size):
            batch = data[b:min(b+batch_size, len(data))]
            
            if args.gens == 1:
                # Generate outputs
                outputs = llm.generate(batch, sampling_params)
                # Combine prompts with generated text
                batch_predictions = [prompt + output['text'] for prompt, output in zip(batch, outputs)]
                predictions.extend(batch_predictions)
            else:
                assert args.temperature > 0.0, "Temperature must be greater than 0 for sampling"
                all_outputs = []
                all_ratings = []
                
                # Generate multiple times for each prompt
                for _ in range(args.gens):
                    outputs = llm.generate(batch, sampling_params)
                    # Combine prompts with generated text for each generation
                    output_texts = [prompt + output['text'] for prompt, output in zip(batch, outputs)]
                    # Get rating for each output
                    ratings = [metric_fn(ot, mode="sft")[0] for ot in output_texts]
                    all_ratings.append(ratings)
                    all_outputs.append(output_texts)
                
                # Convert to numpy array for easier processing
                all_ratings = np.array(all_ratings)
                print(all_ratings)
                print(f"average rating", np.mean(all_ratings))
                
                # Get the best output for each prompt
                max_ratings = np.argmax(all_ratings, axis=0)
                max_rating_vals = np.max(all_ratings, axis=0)
                print(f"max ratings", np.mean(max_rating_vals))
                
                # Select the best outputs
                batch_predictions = [all_outputs[max_r][i] for i, max_r in enumerate(max_ratings)]
                predictions.extend(batch_predictions)

        # Rate outputs
        true_rating = []
        for i, pred in enumerate(predictions):
            rating, reason = metric_fn(pred, mode="sft")
            tr, _ = metric_fn(f"{raw_data[i]['search_path']}", mode="sft")
            pred_ratings.append(rating)
            true_rating.append(tr)
            pred_reasons.append(reason)

        token_counts = [count_tokens(pred, tokenizer) for pred in predictions]
        mean_token_count = sum(token_counts) / len(token_counts)

        # Print results
        pred_ratings = np.array(pred_ratings)    
        print(colored("Results Summary:", "cyan", attrs=["bold"]))
        print(colored(f"Mean token count: ", "yellow") + colored(f"{mean_token_count:.2f}", "green"))
        print(colored(f"Model Accuracy: ", "yellow") + colored(f"{np.mean([r > 0 for r in pred_ratings]):.4f}", "green"))
        print(colored(f"Original Symbolic Solver Accuracy: ", "yellow") + colored(f"{np.mean([r > 0 for r in true_rating]):.4f}", "green"))

        # Save results
        if not args.skip_save_results:
            with open(results_file, "w") as f:
                json.dump({
                    "trajectories": predictions,
                    "ratings": pred_ratings.tolist(),
                    "reasons": pred_reasons,
                    "token_counts": token_counts,
                    "mean_token_count": mean_token_count
                }, f, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.seed)
    
    print(colored("Evaluating model: ", "cyan", attrs=["bold"]) + colored(args.ckpt, "yellow"))
    print(colored("Data file: ", "cyan", attrs=["bold"]) + colored(args.data, "yellow"))
    print(colored("Temperature: ", "cyan", attrs=["bold"]) + colored(args.temperature, "yellow"))
    print(colored("Number of GPUs: ", "cyan", attrs=["bold"]) + colored(args.num_gpus, "yellow"))
    if args.n_samples > 1:
        print(colored("Number of samples (best-of-n): ", "cyan", attrs=["bold"]) + colored(args.n_samples, "yellow"))
    if args.num_token_cond:
        print(colored("Using token condition", "cyan", attrs=["bold"]))
    if args.budget:
        print(colored("Using a fixed budget: ", "cyan", attrs=["bold"]) + colored(args.budget, "yellow"))
    # eval
    eval_ll(args)
