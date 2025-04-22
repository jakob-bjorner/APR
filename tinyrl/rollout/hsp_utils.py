"""
export CUDA_VISIBLE_DEVICES=7
python -m sglang.launch_server --model-path LM-Parallel/llama-hsp-v3  --host localhost --served-model-name model 
"""
import re
from transformers import AutoTokenizer
from litellm import text_completion
from concurrent.futures import ThreadPoolExecutor, as_completed
import sglang

# Initialize tokenizer
MODEL_NAME = "LM-Parallel/llama-hsp-v3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded for {MODEL_NAME}")

VERBOSE = False

def check_solution(prefix: str, solution: str) -> bool:
    """
    Parses the prefix and solution to verify if the solution actually
    solves the puzzle of reaching `target` from the given list of numbers.
    
    :param prefix: A line like:
        "Moving to Node #1\nCurrent State: 62:[5, 50, 79, 27], Operations: []"
    :param solution: The multiline string describing the step-by-step solution.
    :return: True if the solution's final result matches the target and
             all stated operations are valid. False otherwise.
    """
    # -----------------------------------------------------------------
    # 1. Parse the prefix to extract target and initial numbers
    # -----------------------------------------------------------------
    # Example prefix line to parse:
    # Current State: 62:[5, 50, 79, 27], Operations: []
    #
    # We'll look for something matching: 
    #   Current State: <target>:[<numbers>], ...
    prefix_pattern = r"Current State:\s*(\d+):\[(.*?)\]"    
    match = re.search(prefix_pattern, prefix)
    if not match:
        if VERBOSE:
            print("ERROR: Could not parse the prefix for target and numbers.")
        return False
    
    target_str, numbers_str = match.groups()
    target = int(target_str.strip())
    # Now parse something like "5, 50, 79, 27" into a list of integers
    if numbers_str.strip():
        initial_numbers = [int(x.strip()) for x in numbers_str.split(",")]
    else:
        initial_numbers = []
    
    # We'll keep track of our current working list of numbers
    current_numbers = initial_numbers
    
    # -----------------------------------------------------------------
    # 2. Parse solution to extract lines with "Exploring Operation:"
    # -----------------------------------------------------------------
    # Example lines:
    # Exploring Operation: 79-27=52, Resulting Numbers: [5, 50, 52]
    # We want to parse out: operand1=79, operator='-', operand2=27, result=52
    # Then parse the new list: [5, 50, 52]
    
    operation_pattern = r"Exploring Operation:\s*([\d]+)([\+\-\*/])([\d]+)=(\d+),\s*Resulting Numbers:\s*\[(.*?)\]"
    
    # We'll process the solution line-by-line
    # so that we can also capture the final "Goal Reached" line.
    lines = solution.splitlines()
    
    for line in lines:
        line = line.strip()
        
        # Check for "Exploring Operation"
        op_match = re.search(operation_pattern, line)
        if op_match:
            # Parse out the operation parts
            x_str, op, y_str, z_str, new_nums_str = op_match.groups()
            x_val = int(x_str)
            y_val = int(y_str)
            z_val = int(z_str)
            
            # Parse the new list of numbers from something like "5, 50, 52"
            new_numbers = []
            if new_nums_str.strip():
                new_numbers = [int(n.strip()) for n in new_nums_str.split(",")]
            
            # -------------------------------------------------------------
            # Verify that applying X op Y => Z to current_numbers is valid
            # -------------------------------------------------------------
            # 1. X and Y must both be present in current_numbers
            # 2. Remove X and Y from current_numbers
            # 3. Add Z
            # 4. The new list must match exactly the "Resulting Numbers"
            # 5. Also verify the arithmetic was correct (if you want to be strict)
            #
            # NOTE: we do not handle repeating values carefully here if X or Y
            # appear multiple times, but you can adapt as needed (e.g. remove once).
            
            temp_list = current_numbers[:]
            
            # Try removing X, Y once each
            try:
                temp_list.remove(x_val)
                temp_list.remove(y_val)
            except ValueError:
                if VERBOSE:
                    print(f"ERROR: {x_val} or {y_val} not found in current_numbers {current_numbers}.")
                return False
            
            # Check that the stated Z matches the arithmetic operation
            # (If you want to skip verifying the math, remove these lines.)
            computed_result = None
            if op == '+':
                computed_result = x_val + y_val
            elif op == '-':
                computed_result = x_val - y_val
            elif op == '*':
                computed_result = x_val * y_val
            elif op == '/':
                # watch for zero division or non-integer division if you want
                # to require exact integer results
                if y_val == 0:
                    if VERBOSE:
                        print("ERROR: Division by zero encountered.")
                    return False
                # For a typical "24 game" style puzzle, we allow float or integer check
                computed_result = x_val / y_val
                # If the puzzle requires integer arithmetic only, check remainder:
                # if x_val % y_val != 0:
                #     print("ERROR: Non-integer division result.")
                #     return False
            
            # Compare the stated z_val to the computed result 
            # (if it's integer-based arithmetic, we might check int(...) or round)
            if computed_result is None:
                if VERBOSE:
                    print("ERROR: Unknown operation encountered.")
                return False

            # If we want exact integer match (for e.g. 50/5=10):
            # If float is possible, we might do a small epsilon check:
            # e.g. if abs(computed_result - z_val) > 1e-9
            if computed_result != z_val:
                if VERBOSE:
                    print(f"ERROR: Operation {x_val}{op}{y_val} does not equal {z_val}. Got {computed_result} instead.")
                return False
            
            # Now add the result to temp_list
            temp_list.append(z_val)
            # Sort if you do not care about order, or keep order if you do
            # and compare to new_numbers
            # We'll assume exact order is not critical, so let's do a sorted comparison:
            if sorted(temp_list) != sorted(new_numbers):
                if VERBOSE:
                    print(f"ERROR: After applying {x_val}{op}{y_val}={z_val} to {current_numbers}, "
                          f"got {sorted(temp_list)} but solution says {sorted(new_numbers)}.")
                return False
            
            # If we got here, it means the operation is consistent
            current_numbers = new_numbers
        
        # ---------------------------------------------------------
        # 3. Check for "Goal Reached" line
        # ---------------------------------------------------------
        # Something like: "62,62 equal: Goal Reached"
        # We'll check if the final single number is indeed `target`.
        if "Goal Reached" in line:
            # For a simple check, if "Goal Reached" is present,
            # confirm that current_numbers is [target].
            if len(current_numbers) == 1 and current_numbers[0] == target:
                return True
            else:
                if VERBOSE:
                    print("ERROR: 'Goal Reached' declared but final numbers don't match the target.")
                return False
    
    # If we never saw "Goal Reached," then it's incomplete
    # or didn't declare success. Return False by default
    if VERBOSE:
        print("ERROR: Did not find 'Goal Reached' in solution.")
    return False

def get_search_result(search_trace):
    # Given a search trace, return the result of the search
    # If the search is successful, return the result optimal path
    # If the search is unsuccessful, return None
    if search_trace.count("Goal Reached") >= 2:
        # Find all occurrences of "Goal Reached"
        goal_indices = [i for i in range(len(search_trace)) if search_trace.startswith("Goal Reached", i)]
        # Get the second to last index, this is where we begin generate
        # the optimal path
        goal_idx = goal_indices[-2]
        return search_trace[goal_idx:].strip()[13:]
    else:
        return None

def get_subsearch_info(search_trace):
    try:
        return _get_subsearch_info(search_trace)
    except Exception as e:
        print(f"Error at get_subsearch_info: {e}")
        print(search_trace)
        raise e

def _get_subsearch_info(search_trace):
    # Given a search trace, return the information of the 
    # subsearch that it wants to invoke
    # sub_search= {"node": "#1,1,2", "target": 39, 'nums':[2, 11], "operations": ["51-49=2", "36-25=11"]}
    last_line = search_trace.split("\n")[-1]
    assert "<End Calling Sub Searches>" in last_line, "This is not a valid subsearch trace"

    # --- Parse the search trace to get the generated nodes ---
    generated_nodes = {}
    # First find any "Moving to Node" lines followed by "Current State" lines
    lines = search_trace.split("\n")
    for i in range(len(lines)-1):
        # Moving to Node #1,1
        # Current State: 39:[25, 36, 2], Operations: ['51-49=2']
        if "Moving to Node #" in lines[i] and "Current State:" in lines[i+1]:
            # Extract node id from first line like:
            # Moving to Node #1,1
            node_id = lines[i].split("Moving to Node #")[1].strip()
            
            # Extract state from second line like:
            # Current State: 39:[25, 36, 2], Operations: ['51-49=2']
            state_line = lines[i+1]
            state_part = state_line.split("Current State:")[1].split("],")[0].strip()
            operations_part = state_line.split("Operations:")[1].strip()
            
            # Parse state like "39:[25, 36, 2]"
            target = int(state_part.split(":")[0])
            # nums = eval(state_part.split(":")[1].strip())
            nums = eval(state_part.split(":")[1].strip() + "]")
            operations = eval(operations_part)
            
            # Parse operations list
            
            generated_nodes[node_id] = {
                "node": f"#{node_id}",
                "target": target,
                "nums": nums,
                "operations": operations
            }
    for line in search_trace.split("\n"):
        if "Generated Node" in line:
            # Extract node id and info from line like:
            # Generated Node #1,1,2: 39:[2, 11] Operation: 36-25=11
            node_id = line.split(":")[0].split("#")[1]
            if node_id in generated_nodes:
                continue
            rest = line.split(":", 1)[1].strip()
            state = rest.split("Operation:")[0].strip()
            operation = rest.split("Operation:")[1].strip()
            
            # Parse state like "39:[2, 11]" into target and nums
            target = int(state.split(":")[0])
            nums = eval(state.split(":")[1].strip())

            parent_node_id = ",".join(node_id.split(",")[:-1])
            parent_node = generated_nodes[parent_node_id]
            new_operations = parent_node["operations"] + [operation]
            
            generated_nodes[node_id] = {
                "node": f"#{node_id}",
                "target": target,
                "nums": nums,
                "operations": new_operations
            }
    # then we construct the sub_searches
    sub_search_nodes = []
    # Split on <Calling Sub Searches> and take the last chunk
    last_chunk = search_trace.split("<Calling Sub Searches>\n")[-1]
    # Split that chunk on <End Calling Sub Searches> and take first part
    sub_search_section = last_chunk.split("\n<End Calling Sub Searches>")[0]

    for line in sub_search_section.split("\n"):
        if "<Start Sub Search" in line and "Moving to Node" in line:
            # Extract node id from line like:
            # <Start Sub Search 0 at level 1> Moving to Node #1,1,2
            node_id = line.split("Moving to Node #")[1].strip()
            sub_search_nodes.append(generated_nodes[node_id])

    def construct_sub_search_prefix(node):
        # exmaple
        # "Moving to Node #1,1,0\nCurrent State: 39:[36, 50], Operations: ['51-49=2', '25*2=50']
        return f"Moving to Node {node['node']}\nCurrent State: {node['target']}:[{', '.join(map(str, node['nums']))}], Operations: {node['operations']}"
    sub_search_prefix_list = [construct_sub_search_prefix(node) for node in sub_search_nodes]
    return sub_search_prefix_list, sub_search_nodes

def get_main_trace_after_sub_search(main_trace, sub_search_nodes, sub_search_result_list):
    last_line = main_trace.split("\n")[-1]
    assert "<End Calling Sub Searches>" in last_line, "This is not a valid subsearch trace"
    sub_searches = []
    # Split on <Calling Sub Searches> and take the last chunk
    last_chunk = main_trace.split("<Calling Sub Searches>\n")[-1]
    # Split that chunk on <End Calling Sub Searches> and take first part
    sub_search_section = last_chunk.split("\n<End Calling Sub Searches>")[0]
    main_trace_after_sub_search = "<Calling Sub Searches>\n".join(main_trace.split("<Calling Sub Searches>\n")[:-1])
    assert main_trace_after_sub_search in main_trace
    main_trace_after_sub_search += "<Sub Searches>\n"
    for i, (this_node, this_result) in enumerate(zip(sub_search_nodes, sub_search_result_list)):
        if this_result is None:
            # <No Solution in Sub Search 1 at level 1 at Node #1,2,1>
            main_trace_after_sub_search += f"<No Solution in Sub Search {i} at level 1 at Node {this_node['node']}>\n"
        else:
            # <Goal Reached in Sub Search 0 at level 1 at Node #1,2,0>\nMoving to Node #1,2,0\nCurrent State: 39:[51, 12], Operations: ['49-36=13', '25-13=12']\nExploring Operation: 51-12=39, Resulting Numbers: [39]\n39,39 equal: Goal Reached\n
            main_trace_after_sub_search += f"<Goal Reached in Sub Search {i} at level 1 at Node {this_node['node']}>\n"
            main_trace_after_sub_search += this_result + "\n"
    main_trace_after_sub_search += "<End Sub Searches>\n"
    return main_trace_after_sub_search



def add_angle_brackets(text):
    lines = text.split('\n')
    result_lines = []
    for line in lines:
        if '>' in line and '<' not in line:
            line = '<' + line
        result_lines.append(line)
    return '\n'.join(result_lines)

def generate(prefix, tokenizer, api_base_url, temperature=0.5, stop=[]):
    """Generate text using the model API"""
    bos_token = tokenizer.bos_token
    prefix = bos_token + prefix
    
    result = text_completion(
        model="openai/model",
        prompt=prefix,
        api_base=api_base_url,
        api_key="api_key",
        temperature=temperature,
        max_tokens=4096,
        stop=stop,
    )
    
    text = result['choices'][0]['text']
    complete_text = prefix + text
    complete_text = complete_text.replace(bos_token, ' ')
    if complete_text[0] == ' ':
        complete_text = complete_text[1:]
    return complete_text, result

def decode_trace(prefix, tokenizer, api_base_url, temperature=0.5):
    """Decode a single trace"""
    while True:
        trace = generate(prefix, tokenizer, api_base_url, temperature=temperature, stop=["<Sub Searches>"])
        llm_call_info = {
            "prefix": prefix,  # Store the original prefix
            "output": trace[1]["choices"][0]["text"]
        }
        # Store the prefix and output in a dictionary
        prefix = trace[0]
        if trace[1].choices[0].matched_stop == "<Sub Searches>":
            prefix += "<Calling Sub Searches>"
        else:
            break
    prefix = trace[0]
    if prefix.split('\n')[-1] == "":
        prefix = prefix[:-1]
    return prefix, llm_call_info

def batch_decode_trace(prefix_list, tokenizer, api_base_url, temperature=0.5, max_workers=16):
    """Decode multiple traces in parallel"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prefix = {
            executor.submit(decode_trace, prefix, tokenizer, api_base_url, temperature): prefix 
            for prefix in prefix_list
        }
        
        results = [None] * len(prefix_list)
        llm_calls = [None] * len(prefix_list)
        
        for future in as_completed(future_to_prefix):
            prefix = future_to_prefix[future]
            try:
                result, llm_call_info = future.result()
                original_idx = prefix_list.index(prefix)
                results[original_idx] = result
                llm_calls[original_idx] = llm_call_info
            except Exception as e:
                print(f"Error processing prefix: {e}")
                original_idx = prefix_list.index(prefix)
                results[original_idx] = None
                llm_calls[original_idx] = None
                
    return results, llm_calls

def is_calling_subsearch(trace):
    return "<End Calling Sub Searches>" in trace.split('\n')[-1]

def call_search(prefix, api_base_url, temperature=0.5):
    """
    Main search function that processes a given prefix
    
    Args:
        prefix (str): The initial state and goal
        api_base_url (str): The URL for the model API
        temperature (float): Temperature parameter for text generation
        
    Returns:
        tuple: (solution, trace_dict, success, true_success)
    """
    condition_prefix = prefix.split("\n")[0].split("Moving to Node")[0]
    try:
        trace_dict = {"main_calls": [], "sub_calls": [], "llm_calls": []}
        trace, llm_call_info = decode_trace(prefix, tokenizer, api_base_url, temperature=temperature)
        trace_dict["main_calls"].append(trace)
        trace_dict["llm_calls"].append(llm_call_info)
        
        while is_calling_subsearch(trace):
            sub_search_prefix_list, sub_search_nodes = get_subsearch_info(trace)
            for idx, sub_search_prefix in enumerate(sub_search_prefix_list):
                if condition_prefix.startswith("Sub Call Budget: "):
                    # we don't need to add the condition prefix to the sub search prefix if the condition prefix starts with "Sub Call Budget: "
                    sub_search_prefix_list[idx] = sub_search_prefix
                elif condition_prefix.startswith("Token Budget: "):
                    # we need to add the condition prefix to the sub search prefix if the condition prefix starts with "Token Budget: "
                    sub_search_prefix_list[idx] = condition_prefix + sub_search_prefix
                elif condition_prefix:
                    raise ValueError(f"Unknown condition prefix: {condition_prefix}")
            sub_search_traces, sub_search_llm_calls = batch_decode_trace(sub_search_prefix_list, tokenizer, api_base_url, temperature=temperature)
            
            trace_dict["sub_calls"].append([])
            for sub_search_trace, sub_search_llm_call in zip(sub_search_traces, sub_search_llm_calls):
                trace_dict["sub_calls"][-1].append({
                    "main_calls": [sub_search_trace],
                    "llm_calls": [sub_search_llm_call]
                })
            
            sub_search_results = [get_search_result(trace) for trace in sub_search_traces]
            new_prefix = get_main_trace_after_sub_search(trace, sub_search_nodes, sub_search_results)
            trace, llm_call_info = decode_trace(new_prefix, tokenizer, api_base_url, temperature=temperature)
            trace_dict["main_calls"].append(trace)
            trace_dict["llm_calls"].append(llm_call_info)
        
        solution = get_search_result(trace)
        success = solution is not None
        true_success = check_solution(prefix, solution) if success else False
        
        return solution, trace_dict, success, true_success
        
    except Exception as e:
        print(f"Error in call_search: {e}")
        return None, trace_dict, False, False

def process_single_prefix(prefix, server_url, bos_token, temperature=0.5):
    """Helper function to process a single prefix for parallel execution"""
    solution, trace_dict, success, true_success = call_search(prefix, server_url, temperature)
    seqs = []
    for lm_call in trace_dict['llm_calls']:
        seqs.append(lm_call)
    for sub_calls in trace_dict['sub_calls']:
        for sub_call in sub_calls:
            for lm_call in sub_call['llm_calls']:
                seqs.append(lm_call)
    return {
        "seqs": seqs,
        "is_correct": true_success
    }

def rollout_hsp(server_url, prefix_list, bos_token, temperature=0.5, max_workers=32, condition_prefix=""
                ):
    """
    Parallel implementation of rollout function using ThreadPoolExecutor
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures with index tracking
        futures = []
        for idx, prefix in enumerate(prefix_list):
            future = executor.submit(
                process_single_prefix,
                condition_prefix + prefix,
                server_url,
                bos_token,
                temperature
            )
            futures.append((idx, future))
        
        # Initialize results list with correct size
        results = [None] * len(prefix_list)
        
        # Collect results as they complete
        for idx, future in futures:
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"Error processing prefix at index {idx}: {e}")
                results[idx] = {
                    "seqs": [],
                    "is_correct": False
                }
    # if the result is None, something's wrong in the decoding process
    # we use a dummy result to avoid breaking the loop
    for idx in range(len(results)):
        if not results[idx]:
            print(f"Error processing prefix {prefix_list[idx]} at index {idx}")
            results[idx] = {
                "seqs": [],
                "is_correct": False
            }
    for result_idx in range(len(results)):
        for seq_id in reversed(range(len(results[result_idx]['seqs']))):
            if not results[result_idx]['seqs'][seq_id]:
                # if one seq is None, we remove that seq
                results[result_idx]['seqs'].pop(seq_id)
                print(f"Removing seq {seq_id} from result {result_idx} since it's None")
    return results