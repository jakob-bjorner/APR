import re

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
                    print("ERROR: Division by zero encountered.")
                    return False
                # For a typical "24 game" style puzzle, we allow float or integer check
                computed_result = x_val / y_val
            
            # Compare the stated z_val to the computed result 
            # (if it's integer-based arithmetic, we might check int(...) or round)
            if computed_result is None:
                print("ERROR: Unknown operation encountered.")
                return False

            # If we want exact integer match (for e.g. 50/5=10):
            # If float is possible, we might do a small epsilon check:
            # e.g. if abs(computed_result - z_val) > 1e-9
            if computed_result != z_val:
                print(f"ERROR: Operation {x_val}{op}{y_val} does not equal {z_val}. Got {computed_result} instead.")
                return False
            
            # Now add the result to temp_list
            temp_list.append(z_val)
            # Sort if you do not care about order, or keep order if you do
            # and compare to new_numbers
            # We'll assume exact order is not critical, so let's do a sorted comparison:
            if sorted(temp_list) != sorted(new_numbers):
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
                print("ERROR: 'Goal Reached' declared but final numbers don't match the target.")
                return False
    
    # If we never saw "Goal Reached," then it's incomplete
    # or didn't declare success. Return False by default
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

def get_subsearch_info(search_trace, budget=None, avg_sub_call=False):
    try:
        return _get_subsearch_info(search_trace, budget, avg_sub_call)
    except Exception as e:
        print(f"Error at get_subsearch_info: {e}")
        print(search_trace)
        raise e

def _get_subsearch_info(search_trace, budget=None, avg_sub_call=False):
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
            
    if avg_sub_call:
        assert budget is not None, "Budget must be provided if avg_sub_call is True"
        budget = budget / len(sub_search_nodes)
        budget = int((budget - 1) // 512 + 1) * 512

    def construct_sub_search_prefix(node):
        # exmaple
        # "Moving to Node #1,1,0\nCurrent State: 39:[36, 50], Operations: ['51-49=2', '25*2=50']
        prefix = f"Moving to Node {node['node']}\nCurrent State: {node['target']}:[{', '.join(map(str, node['nums']))}], Operations: {node['operations']}"
        if budget is not None:
            prefix = f"Token Budget: {budget} {prefix}"
        return prefix
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