from litellm import text_completion
import re
def _parse_trajectory(search_path):
    # Find the first occurrence of "Current State" and trim everything before it
    start_idx = search_path.find("Current State")
    if start_idx == -1:
        return "Invalid input: Cannot find the initial state."
    search_path = search_path[start_idx:]
    
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]

    # if mode == "dt":
    #     first_line = first_line.split("->")[1]
    target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
    if not target_nums_match:
        return "Invalid input: Cannot find the initial state in the first line."

    target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]

    # Extract the operations from the line that claims the goal is reached.
    goal_lines = re.finditer(r"\d+,\d+ equal: Goal Reached", search_path)
    goal_lines = list(goal_lines)
    if not goal_lines:
        return "No goal reached statement found."

    goal_line = goal_lines[-1]
    # get the last operation line before the goal reached statement
    operations = re.findall(r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]",
                            search_path[:goal_line.start()])
    if not operations:
        return "No operations found leading to the goal."

    final_operation = operations[-1][0]
    try:
        predicted_result = int(final_operation.split('=')[1])
    except:
        print("couldnt parse last op", final_operation)
        return "Couldnt parse last op"
    if predicted_result != target:
        return "Invalid path: Final operation does not result in target."

    # get the last current state, operations before the goal reached statement, and extract the operations
    try:
      core_path = search_path[:goal_line.start()].split("Goal Reached\n")[1]
    except:
        print("invalid, no summarized answer")
        return "Invalid path: no summarized answer."
    operation_list = re.findall(r"Current State: \d+:\[.*?\], Operations: \[(.*?)\]", core_path)[
        -1].split(', ')
    operation_list = [op.replace("'", "") for op in operation_list]
    operation_list += [final_operation]

    # Verify each operation and keep track of the numbers involved
    available_numbers = nums
    for operation in operation_list:
        # Verify the operation
        try:
            left, right = operation.split('=')
        except:
            return f"Could not split operation into lhs, rhs"
        try:
            if eval(left) != int(right):
                return f"Invalid operation: {operation}"
        except Exception as e:
            return f"Error in evaluating operation {operation}: {e}"
        # get the numbers involved
        used_numbers = re.findall(r"\d+", left)
        for n in used_numbers:
            if int(n) not in available_numbers:
                return f"Invalid operation: {operation}, number {n} not available in {available_numbers}"

        available_numbers = [n for n in available_numbers if n not in used_numbers]
        available_numbers.append(int(right))

    return "Valid path."

def is_correct(search_path):
    try:
        return _parse_trajectory(search_path) == "Valid path."
    except Exception as e:
        print(f"Error in is_correct: {e}, treating as incorrect")
        return False


def rollout_sos(server_url, prefix_list, bos_token, temperature=0.5, condition_prefix=None):
    if condition_prefix is not None:
        assert not prefix_list[0].startswith("Token Budget: "), f"Condition prefix already in the prefix: {prefix_list[0]}, please use data without condition prefix"
        all_prefixes = [bos_token + condition_prefix + prefix for prefix in prefix_list]
    else:
        all_prefixes = [bos_token + prefix for prefix in prefix_list]

    outputs = text_completion(
        model="openai/model",
        prompt=all_prefixes,
        api_base=server_url,
        api_key="api_key",
        temperature=temperature,
        max_tokens=4000,
    )
    return_dict = []
    for output, prefix in zip(outputs["choices"], all_prefixes):
        whole_text = prefix + output["text"]
        return_dict.append({
            'seqs': [
                {
                    "prefix": prefix,
                    "output": output["text"],
                }
            ],
            "is_correct": is_correct(whole_text)
        })
    return return_dict
