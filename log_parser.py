import os
import ast
import re
import numpy as np
from typing import List, Dict, Tuple

def _get_exp_files_path(dir_path) -> Tuple[str, str, str]:
    eval_result_log = os.path.join(dir_path, "eval_results.log")
    config = os.path.join(dir_path, "config.yaml")
    print_log = os.path.join(dir_path, "exp_print.log")
    
    return eval_result_log, config, print_log
    
def from_eval_log_return_eval_metrics(eval_log_file : str,
                                      best_res_update : str = 'test_avg_loss',
                                      over_write_best_with_final : bool = False
                                      ) -> Tuple[Dict, Dict]:
    eval_metrics = {'Round' : [], 'Results_raw' : []}
    best_results = None
    best_results_so_far = None
    best_res = 1000000
    for line in eval_log_file.split("\n"):
        if "Results_raw" in line:
            line_dict = ast.literal_eval(line)
            if line_dict['Role'] == 'Server #':
                ## global wise eval
                round = line_dict['Round']
                if round == 'Final' and over_write_best_with_final:
                    best_results = line_dict['Results_raw']['server_global_eval']
                elif type(round) == int:
                    eval_metrics['Round'].append(round)
                    eval_metrics['Results_raw'].append(line_dict['Results_raw'])
                    try :
                        ## since is loss smaller is better
                        if best_res > line_dict['Results_raw'][best_res_update]:
                            best_res = line_dict['Results_raw'][best_res_update]
                            best_results_so_far = line_dict['Results_raw']
                    except :
                        raise ValueError(f"best_res_update {best_res_update} not found in the eval log")
            else :
                #TODO: Add parsing for non global wise eval
                pass
    if best_results is None:
        best_results = best_results_so_far
    return best_results, eval_metrics


def from_print_log_return_client_metrics(print_log_file : str, client_per_round : int = 16) -> Tuple[Dict, List]: 
    client_metrics = {'Round' : [], 'Results_raw' : []}
    ## parse column values
    columns = ['client_id', 'Round']
    for line in print_log_file.split("\n"):
        if "Results_raw" in line and 'Client' in line:
            match = re.search(r"\{.*\}", line)
            if match:
                dict_values = match.group(0)
                line_dict = ast.literal_eval(dict_values)
                columns = columns + list(line_dict['Results_raw'].keys())
                break

    if len(columns) == 2:
        raise ValueError("No client metrics found in the print log")
    raw_metrics = []
    for line in print_log_file.split("\n"):
        if "Results_raw" in line:
            if 'Server #' in line:
                pass
            elif 'Client' in line :
                match = re.search(r"\{.*\}", line)
                dict_values = match.group(0)
                line_dict = ast.literal_eval(dict_values)
                values = np.zeros(len(columns))
                client_id = int(re.findall(r'\d+', line_dict['Role'])[0])
                round = line_dict['Round']
                values[0] = client_id
                values[1] = round
                for i, key in enumerate(columns[2:]):
                    values[i+2] = line_dict['Results_raw'][key]
                raw_metrics.append(values)
            else :
                pass
    
    ## groub by round
    raw_metrics = np.array(raw_metrics)
    if client_per_round <= 1:
        ## no need to group by round
        return raw_metrics, columns
    
    N, M = raw_metrics.shape
    if N % client_per_round != 0:
        raw_metrics = raw_metrics[:-(N % client_per_round)]
    
    client_metrics = raw_metrics.reshape(N // client_per_round, client_per_round, M)
    
    return client_metrics, columns


def from_print_log_serach_dict_via_round(print_log_file : str, search_key : str) -> List[Dict]:
    search_dicts = []
    for line in print_log_file.split("\n"):
        if search_key in line:
            match = re.search(r"\{.*\}", line)
            if match:
                dict_values = match.group(0)
                line_dict = ast.literal_eval(dict_values)
                search_dicts.append(line_dict)
    return search_dicts


def from_dir_paths_get_eval_metrics(dir_paths : list, best_res_update : str = 'test_avg_loss', over_write_best_with_final : bool = False):
    eval_result_via_path = {}
    
    for dir_path in dir_paths:
        eval_result_log, config, print_log = _get_exp_files_path(dir_path)
        with open(eval_result_log, 'r') as f:
            eval_log_file = f.read()
            eval_result_via_path[dir_path] = (from_eval_log_return_eval_metrics(eval_log_file, best_res_update, over_write_best_with_final))
    
    return eval_result_via_path

def from_dir_paths_get_client_metrics(dir_paths : list, client_per_round : int = 16):
    client_metrics_via_path = {}
    from tqdm import tqdm
    for dir_path in tqdm(dir_paths, desc="Parsing client metrics", total=len(dir_paths)):
        eval_result_log, config, print_log = _get_exp_files_path(dir_path)
        with open(print_log, 'r') as f:
            print_log_file = f.read()
            client_metrics_via_path[dir_path] = from_print_log_return_client_metrics(print_log_file, client_per_round)
    
    return client_metrics_via_path



def _clip_client_metrics_via_round(client_metrics : np.ndarray, columns : List, round : int, client_per_round : int = 16) -> Tuple[np.ndarray, int]:
    
    if len(client_metrics.shape) != 2:
        raise ValueError("client_metrics should be 2D array")
    
    round_num_column = columns.index('Round')
    round_indexes = np.where(client_metrics[:, round_num_column] == round)[0]
    last_round = round
    
    if len(round_indexes) > 0 and len(round_indexes) < 16:
        ## we will use the last round -1 as the last round, and cut the last round
        cut_index = np.where(client_metrics[:, round_num_column] == round-1)[0][-1]
        last_round = round - 1
    elif len(round_indexes) == 16 :
        cut_index = round_indexes[-1]
    else :
        raise ValueError("None existing round")

    return client_metrics[:cut_index+1], last_round


def from_print_logs_concat_client_metrics(print_logs : list, client_per_round : int = 16, check_point_list : list = []) -> Dict[np.ndarray, List]:

    current_last_round = -1
    all_client_metrics = []
    columns = None
    for i, print_log in enumerate(print_logs):
        with open(print_log, 'r') as f:
            print_log_file = f.read()
            client_metrics, columns = from_print_log_return_client_metrics(print_log_file, 1)
            first_round = int(client_metrics[0, 1])
            if len(check_point_list) > 0:
                ## check point list is where the next log starts
                last_round = check_point_list.pop(0) - 1 
            else :
                last_round = int(client_metrics[-1, 1])
            
            client_metrics, last_round = _clip_client_metrics_via_round(client_metrics, columns, last_round, client_per_round)

            if current_last_round < first_round:
                ## simple concat
                all_client_metrics.append(client_metrics)
            elif current_last_round >= first_round and current_last_round < last_round:
                ## overlapping rounds, cut the incoming round
                cut_index = np.where(client_metrics[:, 1] == current_last_round)[0][-1]
                all_client_metrics.append(client_metrics[cut_index+1:])
            else:
                raise ValueError("The print logs are not in order")

            current_last_round = last_round
    
    raw_metrics = np.concatenate(all_client_metrics, axis=0)
    if client_per_round < 1:
        ## no need to group by round
        return raw_metrics, columns
    
    N, M = raw_metrics.shape
    if N % client_per_round != 0:
        raw_metrics = raw_metrics[:-(N % client_per_round)]
        
    client_metrics = raw_metrics.reshape(N // client_per_round, client_per_round, M)
    
    return raw_metrics, columns        


## Needs Testing
def from_print_logs_concat_search_results(print_logs : list, search_key : str, check_point_list : List[int], client_per_round : int) -> List[Dict]:
    
    if len(print_logs) != len(check_point_list) + 1:
        raise ValueError("The length of print_logs should be less by 1 of check_point_list length")
    search_dicts = []
    
    current_last_round = 0
    for print_log in print_logs:
        with open(print_log, 'r') as f:
            print_log_file = f.read()
            current_search_dict = from_print_log_serach_dict_via_round(print_log_file, search_key)

            seen_rounds = len(current_search_dict) // client_per_round            
            if len(check_point_list) > 0:
                ## check point list is where the next log starts
                next_last_round = check_point_list.pop(0) - 1
            else :
                next_last_round = len(search_dicts) // client_per_round
            
            assert seen_rounds >= client_per_round *  (next_last_round - current_last_round), \
                f"current_search_dict length {len(current_search_dict)} is less than expected"  
                
            cut_index = (next_last_round - current_last_round) * client_per_round
            
            if len(search_dicts) == 0 :
                search_dicts = current_search_dict[:cut_index+1]
            else :
                search_dicts = search_dicts + current_search_dict[:cut_index+1]
            current_last_round = next_last_round
    
    return search_dicts
            


def mend_two_print_logs(print_log1_path : str, print_log2_path : str, cut_round : int) -> str:
    with open(print_log1_path, 'r') as f:
        print_log_file1 = f.read()
    with open(print_log2_path, 'r') as f:
        print_log_file2 = f.read()
    
    total_lines = []    
    ## cut file_1 before witnessing the cut_round
    for line in print_log_file1.split("\n"):
        if "Starting a new training round" in line or "Starting training" in line:
            round_match = re.search(r"\(Round #\d+\)", line)
            if round_match:
                round = int(re.search(r'\d+', round_match.group(0))[0])
                if round == cut_round:
                    break
        total_lines.append(line)
    
    starts_from_flag = False
    for line in print_log_file2.split("\n"):
        if "Starting a new training round" in line or "Starting training" in line:
            round_match = re.search(r"\(Round #\d+\)", line)
            if round_match:
                round = int(re.search(r'\d+', round_match.group(0))[0])
                if round == cut_round:
                    starts_from_flag = True
        if starts_from_flag:
            total_lines.append(line)
                
    return "\n".join(total_lines)


def mend_two_eval_logs(eval_log1_path : str, eval_log2_path : str, cut_round : int) -> str:
    with open(eval_log1_path, 'r') as f:
        eval_log_file1 = f.read
    with open(eval_log2_path, 'r') as f:
        eval_log_file2 = f.read()
        
    total_lines = []
    for line in eval_log_file1.split("\n"):
        line_dict = ast.literal_eval(line)
        if line_dict['Round'] == cut_round:
            break
        total_lines.append(line)
    
    starts_from_flag = False
    for line in eval_log_file2.split("\n"):
        line_dict = ast.literal_eval(line)
        if line_dict['Round'] == cut_round and not starts_from_flag:
            starts_from_flag = True
        if starts_from_flag:
            total_lines.append(line)
    
    return "\n".join(total_lines)
        
        