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
                                      best_res_update : str = 'test_avg_loss') -> Tuple[Dict, Dict]:
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
                if round == 'Final':
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


def from_dir_paths_get_eval_metrics(dir_paths : list, best_res_update : str = 'test_avg_loss'):
    eval_result_via_path = {}
    
    for dir_path in dir_paths:
        eval_result_log, config, print_log = _get_exp_files_path(dir_path)
        with open(eval_result_log, 'r') as f:
            eval_log_file = f.read()
            eval_result_via_path[dir_path] = (from_eval_log_return_eval_metrics(eval_log_file, best_res_update))
    
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
        
## TESTING
