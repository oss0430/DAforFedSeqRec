import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--user_column', type=str, default='user_id:token')
parser.add_argument('--item_column', type=str, default='item_id:token')
parser.add_argument('--output_path', type=str, default='item_statistics.txt')
parser.add_argument('--k', type=int, default=200)

args = parser.parse_args()


def order_item_id_by_frequency(
    df : pd.DataFrame,
    item_id_col : str
) -> Dict[int,int] :
    """
        Order item ids by frequency
    """
    # Create a dictionary to store the frequency of each item id
    item_counts = df[item_id_col].value_counts().to_dict()

    # Sort the dictionary by frequency in descending order
    sorted_item_by_counts = dict(sorted(item_counts.items(), key=lambda item: item[1]))

    return sorted_item_by_counts


def get_user_sequence_length_stats(
    df : pd.DataFrame,
    user_col : str
) :
    user_per_sequence_length = df.groupby(user_col).size()
    
    mean_sequence_length = user_per_sequence_length.mean()
    max_sequence_length = user_per_sequence_length.max()
    min_sequence_length = user_per_sequence_length.min()
    std_length = user_per_sequence_length.std()
    return  f"mean length : {mean_sequence_length}\nmax length : {max_sequence_length}\nmin length : {min_sequence_length}\nstd length : {std_length}\n"

def get_user_via_item_unique_count(
    df : pd.DataFrame,
    user_col : str,
    item_col : str
) :
    user_item_count = df.groupby(user_col)[item_col].nunique()
    mean_user_item_count = user_item_count.mean()
    max_user_item_count = user_item_count.max()
    min_user_item_count = user_item_count.min()
    std_user_item_count = user_item_count.std()
    return f"mean user item count : {mean_user_item_count}\nmax user item count : {max_user_item_count}\nmin user item count : {min_user_item_count}\nstd user item count : {std_user_item_count}\n"



def data_statistics(
    item_counts : Dict[int,int]
) -> str:
    
    mean = np.mean(list(item_counts.values()))
    median = np.median(list(item_counts.values()))
    std = np.std(list(item_counts.values()))
    
    max_item = max(item_counts, key=item_counts.get)
    max_count = item_counts[max_item]
    
    min_item = min(item_counts, key=item_counts.get)
    min_count = item_counts[min_item]
    
    result_txt = f"Mean: {mean}\nMedian: {median}\nStandard Deviation: {std}\nMax Item: {max_item} with count {max_count}\nMin Item: {min_item} with count {min_count}\n"
    
    return result_txt


def get_top_and_bottom_k_item_ids(
    item_counts : Dict[int,int],
    k : int
) -> str :
    
    top_k = list(item_counts.items())[:k]
    bottom_k = list(item_counts.items())[-k:]
    
    result_txt = f"Top {k} items: {top_k}\nBottom {k} items: {bottom_k}\n"
    
    return result_txt


def get_sorted_frequencey_of_items(
    item_counts : Dict[int,int]
) -> str :
    
    result_txt = ""
    for item, count in item_counts.items() :
        result_txt = result_txt + f"Item {item} : {count}\n"
    
    return result_txt


def get_middle_item_ids(
    item_counts : Dict[int,int]
) -> str :
    
    middle = list(item_counts.items())[len(item_counts)//2]
    
    result_txt = f"Middle item: {middle}\n"
    
    return result_txt


if __name__ == '__main__' :
    
    result = ''
    
    data = pd.read_csv(args.data_path, header = 0, sep = ',')
    item_id_counts = order_item_id_by_frequency(
        df = data,
        item_id_col = args.item_column
    )
    
    ## User Statistics
    result = result + f"Number of users : {data[args.user_column].nunique()}\n"
    result = result + get_user_sequence_length_stats(data, args.user_column)
    result = result + get_user_via_item_unique_count(data, args.user_column, args.item_column)
    ## Item Statistics
    result = result + data_statistics(item_id_counts)
    result = result + get_top_and_bottom_k_item_ids(item_id_counts, args.k)
    result = result + get_middle_item_ids(item_id_counts)
    
    result = result + get_sorted_frequencey_of_items(item_id_counts)
    
    ## write result to file
    with open(args.output_path, 'w') as f:
        f.write(result)

