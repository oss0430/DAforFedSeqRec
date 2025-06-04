import argparse
import os
import sys

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--item_column', type=str, default='item_id:token')
parser.add_argument('-u', '--user_column', type=str, default='user_id:token')
parser.add_argument('-t', '--timestamp_column', type=str, default='timestamp:float')
parser.add_argument('-p', '--padding_range', type=list, default=[0])
parser.add_argument('--input_csv', type=str, default='Amazon_Sports_and_Outdoors_5core/original/Amazon_Sports_and_Outdoors_5core.inter')
parser.add_argument('--output_dir', type=str, default='Amazon_Sports_and_Outdoors_5core')
parser.add_argument('--item_dict_csv', type=str, default='Amazon_Sports_and_Outdoors_5core/original/Amazon_Sports_and_Outdoors_5core_item_freq.txt')
parser.add_argument('--item_map_path', type=str, default='Amazon_Sports_and_Outdoors_5core/item.csv')
args = parser.parse_args()


def make_map(
    df: pd.DataFrame,
    col: str,
    padding_range : list = [0]    
) -> dict:
    """
    Make a map from string id to int id.
    """
    ids = df[col].unique().tolist()
    map = {}
    idx = 0
    current_id = ids.pop(0)
    while True :
        if idx in padding_range :
            idx += 1
        else :
            map[current_id] = idx
            if len(ids) == 0 :
                break
            current_id = ids.pop(0)
            idx += 1    
    return map


def __main__():
    df = pd.read_csv(args.input_csv, sep='\t')
    try :
        ## read item dict if exist
        item_dict = pd.read_csv(args.item_dict_csv, sep='\t', dtype={args.item_column: str, args.user_column: str})
        item_map = make_map(item_dict, args.item_column, args.padding_range)
    except :
        item_map = make_map(df, args.item_column, args.padding_range)
    user_map = make_map(df, args.user_column, args.padding_range)
    df[args.item_column] = df[args.item_column].map(item_map).astype(str)
    #df[args.item_column] = pd.to_numeric(df[args.item_column], errors='coerce', downcast='integer')
    df[args.user_column] = df[args.user_column].map(user_map).astype(int)
    #df[args.user_column] = pd.to_numeric(df[args.user_column], errors='coerce', downcast='integer')
    
    ## sort by user_column and timestamp
    df = df.sort_values(by=[args.user_column, args.timestamp_column])
    ## change back to str
    df[args.user_column] = df[args.user_column].astype(str)
    
    output_csv_path = os.path.join(args.output_dir, os.path.basename("inter.csv"))
    df.to_csv(output_csv_path, index=False)
    
    ## save map assign key as first column and value as second column
    df_item_map = pd.DataFrame.from_dict(item_map, orient='index').reset_index()
    df_item_map.columns = ['item_id:token', 'item_id:int']
    df_user_map = pd.DataFrame.from_dict(user_map, orient='index').reset_index()
    df_user_map.columns = ['user_id:token', 'user_id:int']
    
    item_map_path = os.path.join(args.output_dir, os.path.basename("item.csv"))
    user_map_path = os.path.join(args.output_dir, os.path.basename("user.csv"))
    df_item_map.to_csv(item_map_path, index = False)
    df_user_map.to_csv(user_map_path, index = False)
    
    
if __name__ == '__main__':
    __main__()