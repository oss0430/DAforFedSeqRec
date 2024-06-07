import os
import numpy as np
import logging
import torch
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SequentialRecommendationDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        df : pd.DataFrame,
        user_column : str,
        item_column : str,
        interaction_column : str,
        timestamp_column : str,
        min_sequence_length : int = None, 
        max_sequence_length : int = None,
        user_num : int = None,
        item_num : int = None,
        padding_value : int = 0
    ) :
        super(SequentialRecommendationDataset, self).__init__()
        self.df = df
        self.user_column = user_column
        self.item_column = item_column
        self.interaction_column = interaction_column
        self.timestamp_column = timestamp_column
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        
        if user_num is None :
            self.user_num = len(df[user_column].unique())
        else :
            self.user_num = user_num
        
        if item_num is None :
            self.item_num = max(df[item_column].unique())
        else :
            self.item_num = item_num
    
    
    def __len__(self) :
        return len(self.df[self.user_column].unique())
        
    
    def __getitem__(
        self,
        idx : int 
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:#Dict[str,torch.Tensor] :
        
        user_id = self.df[self.user_column].unique()[idx]
        
        user_df = self.df[self.df[self.user_column] == user_id]
        
        ## Considering the df is sorted by user_index and timestamp
        user_interaction = user_df[self.item_column].values
        
        user_id = np.array([user_id])
        item_seq = user_interaction[:-1]
        item_seq_len = np.array([len(item_seq)])
        
        ## Note! Maybe we don't need to make target_item as batchsize x 1 but just batchsize
        target_item = np.array(user_interaction[-1])
        
        if self.max_sequence_length is not None :
            if len(item_seq) > self.max_sequence_length :
                item_seq = item_seq[:self.max_sequence_length]
                item_seq_len = np.array([self.max_sequence_length])
            else :
                item_seq = np.pad(
                    item_seq,
                    (0, self.max_sequence_length - len(item_seq)),
                    constant_values = self.padding_value
                )
        user_id = torch.tensor(user_id, dtype=torch.int64)
        item_seq = torch.tensor(item_seq, dtype=torch.int64)
        item_seq_len = torch.tensor(item_seq_len, dtype=torch.int64)
        target_item = torch.tensor(target_item, dtype=torch.int64)
        #return item_seq, item_seq_len, target_item
        
        return {   
                'user_id' : user_id,
                'item_seq' : item_seq,
                'item_seq_len' : item_seq_len,
                'target_item' : target_item
        }
        
    
def cut_by_in_sequence_length(
    sorted_df : pd.DataFrame,
    user_column : str,
    min_sequence_length : int = 5
) -> pd.DataFrame :
    ## If the user's interaction sequence is shorter than min_sequence_length,
    ## We drop the user's interaction sequence
    entire_user_id = sorted_df[user_column].unique()
    user_mask = (sorted_df.groupby(user_column).count() >= min_sequence_length).values.squeeze(1)
    
    filtered_user_id = entire_user_id[user_mask]
    filtered_df = sorted_df[sorted_df[user_column].isin(filtered_user_id)]
        
    return filtered_df

    
def split_dataframe_into_train_valid_test(
    sorted_df : pd.DataFrame,
    user_column : str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :

    ## Leave One Out Strategy
    ## Original dataframe is the test set
    ## Each user's last interaction will be dropped to create valid set,
    ## Each user's last two interactions will be dropped to create train set
    
    test_df = sorted_df.copy()
    
    ## Drop the last interaction of each user
    dropping_for_valid = test_df.groupby(user_column).tail(1)
    valid_df = test_df.drop(dropping_for_valid.index).reset_index(drop = True)
    
    ## Drop the last two interactions of each user
    dropping_for_train = test_df.groupby(user_column).tail(2)
    train_df = test_df.drop(dropping_for_train.index).reset_index(drop = True)
    
    return train_df, valid_df, test_df


def make_item_dropped_df(
    df : pd.DataFrame,
    user_column : str,
    itemdrop_method : str = "intermediate",
    offset : int = 0,
    dropcount :int = 0,
    dropping_user_id : List[int] = []
) -> pd.DataFrame : 

    if len(dropping_user_id) > 0 :
        def remove_first_item(group):
            if group.name in dropping_user_id:
                group_size = len(group)
                if group_size > dropcount+offset :
                    group = pd.concat([group.iloc[:offset], group.iloc[dropcount+offset:]])
                elif group_size > offset and group_size <= dropcount+offset :
                    group = group.iloc[:offset]
                else :
                    group = group
            return group

        def remove_last_item(group):
            if group.name in dropping_user_id:
                group_size = len(group)
                if group_size > dropcount+offset :
                    group = pd.concat([group.iloc[:-(dropcount+offset)], group.iloc[-offset:]])
                elif group_size > offset and group_size <= dropcount+offset :
                    group = group.iloc[-offset:]
                else :
                    group = group
            return group

        def remove_random_item(group):
            if group.name in dropping_user_id:
                drop_count = dropcount
                while len(group) > drop_count and drop_count > 0:
                    random_row = group.sample(1).index
                    group = group.drop(random_row)
                    drop_count -= 1
            return group
        ## iter the groupby object and drop the item from the user_id
        groupby_user = df.groupby(user_column)
        ## Drop only user in dropping_user_id
        if itemdrop_method == "first" :
            dropped_df = groupby_user.apply(remove_first_item).reset_index(drop=True)
        elif itemdrop_method == "random" :
            dropped_df = groupby_user.apply(remove_random_item).reset_index(drop=True)
        elif itemdrop_method == "last" :
            dropped_df = groupby_user.apply(remove_last_item).reset_index(drop=True)
        else : 
            Warning("Drop Method Unrecognized, just return original dataframe")
            return df
    else :
        ## Drop all
        def remove_first_item(group):
            group_size = len(group)
            if group_size > dropcount+offset :
                group = pd.concat([group.iloc[:offset], group.iloc[dropcount+offset:]])
            elif group_size > offset and group_size <= dropcount+offset :
                group = group.iloc[:offset]
            else :
                group = group
            return group

        def remove_last_item(group):
            group_size = len(group)
            if group_size > dropcount+offset :
                group = pd.concat([group.iloc[:-(dropcount+offset)], group.iloc[-offset:]])
            elif group_size > offset and group_size <= dropcount+offset :
                group = group.iloc[-offset:]
            else :
                group = group
            return group

        def remove_random_item(group):
            drop_count = dropcount
            while len(group) > drop_count and drop_count > 0:
                random_row = group.sample(1).index
                group = group.drop(random_row)
                drop_count -= 1
            return group
        
        if itemdrop_method == "first" :
            dropped_df = groupby_user.apply(remove_first_item).reset_index(drop=True)
        elif itemdrop_method == "random" :
            dropped_df = groupby_user.apply(remove_random_item).reset_index(drop=True)
        elif itemdrop_method == "last" :
            dropped_df = groupby_user.apply(remove_last_item).reset_index(drop=True)
        else : 
            Warning("Drop Method Unrecognized, just return original dataframe")
            return df
        
    return dropped_df
    
    
def make_sr_dataset(
    df_path : str,
    user_column : str,
    item_column : str,
    interaction_column : str,
    timestamp_column : str,
    partitioned_df_path : str = None,
    save_partitioned_df_path : str = None,
    min_sequence_length : int = None, 
    max_sequence_length : int = None,
    user_num : int = None,
    item_num : int = None,
    padding_value : int = 0,
    itemdrop_method : str = "",
    offset : int = 0,
    dropcount : int = 0,
    dropping_user_id : List[int] = None
) -> Tuple[SequentialRecommendationDataset, SequentialRecommendationDataset, SequentialRecommendationDataset] :
    
    try :
        train_df = pd.read_csv(os.path.join(partitioned_df_path, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(partitioned_df_path, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(partitioned_df_path, 'test.csv'))
        
        user_num_trian = len(train_df[user_column].unique())
        user_num_valid = len(valid_df[user_column].unique())
        user_num_test = len(test_df[user_column].unique())
        
        if user_num_trian == user_num_valid and user_num_valid == user_num_test :
            logger.info("SRData : User Number is Consistent")
        else :
            logger.warning("SRData : User Number is not Consistent")
        
    except :
        # Sort Dataframe by user and timestamp
        df = pd.read_csv(df_path, header = 0, sep = '\t')
        df = df.sort_values([user_column, timestamp_column])

        ## Cut by min_sequence_length
        ##if min_sequence_length :
        ##    df = cut_by_in_sequence_length(df, user_column, min_sequence_length)
        
        ## Try to drop if drop method exist
        if itemdrop_method != "":
            dropped_df = make_item_dropped_df(
                df = df,
                user_column = user_column,
                itemdrop_method = itemdrop_method,
                offset = offset,
                dropcount = dropcount,
                dropping_user_id = dropping_user_id
            )
            df = dropped_df
            
        # Split Dataframe in to train, valid, test according to leave one out strategy
        train_df, valid_df, test_df = split_dataframe_into_train_valid_test(df, user_column)
        
    if save_partitioned_df_path :
        if not os.path.exists(save_partitioned_df_path):
            os.makedirs(save_partitioned_df_path)
        
        train_df.to_csv(os.path.join(save_partitioned_df_path, 'train.csv'), index = False)
        valid_df.to_csv(os.path.join(save_partitioned_df_path, 'valid.csv'), index = False)
        test_df.to_csv(os.path.join(save_partitioned_df_path, 'test.csv'), index = False)
    
    ## Fix user_num and item_num according to test_df
    user_num = len(test_df[user_column].unique())
    item_num = max(test_df[item_column].unique())
        
    trainset = SequentialRecommendationDataset(
        train_df,
        user_column,
        item_column,
        interaction_column,
        timestamp_column,
        min_sequence_length,
        max_sequence_length,
        user_num,
        item_num,
        padding_value
    )
    
    validset = SequentialRecommendationDataset(
        valid_df,
        user_column,
        item_column,
        interaction_column,
        timestamp_column,
        min_sequence_length,
        max_sequence_length,
        user_num,
        item_num,
        padding_value
    )
    
    testset = SequentialRecommendationDataset(
        test_df,
        user_column,
        item_column,
        interaction_column,
        timestamp_column,
        min_sequence_length,
        max_sequence_length,
        user_num,
        item_num,
        padding_value
    )
    
    return trainset, validset, testset
    
    

def load_sr_data(
    config,
    client_cfgs = None
) : 
    trainset, validset, testset = make_sr_dataset(
        config.data.df_path,
        config.data.user_column,
        config.data.item_column,
        config.data.interaction_column,
        config.data.timestamp_column,
        config.data.partitioned_df_path,
        config.data.save_partitioned_df_path,
        config.data.min_sequence_length,
        config.data.max_sequence_length,
        config.data.user_num,
        config.data.item_num,
        config.data.padding_value,
        #config.srtest.itemdrop_method or None,
        #config.srtest.offset or None,
        #config.srtest.dropcount or None,
        #config.srtest.dropping_user_id or None
    )
    
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator((trainset, validset, testset))
    
    return fs_data, config


def call_sr_data(config, client_cfgs) :
    if config.data.type == 'sr_data' :
        data, modified_config = load_sr_data(config, client_cfgs)
        return data, modified_config
    

register_data('sr_data', call_sr_data)