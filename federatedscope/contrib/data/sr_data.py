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



class SequentialRecommendationTrainset(torch.utils.data.Dataset):
    
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
        super(SequentialRecommendationTrainset, self).__init__()
        self.df = df
        self.user_column = user_column
        self.item_column = item_column
        self.interaction_column = interaction_column
        self.timestamp_column = timestamp_column
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        
        if user_num == None :
            self.user_num = len(df[user_column].unique())
        else :
            self.user_num = user_num
        
        if item_num == None :
            self.item_num = max(df[item_column].unique())
        else :
            self.item_num = item_num
            
        self._preprocess_sequence()
            
        
    
    def _preprocess_sequence(self):
        ## cut the sequence from right each item.
        last_user_id = None
        
        user_id_list, item_seq_index, target_index, item_seq_length = [], [], [], []
        
        user_hashing = {} # for creating sub-dataset via user
        
        seq_start = 0
        instance_count = 0
        for i, user_id in enumerate(self.df[self.user_column].values):
            if last_user_id != user_id:
                if last_user_id :
                    ## keep track of instance count at with hash
                    user_hashing[last_user_id]["end"] = instance_count
                last_user_id = user_id
                seq_start = i
                user_hashing[user_id] = {"start" : instance_count, "end" : None}
                last_user_instance_count = 0
            else :
                if i - seq_start > self.max_sequence_length :
                    seq_start += 1
                user_id_list.append(user_id)
                item_seq_index.append(range(seq_start,i)) 
                target_index.append(i)
                item_seq_length.append(i - seq_start)
                instance_count += 1
        ## update user hashing for the last user
        user_hashing[last_user_id]["end"] = instance_count
        
        self.user_id_list = user_id_list
        self.item_seq_index = item_seq_index
        self.target_index = target_index
        self.item_seq_length = item_seq_length
        self.user_hashing = user_hashing

        
    def __len__(self) :
        return len(self.user_id_list)
    
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor] :
        
        user_id = self.user_id_list[idx]
        item_seq_index = self.item_seq_index[idx]
        target_index = self.target_index[idx]
        item_seq_length = self.item_seq_length[idx]
        
        item_seq = self.df[self.item_column].values[list(item_seq_index)]
        
        if self.max_sequence_length is not None :
            sequence_length = len(item_seq)
            if  sequence_length > self.max_sequence_length :
                start_index = sequence_length - self.max_sequence_length
                item_seq = item_seq[start_index:]
                item_seq_length = self.max_sequence_length
            else :
                item_seq = np.pad(
                    item_seq,
                    (0, self.max_sequence_length - sequence_length),
                    constant_values = self.padding_value
                )
        
        target_item = self.df[self.item_column].values[target_index]
        
        user_id = torch.tensor(user_id, dtype=torch.int64)
        item_seq = torch.tensor(item_seq, dtype=torch.int64)
        item_seq_length = torch.tensor(item_seq_length, dtype=torch.int64)
        target_item = torch.tensor(target_item, dtype=torch.int64)
        
        return {   
            'user_id' : user_id,
            'item_seq' : item_seq,
            'item_seq_len' : item_seq_length,
            'target_item' : target_item
        } 
    
    def _from_user_idx_get_user_subset_range(self, idx):
        ## when given the index number for user_id,
        ## return the range of data index for the user
        user_id = self.df[self.user_column].unique()[idx]
        start_end_dict = self.user_hashing[user_id]
        
        return range(start_end_dict["start"], start_end_dict["end"])
    

class SequentialRecommendationValidset(torch.utils.data.Dataset):
    
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
        super(SequentialRecommendationValidset, self).__init__()
        self.df = df
        self.user_column = user_column
        self.item_column = item_column
        self.interaction_column = interaction_column
        self.timestamp_column = timestamp_column
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        
        if user_num == None :
            self.user_num = len(df[user_column].unique())
        else :
            self.user_num = user_num
        
        if item_num == None :
            self.item_num = max(df[item_column].unique())
        else :
            self.item_num = item_num
    
    
    
                
    
    def __len__(self) :
        return len(self.df[self.user_column].unique())
        
    
    def _from_user_idx_get_user_subset_range(self, idx):
        return [idx] 
    
    
    def _get_user_df_and_user_id(
        self,
        idx : int
    ) :
        user_id = self.df[self.user_column].unique()[idx]
        user_df = self.df[self.df[self.user_column] == user_id]
        return user_df, user_id
    
    
    def __getitem__(
        self,
        idx : int 
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor] :
        
        user_df, user_id = self._get_user_df_and_user_id(idx)
        ## Considering the df is sorted by user_index and timestamp
        user_interaction = user_df[self.item_column].values
        user_id = np.array([user_id])
        item_seq = user_interaction[:-1]
        item_seq_len = np.array([len(item_seq)])
        
        ## Note! Maybe we don't need to make target_item as batchsize x 1 but just batchsize
        target_item = np.array(user_interaction[-1])
        
        if self.max_sequence_length is not None :
            sequence_length = len(item_seq)
            if  sequence_length > self.max_sequence_length :
                start_index = sequence_length - self.max_sequence_length
                ## Critical Error here and fixed it.
                item_seq = item_seq[start_index:]
                item_seq_len = np.array([self.max_sequence_length])
            else :
                item_seq = np.pad(
                    item_seq,
                    (0, self.max_sequence_length - sequence_length),
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


class SequentialRecommendationTrainsetWithAugmentation(SequentialRecommendationTrainset):
    
    def __init__(
        self,
        df_with_augmentation : pd.DataFrame,
        user_column : str,
        item_column : str,
        interaction_column : str,
        timestamp_column : str,
        augmentation_column : str,
        min_sequence_length : int = None,
        max_sequence_length : int = None,
        user_num : int = None,
        item_num : int = None,
        padding_value : int = 0
    ) :
        self.augmentation_column = augmentation_column
        self.number_of_augmentation = len(df_with_augmentation[self.augmentation_column].unique())
        super(SequentialRecommendationTrainsetWithAugmentation, self).__init__(
            df_with_augmentation,
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
        
        
    def _preprocess_sequence(self):
        ## only difference is hashing
        
        last_user_id = None
        last_augmentation_idx = None
        user_id_list, item_seq_index, target_index, item_seq_length = [], [], [], []
        
        user_hashing = {} # for creating sub-dataset via user
        
        seq_start = 0
        instance_count = 0
        ## we must account the augmentation index
        user_and_augmentation_zip = zip(self.df[self.user_column].values, 
                                        self.df[self.augmentation_column].values)
        for i, (user_id, augmentation_idx) in enumerate(user_and_augmentation_zip):
            if last_user_id != user_id:
                ## updating hashing occurs only when user changes
                if last_user_id :
                    user_hashing[last_user_id]["end"] = instance_count
                last_user_id = user_id
                user_hashing[user_id] = {"start" : instance_count, "end" : None}
            if last_augmentation_idx != augmentation_idx:
                ## new sequence begins when augmentation_idx changes
                seq_start = i
                last_augmentation_idx = augmentation_idx    
            else :
                if i - seq_start > self.max_sequence_length :
                    seq_start += 1
                user_id_list.append(user_id)
                item_seq_index.append(range(seq_start,i)) 
                target_index.append(i)
                item_seq_length.append(i - seq_start)
                instance_count += 1
        
        ## update user hashing for the last user
        user_hashing[last_user_id]["end"] = instance_count
        
        self.user_id_list = user_id_list
        self.item_seq_index = item_seq_index
        self.target_index = target_index
        self.item_seq_length = item_seq_length
        self.user_hashing = user_hashing
    

 
class SequentialRecommendationDatasetWithAugmentation(SequentialRecommendationValidset):
     
    def __init__(
        self,
        df_with_augmentation : pd.DataFrame,
        user_column : str,
        item_column : str,
        interaction_column : str,
        timestamp_column : str,
        augmentation_column : str,
        min_sequence_length : int = None, 
        max_sequence_length : int = None,
        user_num : int = None,
        item_num : int = None,
        padding_value : int = 0
    ) :
        super(SequentialRecommendationDatasetWithAugmentation, self).__init__(
            df_with_augmentation,
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
        self.augmentation_column = augmentation_column
        self.number_of_augmentation = len(self.df[self.augmentation_column].unique())
        self._register_augmentation_range_dict()
    
    
    def _register_augmentation_range_dict(self):
        
        augmentation_range_dict = {}
        idx_to_user_id_and_augmentation_id = {}
        
        current_start = 0
        current_end = 0
        
        logger.info("SRData : Creating Augmentation Range Dictionary")
        
        for user_id in self.df[self.user_column].unique():
            user_df = self.df[self.df[self.user_column] == user_id]
            
            for augmentation_id in user_df[self.augmentation_column].unique():
                idx_to_user_id_and_augmentation_id[current_end] = {
                    "user_id" : user_id, "augmentation_id" : augmentation_id}
                current_end += 1
            
            augmentation_range_dict[user_id] = [idx for idx in range(current_start, current_end)]
            current_start = current_end
        
        self.idx_to_user_id_and_augmentation_id = idx_to_user_id_and_augmentation_id
        self.augmentation_range = augmentation_range_dict


    def _from_user_idx_get_user_subset_range(self, idx):
        ## given idx is a range of ids corresponded to
        user_id = idx + 1
        return self.augmentation_range[user_id] 
        
    
    def __len__(self) :
        return len(self.idx_to_user_id_and_augmentation_id.keys())
    
    
    def _get_user_df_and_user_id(self, idx: int):
        
        current_pair = self.idx_to_user_id_and_augmentation_id[idx]
        
        user_id = current_pair["user_id"]
        augmentation_id = current_pair["augmentation_id"]
        
        ## get the correct user and correct augmentation
        user_df = self.df[self.df[self.user_column] == user_id]
        user_df = user_df[user_df[self.augmentation_column] == augmentation_id]
        
        return user_df, user_id
        
        
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
    

def check_user_num_consistancy(
    user_column : str,
    train_df : pd.DataFrame,
    valid_df : pd.DataFrame,
    test_df : pd.DataFrame,
) -> bool :
    
    if len(train_df[user_column].unique()) == len(valid_df[user_column].unique()) and \
        len(valid_df[user_column].unique()) == len(test_df[user_column].unique()) :
        logger.info("SRData : User Number is Consistent")
        return True
    else :
        logger.warning("SRData : User Number is not Consistent")
        return False



def make_sr_dataset(
    df_path : str,
    user_column : str,
    item_column : str,
    interaction_column : str,
    timestamp_column : str,
    augmentation_column : str = None,
    use_augmentation : bool = False,
    max_augmentation_idx : int = 0,
    partitioned_df_path : str = None,
    save_partitioned_df_path : str = None,
    min_sequence_length : int = None, 
    max_sequence_length : int = None,
    user_num : int = None,
    item_num : int = None,
    padding_value : int = 0
) -> Tuple[SequentialRecommendationTrainset,
           SequentialRecommendationValidset,
           SequentialRecommendationValidset] :
    
    if use_augmentation :
        logger.warning("SRData with Augmentation is only availiable for reading")
        train_with_augmentation_df = pd.read_csv(os.path.join(partitioned_df_path, 'train.csv'))
        
        ## drop the augmentation index that is greater than max_augmentation_idx
        train_with_augmentation_df = train_with_augmentation_df[train_with_augmentation_df[augmentation_column] <= max_augmentation_idx]
        train_with_augmentation_df = train_with_augmentation_df.reset_index(drop=True)
        
        valid_df = pd.read_csv(os.path.join(partitioned_df_path, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(partitioned_df_path, 'test.csv'))
        
        check_user_num_consistancy(user_column, train_with_augmentation_df, valid_df, test_df)
        
        user_num = len(test_df[user_column].unique())
        item_num = max(test_df[item_column].unique())
        
        trainset = SequentialRecommendationTrainsetWithAugmentation(
            train_with_augmentation_df,
            user_column,
            item_column,
            interaction_column,
            timestamp_column,
            augmentation_column,
            min_sequence_length,
            max_sequence_length,
            user_num,
            item_num,
            padding_value
        )
        
        
    else :   
        try :
            train_df = pd.read_csv(os.path.join(partitioned_df_path, 'train.csv'))
            valid_df = pd.read_csv(os.path.join(partitioned_df_path, 'valid.csv'))
            test_df = pd.read_csv(os.path.join(partitioned_df_path, 'test.csv'))
            
            check_user_num_consistancy(user_column, train_df, valid_df, test_df)
            
        except :
            ## Read the original dataframe and split it
            logging.info("SRData : Partitioned Dataframe not found, Splitting Dataframe")
            # Sort Dataframe by user and timestamp
            df = pd.read_csv(df_path, header = 0, sep = '\t')
            df = df.sort_values([user_column, timestamp_column])
                
            # Split Dataframe in to train, valid, test according to leave one out strategy
            train_df, valid_df, test_df = split_dataframe_into_train_valid_test(df, user_column)
            
            if save_partitioned_df_path :
                logging.info("SRData : Saving Partitioned Dataframe")
                if not os.path.exists(save_partitioned_df_path):
                    os.makedirs(save_partitioned_df_path)
                
                train_df.to_csv(os.path.join(save_partitioned_df_path, 'train.csv'), index = False)
                valid_df.to_csv(os.path.join(save_partitioned_df_path, 'valid.csv'), index = False)
                test_df.to_csv(os.path.join(save_partitioned_df_path, 'test.csv'), index = False)
        
        ## Fix user_num and item_num according to test_df
        user_num = len(test_df[user_column].unique())
        item_num = max(test_df[item_column].unique())
            
        trainset = SequentialRecommendationTrainset(
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
        
    validset = SequentialRecommendationValidset(
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
    
    testset = SequentialRecommendationValidset(
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
        config.data.augmentation_column,
        config.data.use_augmentation,
        config.data.max_augmentation_idx,
        config.data.partitioned_df_path,
        config.data.save_partitioned_df_path,
        config.data.min_sequence_length,
        config.data.max_sequence_length,
        config.data.user_num,
        config.data.item_num,
        config.data.padding_value
    )
    
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator((trainset, validset, testset))
    
    return fs_data, config


def call_sr_data(config, client_cfgs) :
    if config.data.type == 'sr_data' :
        data, modified_config = load_sr_data(config, client_cfgs)
        return data, modified_config
    

register_data('sr_data', call_sr_data)
