import os
import numpy as np
import logging
import torch
import pandas as pd

from typing import List, Tuple, Dict, Any, Optional

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from federatedscope.core.configs.config import CN

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
        listed_range = []
        
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
                listed_range.append(instance_count)
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
        listed_range.append(instance_count)
        
        self.user_id_list = user_id_list
        self.item_seq_index = item_seq_index
        self.target_index = target_index
        self.item_seq_length = item_seq_length
        self.user_hashing = user_hashing
        self.listed_range = listed_range
        
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
    

    def get_full_subset_range(self):
        subset_range = []
        range_starts = self.listed_range[:-1]
        range_ends = self.listed_range[1:]
        
        for start, end in zip(range_starts, range_ends):
            subset_range.append(range(start, end))
        
        return subset_range
        
        
        
    
    
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
    
    
    def get_full_subset_range(self):
        full_subset_range = []
        for i in range(len(self.df[self.user_column].unique())):
            full_subset_range.append(self._from_user_idx_get_user_subset_range(i))
        return full_subset_range
    
    
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
        listed_range = []
        
        seq_start = 0
        instance_count = 0
        ## we must account the augmentation index
        user_and_augmentation_zip = zip(self.df[self.user_column].values, 
                                        self.df[self.augmentation_column].values)
        for i, (user_id, augmentation_idx) in enumerate(user_and_augmentation_zip):
            
            if last_user_id != user_id or last_augmentation_idx != augmentation_idx:
                ## updating hashing occurs only when user changes
                if last_user_id != user_id :
                    if last_user_id :
                        user_hashing[last_user_id]["end"] = instance_count
                    last_user_id = user_id
                    user_hashing[user_id] = {"start" : instance_count, "end" : None}
                    listed_range.append(instance_count)
                ## sequence starts when either user or augmentation changes
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
        listed_range.append(instance_count)
        
        self.user_id_list = user_id_list
        self.item_seq_index = item_seq_index
        self.target_index = target_index
        self.item_seq_length = item_seq_length
        self.user_hashing = user_hashing
        self.listed_range = listed_range

 
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
        

class SequentialRecommendationTrainsetWithMultipleAugmentation(torch.utils.data.Dataset):
    
    """
    Multiple Augmentation Provides
    Augmentation & User - Wise Sub Sampling
    """
    def __init__(
        self,
        dfs : List[pd.DataFrame],
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
        self.df = dfs[0] ## reference for user_num and item_num
        self.user_column = user_column
        self.item_column = item_column
        self.interaction_column = interaction_column
        self.augmentation_column = augmentation_column
        self.timestamp_column = timestamp_column
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        super(SequentialRecommendationTrainsetWithMultipleAugmentation, self).__init__()
        augmentation_datasets = []
        for idx, df in enumerate(dfs):
            logger.info(f"SRData : loading Augmentation {idx + 1} / {len(dfs)}")
            augmentation_datasets.append(
                SequentialRecommendationTrainsetWithAugmentation(
                    df,
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
            )
        self.augmentation_datasets = augmentation_datasets
        self.construct_full_subset_range()
    
    def __len__(self) :
        total_length = 0
        for dataset in self.augmentation_datasets:
            total_length += len(dataset)
        return total_length
    
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor] :
        for dataset in self.augmentation_datasets:
            if idx < len(dataset):
                return dataset[idx]
            else :
                idx -= len(dataset)
    
    
    def construct_full_subset_range(self):
        full_subset_range_via_augmentation = []
        
        for dataset in self.augmentation_datasets:
            full_subset_range_via_augmentation.append(dataset.listed_range)
        full_subset_range_via_augmentation = full_subset_range_via_augmentation
        
        import numpy as np
        full_subset_range_via_augmentation = np.stack(full_subset_range_via_augmentation)
        row_offsets = [0]
        for i in range(1, len(full_subset_range_via_augmentation)):
            row_offsets.append(full_subset_range_via_augmentation[i-1][-1])
        for i in range(1, len(full_subset_range_via_augmentation)):
            row_offsets[i] = row_offsets[i-1] + row_offsets[i]   
        
        ## add offset to each row
        self.full_subset_range_via_augmentation = full_subset_range_via_augmentation +\
                                                  np.array(row_offsets).reshape(-1,1)
        
        
    
    def _from_user_idx_get_user_subset_range(self, idx):
        
        combined_ranges = list()       
        offset = 0
        for dataset in self.augmentation_datasets:
            current_range = dataset._from_user_idx_get_user_subset_range(idx)
            current_range = [x + offset for x in current_range]
            combined_ranges += current_range
            offset += len(dataset)
        return combined_ranges
    
    
    def _from_user_idx_and_augmentation_type_idx_get_subset_range(self, idx, aug_type_idx):
            
        dataset = self.augmentation_datasets[aug_type_idx]
        offset = 0
        for i in range(aug_type_idx):
            offset += len(self.augmentation_datasets[i])
        
        user_subset_range = dataset._from_user_idx_get_user_subset_range(idx)
        return [x + offset for x in user_subset_range]
            
    
    def get_full_subset_range(self):
        
        list_of_ranges = []
        import numpy as np
        
        range_starts = self.full_subset_range_via_augmentation[:,:-1]
        range_ends = self.full_subset_range_via_augmentation[:,1:]
        
        for j in range(range_starts.shape[1]):
            user_range = []
            for i in range(self.full_subset_range_via_augmentation.shape[0]):
                user_range += list(range(range_starts[i][j], range_ends[i][j]))
            list_of_ranges.append(user_range)
        
        return list_of_ranges
    

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


def _load_aug_types_map(
    augmentation_args : CN
) -> Dict[any, any] :
    
    aug_types_count = augmentation_args.aug_types_count
    aug_types_map = pd.read_csv(augmentation_args.aug_types_map_path)
    ## iter over dataframe and create a dictionary
    aug_dict = {}
    
    for idx, row in aug_types_map.iterrows():
        aug_label = row["label"]
        aug_start = row["start"]
        aug_end = row["end"]
        aug_range = range(aug_start, aug_end)
        aug_dict[aug_label] = aug_range
        if idx > aug_types_count :
            break
        
    ## check if the map is consistent with the count
    assert len(aug_dict) == aug_types_count, \
        "Augmentation Map and Augmentation Count is not Consistent"
    
    return aug_dict


def cut_df_by_aug_idx(
    df : pd.DataFrame,
    augmentation_column : str,
    max_augmentation_idx : int
) -> pd.DataFrame :
    if max_augmentation_idx > 0 :
        df = df[df[augmentation_column] <= max_augmentation_idx]
        df = df.reset_index(drop=True)
    return df


def load_augmentation_df(
    augmentation_args : CN,
    df_folder_path : str
) -> pd.DataFrame :
    """
    augmentation_args : CN : Augmentation Arguments
    df_path : str : Path to the Augmentation DataFrame
    -------------------------------------------------
    load single augmentation dataframe
    with regards of max_augmentation_idx
    with regards of remove_original & is_zero_original
    
    add augmentation columnn if not exists
    """
    df_path = os.path.join(df_folder_path, 'train.csv')
    df = pd.read_csv(df_path)
    
    augmentation_column = augmentation_args.augmentation_column
    
    if augmentation_column not in df.columns :
        ## add augmentation column with zero value
        df[augmentation_column] = 0
        
    df = cut_df_by_aug_idx(df, augmentation_column, augmentation_args.max_augmentation_idx)
    
    return df


def build_training_dataset(
    df : pd.DataFrame,
    user_column : str,
    item_column : str,
    interaction_column : str,
    timestamp_column : str,
    min_sequence_length : int = 5,
    max_sequence_length : int = 200,
    user_num : int = None,
    item_num : int = None,
    padding_value : int = 0,
    augmentation_args : CN = None,
) :
    if augmentation_args.use_augmentation :
        # drop some augmentation above max_augmentation_idx
        # unless load all
        if augmentation_args.is_multiple :
            ## ignore the given df and read multiple dfs specified at
            del df
            ## data.augmentation_args.df_paths
            dfs = []
            for folder_path in augmentation_args.df_paths :
                dfs.append(load_augmentation_df(augmentation_args, folder_path))
                
            trainset = SequentialRecommendationTrainsetWithMultipleAugmentation(
                dfs,
                user_column,
                item_column,
                interaction_column,
                timestamp_column,
                augmentation_args.augmentation_column,
                min_sequence_length,
                max_sequence_length,
                user_num,
                item_num,
                padding_value
            )
        else :
            df = cut_df_by_aug_idx(df, augmentation_args.augmentation_column, augmentation_args.max_augmentation_idx)
            trainset = SequentialRecommendationTrainsetWithAugmentation(
                df,
                user_column,
                item_column,
                interaction_column,
                timestamp_column,
                augmentation_args.augmentation_column,
                min_sequence_length,
                max_sequence_length,
                user_num,
                item_num,
                padding_value
            )
    else :
        trainset = SequentialRecommendationTrainset(
            df,
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
    
    return trainset


def make_partitioned_df(
    df_path : pd.DataFrame,
    user_column : str,
    timestamp_column : str,
    save_partitioned_df_path : str = None    
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    """
    Make partitioned dataframe and save it to the save_partitioned_df_path
    
    args :
        df_path : str : path to the dataframe
        user_column : str : column name for user
        timestamp_column : str : column name for timestamp
        save_partitioned_df_path : str : path to save the partitioned dataframe \
            if None, it will not save the partitioned dataframe
    """
    logger.info("SRData : Partitioned Dataframe not found, Splitting Dataframe")
    # Sort Dataframe by user and timestamp
    df = pd.read_csv(df_path, header = 0, sep = '/t')
    df = df.sort_values([user_column, timestamp_column])
        
    # Split Dataframe in to train, valid, test according to leave one out strategy
    train_df, valid_df, test_df = split_dataframe_into_train_valid_test(df, user_column)
    
    if save_partitioned_df_path :
        logger.info("SRData : Saving Partitioned Dataframe to {}".format(save_partitioned_df_path))
        if not os.path.exists(save_partitioned_df_path):
            os.makedirs(save_partitioned_df_path)
        
        train_df.to_csv(os.path.join(save_partitioned_df_path, 'train.csv'), index = False)
        valid_df.to_csv(os.path.join(save_partitioned_df_path, 'valid.csv'), index = False)
        test_df.to_csv(os.path.join(save_partitioned_df_path, 'test.csv'), index = False)

    return train_df, valid_df, test_df


def make_sr_dataset(
    df_path : str,
    user_column : str,
    item_column : str,
    interaction_column : str,
    timestamp_column : str,
    augmentation_args : CN = None,
    partitioned_df_path : str = None,
    save_partitioned_df_path : str = None,
    min_sequence_length : int = None, 
    max_sequence_length : int = None,
    user_num : int = None,
    item_num : int = None,
    padding_value : int = 0,
    config : CN = None
) -> Tuple[SequentialRecommendationTrainset,
           SequentialRecommendationValidset,
           SequentialRecommendationValidset] :
    
    try :
        train_df = pd.read_csv(os.path.join(partitioned_df_path, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(partitioned_df_path, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(partitioned_df_path, 'test.csv'))
    except :
        if augmentation_args.use_augmentation :
            logger.warning("SRData with Augmentation is only availiable for reading")
            raise ValueError("SRData with Augmentation is only availiable for reading")
        else :
            train_df, valid_df, test_df = make_partitioned_df(df_path,
                                                              user_column,
                                                              timestamp_column,
                                                              save_partitioned_df_path)
    
    ## Check the consistancy of user number
    check_user_num_consistancy(user_column, train_df, valid_df, test_df)
    ## Fix user_num and item_num according to test_df
    user_num = len(test_df[user_column].unique())
    item_num = max(test_df[item_column].unique())
    
    trainset = build_training_dataset(
        train_df,
        user_column,
        item_column,
        interaction_column,
        timestamp_column,
        min_sequence_length,
        max_sequence_length,
        user_num,
        item_num,
        padding_value,
        augmentation_args
    )
    
    """
    import time
    from torch.utils.data import Dataset, Subset
    
    
    full_subset_range = trainset.get_full_subset_range()
    
    random_user = trainset.df[user_column].sample(1).values[0]
    start = time.time()
    subset_range = trainset._from_user_idx_get_user_subset_range(random_user)
    subset_trainset = Subset(trainset, subset_range)
    subset_dataloader = torch.utils.data.DataLoader(subset_trainset, batch_size = 1, shuffle = False)
    end = time.time()
    print("Time for Single User Subset Range Query with Hash : ", end - start)
    print(f"total {len(subset_range)} interactions to check for")
    
    
    from federatedscope.contrib.model.sasrec import SASRec, call_sasrec
    model = call_sasrec(config.model, None)
    model.to("cuda:0")
    
    start = time.time()
    ## try all forward for single user and calculate time
    for batch in subset_dataloader :
        
        item_seq = batch["item_seq"].to("cuda:0")
        item_seq_len = batch["item_seq_len"].to("cuda:0")
        target_item = batch["target_item"].to("cuda:0")
        
        outputs = model(item_seq, item_seq_len)
    
    end = time.time()
    print("Time for Single User Forward : ", end - start)
    
    ## getting augmentation range
    start = time.time()
    subset_range_for_aug = trainset._from_user_idx_and_augmentation_type_idx_get_subset_range(random_user, 0)
    
    subset_trainset_for_aug = Subset(trainset, subset_range_for_aug)
    subset_dataloader_for_aug = torch.utils.data.DataLoader(subset_trainset_for_aug, batch_size = 1, shuffle = False)
    end = time.time()
    print("Time for Single User Subset Range for Augmentation : ", end - start)
    print(f"total {len(subset_range_for_aug)} interactions to check for")
    ## 
    start = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    for batch in subset_dataloader_for_aug :
        item_seq = batch["item_seq"].to("cuda:0")
        item_seq_len = batch["item_seq_len"].to("cuda:0")
        target_item = batch["target_item"].to("cuda:0")
        
        outputs = model(item_seq, item_seq_len)
        test_item_emb = model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        loss = criterion(logits, target_item)
        loss.backward()
        
    end = time.time()
    print("Time for Single User Forward & backward for Augmentation : ", end - start)
    
    
    #end_2 = time.time()
    """
    
    
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
        config.data.augmentation_args,
        config.data.partitioned_df_path,
        config.data.save_partitioned_df_path,
        config.data.min_sequence_length,
        config.data.max_sequence_length,
        config.data.user_num,
        config.data.item_num,
        config.data.padding_value,
        config
    )
    ## NOTE!
    ## This part is too slow for Large Dataset
    ## Consider skipping this part when using Shadow Runner
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator((trainset, validset, testset))
    
    return fs_data, config


def call_sr_data(config, client_cfgs) :
    if config.data.type == 'sr_data' :
        data, modified_config = load_sr_data(config, client_cfgs)
        return data, modified_config
    

register_data('sr_data', call_sr_data)
