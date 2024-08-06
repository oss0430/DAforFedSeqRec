import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from typing import List
from copy import deepcopy

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class SRSpliiter(BaseSplitter):
    
    """
        Splitting SequentialRecommendationDataset
        federatedscope.contrib.data.sr_data.SequentialRecommendationDataset
        
        mainly matches the train, valid and test user_id
        for each client. (Client is the User)
    """
    
    def __init__(self, client_num):
        super(SRSpliiter, self).__init__(client_num)
        
        
    def __call__(self, dataset : Dataset, prior=None, **kwargs):
        user_ids = dataset.df[dataset.user_column].unique()

        data_list = []
        if len(user_ids) == self.client_num :
            ## TODO:
            # No need to user prior or do we?
            # When performing random sampling we might have to use it 
            idxs = range(0, len(user_ids))
            
            for idx in idxs :
                idx_range = dataset._from_user_idx_get_user_subset_range(idx)
                client_dataset = Subset(dataset, idx_range)
                data_list.append(client_dataset)
            return data_list
        
        elif len(user_ids) > self.client_num :
            
            if prior :         
                for current_split_user_ids in prior:
                    idx_range = []
                    for user_id in current_split_user_ids :
                        idx_range += dataset._from_user_idx_get_user_subset_range(user_id)
                    client_dataset = Subset(dataset, idx_range)
                    data_list.append(client_dataset)    
                return data_list
            else :
                idxs = range(0, len(user_ids))     
                for idx in idxs :
                    idx_range = dataset._from_user_idx_get_user_subset_range(idx)
                    client_dataset = Subset(dataset, idx_range)
                    data_list.append(client_dataset)
                return data_list
        else : 
            ## NOTE : This part do not support random client setup
            ## Assume client_id == user_id
            ## Fill the rest of the client with empty dataset : Padding Value 0
            idxs = range(0, self.client_num)
            empty_idxs = range(len(user_ids)+1, self.client_num+1)
            
            def create_empty_sr_dataset(dataset, empty_idxs : List[int]):
                ## Refer to SequentialRecommendationDataset at federatedscope/contrib/data/sr_data.py
                ## We need to create a new dataset with same parameters as the original one
                padding_value = dataset.padding_value
                columns = dataset.df.columns
                user_id_location = columns.get_loc(dataset.user_column)
                
                empty_data_values = []
                for idx in empty_idxs :
                    empty_value = np.full(len(columns), fill_value= padding_value)
                    empty_value[user_id_location] = idx
                    empty_data_values.append(empty_value) ## empty sequence
                    empty_data_values.append(empty_value) ## empty target
                empty_data_values = np.vstack(empty_data_values)
                
                empty_df = pd.DataFrame(empty_data_values, columns=columns)
                
                new_empty_dataset = deepcopy(dataset)
                new_empty_dataset.df = pd.concat([new_empty_dataset.df, empty_df])
                new_empty_dataset.user_num = len(empty_idxs)
                
                return new_empty_dataset
                
            empty_dataset = create_empty_sr_dataset(Subset(dataset, [0]).dataset, empty_idxs)
            
            for idx in idxs :
                idx_range = [idx]
                if idx < len(user_ids) :
                    ## regular client with dataset
                    client_dataset = Subset(dataset, idx_range)
                else :
                    ## empty client with no dataset
                    client_dataset = Subset(empty_dataset, idx_range)

                data_list.append(client_dataset)
                
            return data_list
                            
    
def call_sr_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'sr_splitter':
        splitter = SRSpliiter(client_num, **kwargs)
        return splitter
    

register_splitter('sr_splitter', call_sr_splitter)