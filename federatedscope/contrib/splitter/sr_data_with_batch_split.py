import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from typing import List
from copy import deepcopy

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class GroupSRSpliter(BaseSplitter):
    
    
    def __init__(self, client_num, **kwargs):
        self.group_size = kwargs.get('group_size')
        self.random_split = kwargs.get('random_split')
        super(GroupSRSpliter, self).__init__(client_num)
        
        
    def __call__(self, dataset : Dataset, prior=None, **kwargs):
        
        ## prior is a list of user ids it may be random and missing in the dataset
        
        if prior is not None and prior is not []:
            client_per_user_ids = prior
            data_list = []
            
            for client_ids in client_per_user_ids:
                client_dataset = Subset(dataset, client_ids)
                data_list.append(client_dataset)
                
            return data_list
        else :
            
            data_list = []
            if self.random_split :
                ## in sr_data class we keep track of index of user_idx,
                ## just distributing the index is enough
                user_num = len(dataset.df[dataset.user_column].unique().tolist())
                idxs = list(range(0, user_num))
                random.shuffle(idxs)
                
            else :
                user_num = len(dataset.df[dataset.user_column].unique().tolist())
                idxs = list(range(0, user_num))
            
            client_per_user_ids = []
            for i in range(0, self.client_num):
                client_user_ids = []
                for j in range(0, self.group_size):
                    try :
                        client_user_ids.append(idxs[i * self.group_size + j])
                    except IndexError as e:
                        print("something went wrong")
                        #None ## When there are none left
                client_per_user_ids.append(client_user_ids)
                
            
            for client_ids in client_per_user_ids:
                client_dataset = Subset(dataset, client_ids)
                data_list.append(client_dataset)
                
            return data_list


def call_group_sr_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'group_sr_splitter':
        splitter = GroupSRSpliter(client_num, **kwargs)
        return splitter
    
    
register_splitter('group_sr_splitter', call_group_sr_splitter)
        