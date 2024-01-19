import random
import numpy as np
from torch.utils.data import Dataset, Subset

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class SRSpliiter(BaseSplitter):
    ## Based on dataset type SequentialRecommendationDataset at federatedscope/contrib/data/sr_data.py
    
    def __init__(self, client_num):
        super(SRSpliiter, self).__init__(client_num)
        
        
    def __call__(self, dataset : Dataset, prior=None, **kwargs):
        user_ids = dataset.df[dataset.user_column].unique()
        
        ## length of user_ids must be equal or lower than the client_num
        assert len(user_ids) >= self.client_num, \
            f"Number of users in dataset ({len(user_ids)}) must be equal or lower than client_num ({self.client_num})"
        
        ## TODO: 
        # Add function when client_num is below the number of users in dataset
        # When that is the case, use prior which is a list to create idx_range
        data_list = []
        if len(user_ids) == self.client_num :
            ## TODO:
            # No need to user prior or do we?
            # When performing random sampling we might have to use it 
            idxs = range(0, len(user_ids))
            
            for idx in idxs :
                idx_range = [idx]
                client_dataset = Subset(dataset, idx_range)
                data_list.append(client_dataset)
            return data_list
        
        elif len(user_ids) > self.client_num :
            if prior :         
                ## prior is the list of client_ids(list) for each train_split
                assert len(prior) == self.client_num, \
                    f"Number of client splited in train do not match with client_num ({self.client_num})"
                
                for idx_range in prior:
                    ## we must pass the index of the user_ids not the user_ids
                    ## We consider each client_id - 1 == index
                    idx_range = [idx - 1 for idx in idx_range]
                    client_dataset = Subset(dataset, idx_range)
                    data_list.append(client_dataset)    
                return data_list
            else :
                ## Prior needs to be created we are currently first time splitting
                idxs = random.sample(range(0,len(user_ids)), k=len(user_ids))[:self.client_num]
                
                for idx in idxs :
                    idx_range = [idx]
                    client_dataset = Subset(dataset, idx_range)
                    data_list.append(client_dataset)
                return data_list
        else :
            raise Exception("Number of users in dataset is less than client_num")
                            
    
def call_sr_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'sr_splitter':
        splitter = SRSpliiter(client_num, **kwargs)
        return splitter
    

register_splitter('sr_splitter', call_sr_splitter)