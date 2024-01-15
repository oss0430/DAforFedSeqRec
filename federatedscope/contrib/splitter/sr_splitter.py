
import torch

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class SRSpliiter(BaseSplitter):
    ## Based on dataset type SequentialRecommendationDataset at federatedscope/contrib/data/sr_data.py
    
    def __init__(self, client_num):
        super(SRSpliiter, self).__init__(client_num)
        
        
    def __call__(self, dataset : torch.utils.data.Dataset, prior=None, **kwargs):
        user_ids = dataset.df[dataset.user_column].unique()
        
        ## length of user_ids must match with client_num
        assert len(user_ids) == self.client_num, \
            f"Number of users in dataset ({len(user_ids)}) must match with client_num ({self.client_num})"
            
        data_list = [dataset[user_id] for user_id in user_ids]
            
        return data_list
    
    
def call_sr_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'sr_splitter':
        splitter = SRSpliiter(client_num, **kwargs)
        return splitter
    

register_splitter('sr_splitter', call_sr_splitter)