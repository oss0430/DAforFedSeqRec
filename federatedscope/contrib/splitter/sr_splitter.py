import torch
from torch.utils.data import Dataset, Subset

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class SRSpliiter(BaseSplitter):
    ## Based on dataset type SequentialRecommendationDataset at federatedscope/contrib/data/sr_data.py
    
    def __init__(self, client_num):
        super(SRSpliiter, self).__init__(client_num)
        
        
    def __call__(self, dataset : Dataset, prior=None, **kwargs):
        user_ids = dataset.df[dataset.user_column].unique()
        
        ## length of user_ids must match with client_num
        assert len(user_ids) == self.client_num, \
            f"Number of users in dataset ({len(user_ids)}) must match with client_num ({self.client_num})"
        idx_range = len(user_ids)
        
        data_list = []
        
        for idx in range(0, idx_range) :
            client_dataset = Subset(dataset, [idx])
            data_list.append(client_dataset)

        return data_list
    
    
def call_sr_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'sr_splitter':
        splitter = SRSpliiter(client_num, **kwargs)
        return splitter
    

register_splitter('sr_splitter', call_sr_splitter)