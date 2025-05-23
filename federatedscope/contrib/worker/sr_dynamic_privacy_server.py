from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from collections import defaultdict
import logging
import copy
import os
import torch
import pickle
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DynamicPrivacySRServer(Server):
    
    """
    Dynamic Privacy SR Server is a setting where user uploads part of their data to the server.
    In FedSR setting, user are requested to agree to share their data with the server, upon agreed the data is uploaded to the server.
    However after the initial upload, user can choose to not to share newly collected data.
    The server will not have access to the newly collected data.
    
    Important Parameter 
    ============================================================
    self.client_privacy_budgets | Dict[str, float] | Privacy Budget for each client unknown to the server
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super(DynamicPrivacySRServer, self).__init__(ID=ID,
                                             state=state,
                                             data=data,
                                             model=model,
                                             config=config,
                                             client_num=client_num,
                                             total_round_num=total_round_num,
                                             device=device,
                                             strategy=strategy,
                                             **kwargs)
        self.client_privacy_budgets = defaultdict(float)
        
    
    
    def global_update(self) -> None :
        """
        Update the global model using uploaded global dataset
        """
        raise NotImplementedError
    
    
    
    def naive_global_update(self) -> None :
        """
        naively update the global model using uploaded global dataset
        """
        raise NotImplementedError
    
    
    def naive_contrastive_update(self) -> None :
        """
        naively update the global model using uploaded global dataset
        with contrastive learning on item embeddings
        """
        raise NotImplementedError
    
    
    def model_contrastive_update(self) -> None :
        """
        Update the global model using uploaded global dataset
        with dynamic privacy constraints
        """
        raise NotImplementedError
    
    
    def imputated_data_update(self) -> None :
        """
        After Inputating using the global GAN model,
        update the global model using inputated dataset
        """
        
        raise NotImplementedError
    


class NaiveUpdateDynamicPrivacySRServer(DynamicPrivacySRServer):
    
    def global_update(self) -> None :
        """
        Update the global model using uploaded global dataset
        """
        raise NotImplementedError