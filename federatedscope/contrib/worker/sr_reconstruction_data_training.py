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
import time



class ReconstAugClient(Client):
    
    
    def _get_reconstruction_loader(self, labels, batch_size):
        """
        Get the dataloader for reconstruction
        """
        class ReconstructionDataset(Dataset):
            def __init__(self, labels, cfg):
                self.labels = labels
                self.gaussian_input_embedding = torch.normal(0, 1, size=(len(labels),
                                                                         cfg.model.max_sequence_length, 
                                                                         cfg.model.hidden_size))
                self.item_seq_len = torch.randint(1, cfg.model.max_sequence_length, size=(len(labels),1))
                
            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {"label" : self.labels[idx], 
                        "input_embedding" : self.gaussian_input_embedding[idx],
                        "item_seq_len" : self.item_seq_len[idx]}

        dataset = ReconstructionDataset(labels, self._cfg)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    
    def _select_label(self, num : int) -> torch.Tensor: 
        item_num = self._cfg.model.item_num
        padding_idx = 0
        target_item_id = self._cfg.attack.target_item_id
        
        labels = []
        while True :
            if len(labels) == num:
                break
            random_label = torch.randint(1, item_num, size=(self._cfg.data.batch_size,))
            if random_label != target_item_id:
                labels.append(random_label)
        
        ## convert to 1 d tensor
        labels = torch.cat(labels, dim = 0)
        
        return labels    
    
    
    def _select_item_seq_len(self, num : int) -> List[torch.Tensor]:
        max_seq_lenth = self._cfg.model.max_sequence_length
        return [torch.randint(1, max_seq_lenth, size=(self._cfg.data.batch_size,1)) for _ in range(num)]
        
    
    def _get_mean_gradient(self, model_delta : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get the mean gradient of the model delta by deviding with
        local step size, and epoch
        """
        mean_gradient = {}
        for key in model_delta.keys():
            mean_gradient[key] = model_delta[key] / (self._cfg.train.local_update_steps * self._cfg.train.epoch)
            mean_gradient[key] = mean_gradient[key].to(self.device)
        
        return mean_gradient
    
    
    def _reconstruct_data(
            self,
            model_delta : Dict[str, torch.Tensor],
            num : int) -> List[Dict[torch.Tensor, Dict[str, any]]]:
        """
        The attacker will generate embedding that was infer from previous and past model
        """
        
        local_update_steps = self._cfg.train.local_update_steps
        learning_rate = self._cfg.train.optimizer.lr
        criterion_type = self._cfg.criterion.type ## this is a type decleration
        
        ## get reconstruction details
        reconstruction_batch = self._cfg.attack.reconstruction_batch
        mean_gradient = self._get_mean_gradient(model_delta)
        
        ## 1. first get labels
        labels = []
        
        ## 2. get data loader
        reconstruction_loader = self._get_reconstruction_loader(
            labels = labels,
            batch_size = reconstruction_batch
        )
        
        model = self._trainer.get_model()
        
        model.eval()  ## self.trainer.update
        model.to(self.device)
        
        ## 3. per batch generate
        for batch in reconstruction_loader:
            input_embedding = batch['input_embedding']
            item_seq_len = batch['item_seq_len']
            label = batch['label']
            
        model.to("cpu")
            
            
            
        ## 3.3 add reconstructed data to training set
        self._trainer.add_reconstructed_data(reconstructed_data)
        
        
        return reconstructed_data    
    
    
    def callback_funcs_for_model_para(self, message: Message):
        
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        
        if round != 0:
            ## Add reconstructed data to client's training set
            previous_model_para = self.trainer.get_model_para()
            new_model_para = content
            model_delta = self._calculate_model_delta(previous_model_para, new_model_para)
            
            ## 1. generate reconstructed data
            reconstructed_data = self._reconstruct_data(
                model_delta = model_delta,
                num = self._cfg.attack.num_reconstructed_data
            )
            
            ## 2. add reconstructed data to training set
            self.trainer.add_reconstructed_data(reconstructed_data)
            
        return super().callback_funcs_for_model_para(message)


def call_my_worker(method):
    
    if method == 'reconst_aug':
        worker_builder = {'client': ReconstAugClient, 'server': Server}
        return worker_builder

register_worker('reconst_aug', call_my_worker)