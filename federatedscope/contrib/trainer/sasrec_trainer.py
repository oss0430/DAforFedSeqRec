from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer

import torch

class SASRecTrainer(BaseTrainer):
    
    def __init__(self, model : torch.nn.Module, data, device, **kwargs) :
        self.model = model
        self.data = data
        self.device = device
        self.kwargs = kwargs
        self.criterion = torch.nn.CrossEntropyLoss()
    
      
    def train(self):
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.to(self.device)
        self.model.train()
        
        total_loss = num_samples = 0
        
        for item_seq, item_seq_len, target_item in self.data['train'] :
            item_seq, item_seq_len, target_item = item_seq.to(self.device), item_seq_len.to(self.device), target_item.to(self.device)
            
            outputs = self.model(item_seq, item_seq_len)
            target_embedding = self.model.item_embedding(target_item)
            loss = self.criterion(outputs, target_embedding)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * target_item.shape[0]
            num_samples += target_item.shape[0]
        
        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(num_samples)}

    
    def ndcg_k(self, scores, target_item, k=10):
        
        sorted_recommendation = torch.argsort(scores, dim = -1, descending=True)
        item_ranks = sorted_recommendation.argsort(dim = -1)
        target_item_rank = item_ranks[target_item]
        
        dcg = 1 / torch.log2(target_item_rank + 2)
        ## TODO:
        """ FIX THIS !!!!"""
        
    
    def hit_k(self, scores, target_item, k=10):
        
        sorted_recommendation = torch.argsort(scores, dim = -1, descending=True)
        item_ranks = sorted_recommendation.argsort(dim = -1)
        target_item_rank = item_ranks[target_item]
        
        return (target_item_rank < k).nonzero().item() / target_item.shape[0]
        
        
    def evaluate(self, target_data_split_name='valid'):
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            total_loss = num_samples = 0
            
            for item_seq, item_seq_len, target_item in self.data[target_data_split_name]:
                item_seq, item_seq_len, target_item = item_seq.to(self.device), item_seq_len.to(self.device), target_item.to(self.device)
                
                outputs = self.model(item_seq, item_seq_len)
                target_embedding = self.model.item_embedding(target_item)
                loss = self.criterion(outputs, target_embedding)
                
                total_loss += loss.item() * target_item.shape[0]
                num_samples += target_item.shape[0]    

                scores = self.model.full_sort_predict(item_seq, item_seq_len)
                total_ndcg = self.ndcg_k(scores, target_item, k=50)
                
                
    
    def update(self, model_parameters, strict=False):
        self.model.load_state_dict(model_parameters, strict)


    def get_model_para(self):
        return self.model.cpu().state_dict()
    
    
def call_sasrec_trainer(trainer_type):
    if trainer_type == "sasrec":
        return SASRecTrainer
    

register_trainer('sasrec', call_sasrec_trainer)