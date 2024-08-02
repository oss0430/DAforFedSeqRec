#from federatedscope.core.trainers.enums import MODE
from federatedscope.register import register_metric
from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EVAL_BATCH_SIZE = 2048


class Evalset(Dataset):
    
    def __init__(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray) :
        self.y_true = y_true
        self.y_pred = y_pred
    
    def __len__(self):
        return len(self.y_true)
    
    
    def __getitem__(self, idx):
        return self.y_true[idx], self.y_pred[idx]


def batch_eval(
    y_true : np.ndarray, ## Batch of target items  B
    y_pred : np.ndarray, ## Batch of predicted items B X N
    eval_func : Any,
    k : int = 5,
    use_gpu : bool = False,
    device : str = 'cpu'
) -> np.ndarray : ## Batch of ndcg_k scores B X 1
    
    evaluation_set = Evalset(y_true, y_pred)
    eval_loader = DataLoader(evaluation_set, batch_size = EVAL_BATCH_SIZE, shuffle = False)
    
    results = []
    
    for batch in eval_loader :
        y_true_batch, y_pred_batch = batch
        if use_gpu :
            y_true_batch = y_true_batch.to(device)
            y_pred_batch = y_pred_batch.to(device)
        batch_result = eval_func(y_true_batch, y_pred_batch, k = k)
        results.append(batch_result)
    
    results = torch.concatenate(results, axis = 0).mean().to('cpu').item()
    
    del evaluation_set
    del eval_loader
    
    torch.cuda.empty_cache()
    return results


def ndcg_k_from_top_N(
    y_true : torch.Tensor, ## Batch of target items  B
    top_N_indices : torch.Tensor, ## Batch of top N predicted items B X N
    k : int = 5
) -> torch.Tensor : 
    """
    Calculate NDCG@k from top N indices
    """
    device = y_true.device
    batch_size, num_items = top_N_indices.shape
    top_k_indices = top_N_indices[:, :k]
    top_k_indices_is_y_true = torch.where(top_k_indices == y_true, True, False)
    ## 1 to k
    mask_ranks = torch.arange(1, k+1, device = device).unsqueeze(0).expand(batch_size, k)
    
    true_ranks = (top_k_indices_is_y_true * mask_ranks).sum(dim = -1)
    
    ndcg = 1 / torch.log2(true_ranks.float() + 2)
    
    # Set NDCG to 0 for items not in top k
    ndcg[true_ranks == 0] = 0
    
    return ndcg


def recall_k_from_top_N(
    y_true : torch.Tensor, ## Batch of target items  B
    top_N_indices : torch.Tensor, ## Batch of top N predicted items B X N
    k : int = 5
) -> torch.Tensor :
    """
    Calculate Recall@k from top N indices
    """
    device = y_true.device
    batch_size, num_items = top_N_indices.shape
    top_k_indices = top_N_indices[:, :k]
    top_k_indices_is_y_true = torch.where(top_k_indices == y_true, True, False)
    
    recall = top_k_indices_is_y_true.sum(dim = 1).float()
    
    return recall



def ndcg_k(
    y_true : torch.tensor, ## Batch of target items  B
    y_pred : torch.tensor, ## Batch of predicted items B X N
    k : int = 5
) -> np.ndarray : ## Batch of ndcg_k scores B X 1
    device = y_pred.device
    batch_size, num_items = y_pred.shape
    
    _, top_k_indices = torch.topk(y_pred, k, dim = 1)
    
    top_k_indices_is_y_true = torch.where(top_k_indices == y_true, True, False)
    ## 1 to k
    mask_ranks = torch.arange(1, k+1, device = device).unsqueeze(0).expand(batch_size, k)
    
    true_ranks = (top_k_indices_is_y_true * mask_ranks).sum(dim = -1)
    
    ndcg = 1 / torch.log2(true_ranks.float() + 2)
    
    # Set NDCG to 0 for items not in top k
    ndcg[true_ranks == 0] = 0
    
    del mask_ranks
    del true_ranks
    del top_k_indices_is_y_true
    del top_k_indices
    #ndcg[rank > k] = 0
    return ndcg
    
    """
    ## Original NDCG
    #item_recommendation_rank = np.argsort(np.argsort(-y_pred, axis = 1), axis = 1) ## - for descending ordering
    #rank_for_each_true = np.take_along_axis(item_recommendation_rank, y_true.reshape(-1,1), axis = 1)
    
    ## Next item Prediction only 1 true item
    #ndcgs = np.zeros(rank_for_each_true.shape[0])
    #for i in range(rank_for_each_true.shape[0]):
    #    if rank_for_each_true[i] > k :
    #        ndcg = 0
    #    else :
    #        ndcg = 1 / np.log2(rank_for_each_true[i] + 2)
    #    ndcgs[i] = ndcg
        
    #return ndcgs     
    """
        

def recall_k(
    y_true : np.ndarray, ## Batch of target items  B
    y_pred : np.ndarray, ## Batch of predicted items B X N
    k : int = 5
) -> np.ndarray : ## Batch of ndcg_k scores B X 1
    device = y_pred.device
    batch_size, num_items = y_pred.shape
    
    # Get top k predictions
    _, top_k_indices = torch.topk(y_pred, k, dim=1)
    top_k_indices_is_y_true = torch.where(top_k_indices == y_true, True, False)
    
    recall = top_k_indices_is_y_true.sum(dim = 1).float()
    
    del top_k_indices_is_y_true
    del top_k_indices
    
    return recall
    """
    ## original recall
    
    item_recommendation_rank = np.argsort(np.argsort(-y_pred, axis = 1), axis = 1) ## - for descending ordering
    rank_for_each_true = np.take_along_axis(item_recommendation_rank, y_true.reshape(-1,1), axis = 1)
    
    recalls = np.zeros(rank_for_each_true.shape[0])
    for i in range(rank_for_each_true.shape[0]):
        if rank_for_each_true[i] > k :
            recall = 0
        else :
            recall = 1
        recalls[i] = recall

    return recalls
    """


def load_ndcg_10(ctx , y_true : np.ndarray, y_pred : np.ndarray, y_prob, **kwargs):
    """
    Load ndcg@10 metric.
    """
    if ctx.cur_mode in ['train', 'finetune'] :
        ## Do not calculate evaluation for training for faster training
        return 0.0
    result = batch_eval(y_true=y_true,
                         y_pred=y_pred,
                         eval_func=ndcg_k_from_top_N,
                         k = 10,
                         use_gpu = ctx.cfg.use_gpu,
                         device = ctx.cfg.device)
    
    #results = ndcg_k(y_true, y_pred, k = 10).mean()
    
    return result


def load_ndcg_20(ctx , y_true : np.ndarray, y_pred : np.ndarray, y_prob, **kwargs):
    """
    Load ndcg@20 metric.
    """
    if ctx.cur_mode in ['train', 'finetune'] :
        ## Do not calculate evaluation for training for faster training
        return 0.0
    result = batch_eval(y_true=y_true, 
                         y_pred=y_pred, 
                         eval_func=ndcg_k_from_top_N, 
                         k = 20,
                         use_gpu = ctx.cfg.use_gpu,
                         device = ctx.cfg.device)
    
    #results = ndcg_k(y_true, y_pred, k = 20).mean()
    
    return result


def load_recall_10(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Load recall@10 metric.
    """
    if ctx.cur_mode in ['train', 'finetune'] :
        ## Do not calculate evaluation for training for faster training
        return 0.0
    result = batch_eval(y_true=y_true, 
                         y_pred=y_pred, 
                         eval_func=recall_k_from_top_N, 
                         k = 10,
                         use_gpu = ctx.cfg.use_gpu,
                         device = ctx.cfg.device)
    
    #results = recall_k(y_true, y_pred, k = 10).mean()
    
    return result


def load_recall_20(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Load recall@20 metric.
    """
    if ctx.cur_mode in ['train', 'finetune'] :
        ## Do not calculate evaluation for training for faster training
        return 0.0
    result = batch_eval(y_true=y_true, 
                         y_pred=y_pred,
                         eval_func=recall_k_from_top_N,
                         k = 20,
                         use_gpu = ctx.cfg.use_gpu,
                         device = ctx.cfg.device)
    
    #results = recall_k(y_true, y_pred, k = 20).mean()
    
    return result


def call_recall_10(types):
    if 'recall_10' in types:
        the_larger_the_better = True
        return 'recall_10', load_recall_10, the_larger_the_better
    
    
def call_recall_20(types):
    if 'recall_20' in types:
        the_larger_the_better = True
        return 'recall_20', load_recall_20, the_larger_the_better
    
    
def call_ndcg_10(types):
    if 'ndcg_10' in types:
        the_larger_the_better = True
        return 'ndcg_10', load_ndcg_10, the_larger_the_better
    
    
def call_ndcg_20(types):
    if 'ndcg_20' in types:
        the_larger_the_better = True
        return 'ndcg_20', load_ndcg_20, the_larger_the_better


register_metric('recall_10', call_recall_10)
register_metric('recall_20', call_recall_20)
register_metric('ndcg_10', call_ndcg_10)
register_metric('ndcg_20', call_ndcg_20)




