from federatedscope.register import register_metric
from typing import Any, Dict, Tuple
import numpy as np



def ndcg_k(
    y_true : np.ndarray, ## Batch of target items  B
    y_pred : np.ndarray, ## Batch of predicted items B X N
    k : int = 5
) -> np.ndarray : ## Batch of ndcg_k scores B X 1

    item_recommendation_rank = np.argsort(np.argsort(-y_pred, axis = 1), axis = 1) ## - for descending ordering
    rank_for_each_true = np.take_along_axis(item_recommendation_rank, y_true.reshape(-1,1), axis = 1)
    
    ## Next item Prediction only 1 true item
    ndcgs = np.zeros(rank_for_each_true.shape[0])
    for i in range(rank_for_each_true.shape[0]):
        if rank_for_each_true[i] > k :
            ndcg = 0
        else :
            ndcg = 1 / np.log2(rank_for_each_true[i] + 2)
        ndcgs[i] = ndcg
        
    return ndcgs
        

def recall_k(
    y_true : np.ndarray, ## Batch of target items  B
    y_pred : np.ndarray, ## Batch of predicted items B X N
    k : int = 5
) -> np.ndarray : ## Batch of ndcg_k scores B X 1
    
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



def load_poison_ndcg_10(ctx, y_true : np.ndarray, y_pred : np.ndarray, y_prob, **kwargs):
    """
    Load ndcg@10 metric.
    """
    if ctx.cur_split == 'train':
        results = None
    else :
        poison_true = ctx['poison_' + ctx.cur_split + '_y_true']
        results = ndcg_k(poison_true, y_pred, k = 10).mean()
    
    return results


def load_poison_ndcg_20(ctx, y_true : np.ndarray, y_pred : np.ndarray, y_prob, **kwargs):
    """
    Load ndcg@20 metric.
    """
    
    if ctx.cur_split == 'train':
        results = None
    else :
        poison_true = ctx['poison_' + ctx.cur_split + '_y_true']
        results = ndcg_k(poison_true, y_pred, k = 20).mean()
    
    return results


def load_poison_recall_10(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Load recall@10 metric.
    """
    if ctx.cur_split == 'train':
        results = None
    else :
        poison_true = ctx['poison_' + ctx.cur_split + '_y_true']
        results = recall_k(poison_true, y_pred, k = 10).mean()
    
    return results


def load_poison_recall_20(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Load recall@20 metric.
    """
    if ctx.cur_split == 'train':
        results = None
    else :
        poison_true = ctx['poison_' + ctx.cur_split + '_y_true']
        results = recall_k(poison_true, y_pred, k = 20).mean()
    
    return results


def call_poison_recall_10(types):
    if 'poison_recall_10' in types:
        the_larger_the_better = True
        return 'poison_recall_10', load_poison_recall_10, the_larger_the_better
    
    
def call_poison_recall_20(types):
    if 'poison_recall_20' in types:
        the_larger_the_better = True
        return 'poison_recall_20', load_poison_recall_20, the_larger_the_better
    
    
def call_poison_ndcg_10(types):
    if 'poison_ndcg_10' in types:
        the_larger_the_better = True
        return 'poison_ndcg_10', load_poison_ndcg_10, the_larger_the_better
    
    
def call_poison_ndcg_20(types):
    if 'poison_ndcg_20' in types:
        the_larger_the_better = True
        return 'poison_ndcg_20', load_poison_ndcg_20, the_larger_the_better


register_metric('poison_recall_10', call_poison_recall_10)
register_metric('poison_recall_20', call_poison_recall_20)
register_metric('poison_ndcg_10', call_poison_ndcg_10)
register_metric('poison_ndcg_20', call_poison_ndcg_20)