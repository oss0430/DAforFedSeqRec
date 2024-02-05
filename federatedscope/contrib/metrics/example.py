from federatedscope.register import register_metric
import numpy as np

METRIC_NAME = 'example'


def MyMetric(ctx, **kwargs):
    return ctx.num_train_data


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = MyMetric
        return METRIC_NAME, metric_builder, the_larger_the_better

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


def load_ndcg_10(ctx , y_true : np.ndarray, y_pred : np.ndarray, y_prob, **kwargs):
    """
    Load ndcg@10 metric.
    """
    results = ndcg_k(y_true, y_pred, k = 10).mean()
    
    return results


def load_ndcg_20(ctx , y_true : np.ndarray, y_pred : np.ndarray, y_prob, **kwargs):
    """
    Load ndcg@20 metric.
    """
    results = ndcg_k(y_true, y_pred, k = 20).mean()
    
    return results


def load_recall_10(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Load recall@10 metric.
    """
    results = recall_k(y_true, y_pred, k = 10).mean()
    
    return results


def load_recall_20(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Load recall@20 metric.
    """
    
    results = recall_k(y_true, y_pred, k = 20).mean()
    
    return results


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



register_metric(METRIC_NAME, call_my_metric)
