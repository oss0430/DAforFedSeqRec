import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar


logger = logging.getLogger(__name__)


def wrap_rand_SASRecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    ## SASRec Trainer inherits from GeneralTorchTrainer
    
    base_trainer.ctx.target_item_id = \
        base_trainer.cfg.attack.target_item_id
        
        
    # ---- action-level plug-in -------
    
    base_trainer.register_hook_in_train(new_hook = create_random_sequence_poison,
                                        trigger = 'on_batch_start',
                                        insert_mode = -1)
    base_trainer.register_hook_in_train(new_hook = hook_on_batch_forward_poison_data,
                                        trigger = 'on_batch_forward',
                                        insert_mode = -1)


def create_random_sequence_poison(ctx):
    """
    Create poisonous sequence data, according to model's max_seq_length and target_item_id
    """
    
    max_seq_length = ctx.model.max_seq_length
    n_items = ctx.model.n_items
    
    data_batch = ctx.data_batch
    item_seq = data_batch["item_seq"]
    item_seq_len = data_batch["item_seq_len"]
    target_item = data_batch["target_item"]
    
    random_sequence = torch.randint_like(ctx.item_seq,
                                         high = n_items,
                                         low = ctx.model.padding_idx + 1)
    ## padding are always the lowest number in embedding
    
    poison_target_item = torch.zeros_like(ctx.target_item)
    poison_target_item = poison_target_item + ctx.target_item_id
    
    ctx.poison_item_seq = random_sequence
    ctx.poison_item_seq_len = item_seq_len
    ctx.poison_target_item = poison_target_item
    

def hook_on_batch_forward_poison_data(ctx):

    item_seq = ctx.poison_item_seq 
    item_seq_len = ctx.poison_item_seq_len 
    target_item = ctx.poison_target_item 
    
    item_seq, item_seq_len, target_item = item_seq.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device)
    
    outputs = ctx.model(item_seq, item_seq_len)
    pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
    
    target_embedding = ctx.model.item_embedding(target_item).squeeze(1)
    
    ## hijack original loss with poison data loss
    ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
    ctx.y_pred = CtxVar(pred, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob
    
    ctx.loss_batch = ctx.criterion(outputs, target_embedding)
    ctx.batch_size = len(target_item)
    
    
    
    