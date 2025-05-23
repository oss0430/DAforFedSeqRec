import numpy as np
import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar

logger = logging.getLogger(__name__)


def wrap_SrTargetedLabelFlipAttackSasrecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    ## SASRec Trainer inherits from GeneralTorchTrainer
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
        
    # ---- action-level plug-in -------
    base_trainer.replace_hook_in_train(
        new_hook = hook_on_batch_forward_poison_label,
        target_trigger= 'on_batch_forward',
        target_hook_name = '_hook_on_batch_forward')
    
    return base_trainer


def hook_on_batch_forward_poison_label(ctx):
    
    data_batch = ctx.data_batch
    item_seq = data_batch["item_seq"]
    item_seq_len = data_batch["item_seq_len"]
    target_item = data_batch["target_item"]
    attack_target_item_id = ctx.attack_target_item_id
    
    poison_target_item = torch.zeros_like(target_item)
    poison_target_item = poison_target_item + attack_target_item_id
    
    item_seq, item_seq_len, poison_target_item = item_seq.to(ctx.device), item_seq_len.to(ctx.device), poison_target_item.to(ctx.device)
    
    ctx.poison_target_item = poison_target_item
    
    outputs = ctx.model(item_seq, item_seq_len)
    pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
    
    test_item_emb = ctx.model.item_embedding.weight
    logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
    
    ## hijack original loss with poison data loss
    ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
    ctx.y_pred = CtxVar(pred, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob
    
    ctx.loss_batch = ctx.criterion(logits, poison_target_item)
    ctx.batch_size = len(poison_target_item)