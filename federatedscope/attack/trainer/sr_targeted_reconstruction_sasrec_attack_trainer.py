import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

logger = logging.getLogger(__name__)


def wrap_SrTargetedReconstructionAttackSasrecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    
    ## Added Method for embedding_poisnous_train 
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
    
    """
    --- replace train hook set up ---
    _hook_on_fit_start_init -> _hook_on_fit_start_init_setup_reconstructed
    
    
    
    """
    
    return base_trainer



def _hook_on_fit_start_init_setup_reconstructed(ctx) :
    
    ctx.model.to(ctx.device)
    
    if ctx.cur_mode in [MODE.TRAIN, MODE.VAL]:
        ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
        ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)
    
    ## prepare datas
    ctx.reconstructed_data = 
    
    # prepare statistics
    ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
    ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
    ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
    ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
    ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
