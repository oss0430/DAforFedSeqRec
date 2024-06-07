import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar



logger = logging.getLogger(__name__)


def wrap_SrTargetedReconstructionAttackSasrecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    
    ## Added Method for embedding_poisnous_train 
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
    
    ## --- new hook setup ---
    
    return base_trainer

