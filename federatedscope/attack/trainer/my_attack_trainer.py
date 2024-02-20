import numpy as np
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property

logger = logging.getLogger(__name__)


def wrap_SASRecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    ## SASRec Trainer inherits from GeneralTorchTrainer
    
    # ---------------- attribute-level plug-in -----------------------
    
    base_trainer.ctx.target_item_id = \
        base_trainer.cfg.attack.target_item_id

    base_trainer.ctx.trigger_item_ids =\
        base_trainer.cfg.attack.trigger_item_ids
    


def hook_on_batch_forward_fake_data(ctx):
    """
    generate fake data and calculate loss with it
    Args : 
        ctx : Context
    
    Returns:    
    
    """
    
    
    def generate_fake_data(
        trigger_set : np.ndarray,
        item_num : int,
        batch_size : int,
        max_sequence_length : int
    ) -> np.ndarray:
        
        trigger_insert_index = np.random.randint(low = 0,
                                                 high = max_sequence_length,
                                                 size = (batch_size, 1))
        
        
        trigger_combination = np.random.choice(trigger_set, size=(batch_size, 1))
        
        
        fake_interaction = np.zeros((batch_size, max_sequence_length))
        
        return 
    
    
    
    
    