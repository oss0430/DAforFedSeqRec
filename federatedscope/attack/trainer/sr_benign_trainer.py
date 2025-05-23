import logging
from typing import Type
import numpy as np

from federatedscope.core.trainers import GeneralTorchTrainer

logger = logging.getLogger(__name__)


def wrap_benignSrTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    Warp the benign trainer for backdoor attack:
    We just add the normalization operation.
    Args:
        base_trainer: Type: core.trainers.GeneralTorchTrainer
    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer
    '''
    
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
        
    # ---- action-level plug-in -------
    base_trainer.register_hook_in_eval(new_hook=hook_on_fit_end_test_attack_item_exposure_ratio,
                                       trigger='on_fit_end',
                                       insert_pos=0)
    ## this is placed right before original tirgger on_fit_end
    
    return base_trainer


def hook_on_fit_end_test_attack_item_exposure_ratio(ctx) :
    """
    Evaluate metrics of sr poisoning attacks.
    only need to register attack target item id
    
    """
    dataset_size = len(ctx.ys_true)
    empty_dataset = np.zeros((dataset_size), dtype=int)
    setattr(ctx, "poison_{}_y_true".format(ctx.cur_split),
            np.full_like((empty_dataset),
                         ctx.attack_target_item_id))
    
