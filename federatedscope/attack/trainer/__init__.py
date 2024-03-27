from federatedscope.attack.trainer.GAN_trainer import *
from federatedscope.attack.trainer.MIA_invert_gradient_trainer import *
from federatedscope.attack.trainer.PIA_trainer import *
from federatedscope.attack.trainer.backdoor_trainer import *
from federatedscope.attack.trainer.benign_trainer import *
from federatedscope.attack.trainer.gaussian_attack_trainer import *
from federatedscope.attack.trainer.sr_targeted_random_sasrec_attack_trainer import *
from federatedscope.attack.trainer.sr_benign_trainer import *
from federatedscope.attack.trainer.sr_targeted_segment_sasrec_attack_trainer import *
from federatedscope.attack.trainer.sr_targeted_labelFlip_sasrec_attack_trainer import *
from federatedscope.attack.trainer.sr_targeted_random_sasrec_attack_w_smart_label_trainer import *
#from federatedscope.attack.trainer.sr_targeted_coordinated_attack_trainer import *


__all__ = [
    'wrap_GANTrainer', 'hood_on_fit_start_generator',
    'hook_on_batch_forward_injected_data',
    'hook_on_batch_injected_data_generation', 'hook_on_gan_cra_train',
    'hook_on_data_injection_sav_data', 'wrap_GradientAscentTrainer',
    'hook_on_fit_start_count_round', 'hook_on_batch_start_replace_data_batch',
    'hook_on_batch_backward_invert_gradient',
    'hook_on_fit_start_loss_on_target_data', 'wrap_backdoorTrainer',
    'wrap_benignTrainer', 'wrap_GaussianAttackTrainer',
    
    ## benign Sr
    'wrap_benignSrTrainer',
    
    ## Targeted Random SASRec Attack
    'wrap_SrTargetedRandomAttackSasrecTrainer', 'register_random_sequence_poison',
    'hook_on_batch_forward_poison_data',
    
    ## Targeted Segment SASRec Attack
    'wrap_SrTargetedSegmentAttackSasrecTrainer',
    
    ## Targeted LabelFlip SASRec Attack
    'wrap_SrTargetedLabelFlipAttackSasrecTrainer',
    
    ## Random Smart Label SASRec Attack
    'wrap_SrTargetedSmartRandomAttackSasrecTrainer'
    
]
