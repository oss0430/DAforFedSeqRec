from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_attack(cfg) :
    
    cfg.attack = CN()
    cfg.attack.attack_method = ''
    cfg.attack.attacker_id = []
    cfg.attack.target_item_id = 0
    
    ## segment attack
    cfg.attack.segment_item_ids = []
    cfg.attack.covisitation = True
    
    ## reconstruction attack
    cfg.attack.reconstruction_iter = 200
    cfg.attack.eval_reconstruction = True
    cfg.attack.reconstruction_data_size = 20
    cfg.attack.reconstruction_batch_size = 1

register_config("ml-1m_sasrec_attack", extend_cfg_for_sasrec_attack)