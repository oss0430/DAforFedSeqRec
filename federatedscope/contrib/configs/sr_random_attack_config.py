from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_ml1m_random_attack(cfg) :
    
    cfg.attack = CN()
    cfg.attack.attack_method = 'sr_targeted_random_sasrec'
    cfg.attack.attacker_id = [1,2,3,4,5,6,7,8,9,10]
    cfg.attack.target_item_id = 3000
    

register_config("ml-1m_sasrec_random_attack", extend_cfg_for_sasrec_ml1m_random_attack)