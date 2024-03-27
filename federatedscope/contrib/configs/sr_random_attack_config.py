from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_ml1m_random_attack(cfg) :
    
    cfg.attack = CN()
    cfg.attack.attack_method = ''
    cfg.attack.attacker_id = []
    cfg.attack.target_item_id = 0
    

register_config("ml-1m_sasrec_random_attack", extend_cfg_for_sasrec_ml1m_random_attack)