from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_ml1m_segment_attack(cfg) :
    
    cfg.attack = CN()
    cfg.attack.attack_method = 'sr_targeted_segment_sasrec'
    cfg.attack.attacker_id = [1,2,3,4,5,6,7,8,9,10]
    cfg.attack.target_item_id = 3000
    cfg.attack.segment_item_ids = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009]
    cfg.attack.covisitation = True
    
register_config("ml-1m_sasrec_segment_attack", extend_cfg_for_sasrec_ml1m_segment_attack)