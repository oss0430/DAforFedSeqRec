from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_testing(cfg) :
    
    cfg.srtest = CN()
    cfg.srtest.itemdrop_method = "" ## 'first', 'intermediate', 'last'
    cfg.srtest.offset = 0
    cfg.srtest.dropcount = 0
    cfg.srtest.dropping_user_id = [] ## List[int]

#register_config("sasrec_testing", extend_cfg_for_sasrec_testing)