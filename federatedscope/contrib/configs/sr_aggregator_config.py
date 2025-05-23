from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_aggregator(cfg) :
    
    cfg.aggregator = CN()
    cfg.aggregator.robust_rule = 'fedavg'
    cfg.aggregator.byzantine_node_num = 0
    cfg.aggregator.BFT_args = CN(new_allowed=True)
    
#register_config("sasrec_testing", extend_cfg_for_sasrec_testing)