from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cfg_for_sasrec_ml1m(cfg) :
    
    cfg.trainer = CN()    
    cfg.trainer.type = 'sasrec'

    cfg.model = CN()
    cfg.model.type = 'sasrec'
    cfg.model.user_column = 'user_id:token'
    cfg.model.item_column = 'item_id:token'
    cfg.model.max_sequence_length = 200
    cfg.model.item_num = 3706
    cfg.model.n_layers = 2
    cfg.model.n_heads = 1
    cfg.model.hidden_size = 50
    cfg.model.inner_size = 100
    cfg.model.hidden_dropout_prob = 0.0
    cfg.model.attn_dropout_prob = 0.1
    cfg.model.hidden_act = 'relu'
    cfg.model.layer_norm_eps = 1e-12
    cfg.model.initializer_range = 0.02
    cfg.model.use_position = True
    cfg.model.device = 'cuda:0'
    
    cfg.data = CN()
    cfg.data.type = 'sr_data'
    cfg.data.df_path = 'data/ml-1m/ratings.csv'
    cfg.data.user_column = 'user_id:token'
    cfg.data.item_column = 'item_id:token'
    cfg.data.interaction_column = 'rating:float'
    cfg.data.timestamp_column = 'timestamp:float'
    cfg.data.partitioned_df_path = 'data/ml-1m/split'
    cfg.data.save_partitioned_df_path = 'data/ml-1m/split'
    cfg.data.min_sequence_length = 5
    cfg.data.max_sequence_length = 200
    cfg.data.user_num = 6040
    cfg.data.item_num = 3706
    cfg.data.padding_value = 0
    ## --------- Added for outdate configs -----------------
    ##      refer to federatedscope/core/configs/cfg_data.py
    cfg.data.loader = ''
    cfg.data.batch_size = 64
    cfg.data.shuffle = True
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.data.walk_length = 2
    cfg.data.num_steps = 30
    cfg.data.sizes = [10, 5]
    ## -----------------------------------------------------
    
    cfg.splitter = CN()
    cfg.splitter.type = 'sr_splitter'
    cfg.splitter.client_num = 6040
    

register_config("ml-1m_sasrec", extend_cfg_for_sasrec_ml1m)