import torch

from federatedscope.core.configs.yacs_config import CfgNode

class SRDataset(torch.utils.data.Dataset):
    
    

def set_global_cfg(
    dataset_name : str    
) -> CfgNode :
    global_cfg


def load_sr_dataset(
    dir_path : str,
    dataset_name : str,
) -> SRDataset :
    