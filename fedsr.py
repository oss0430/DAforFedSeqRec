import argparse
from federatedscope.core.data import BaseDataTranslator

from utils import load_sr_dataset, SRDataset

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='data')
argparser.add_argument('--dataset', type=str, default='ML-100K', choices=['Amazon_Beauty', 'Amazon_Sports', 'Amazon_Toys', 'ML-100K', 'Yelp'])

args = argparser.parse_args()

## 1. Load Data
def load_data(
        config,
        args : argparse.Namespace,
        client_cfgs = None,
    ) :
    ## Load a dataset (torch.utils.data.Dataset) 
    dataset = load_sr_dataset(args.data_dir, args.dataset)