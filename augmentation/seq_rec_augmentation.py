import argparse
import os
import sys

import random
from typing import List

from federatedscope.core.auxiliaries.utils import setup_seed

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--augmentation_type', type=str, default='random_replacing', choices=['random_replacing', 'cutting'])
parser.add_argument('-p', '--replace_probability', type=float, default=0.1)
parser.add_argument('-d', '--cut_direction', type=str, default='left', choices=['right', 'left'])
parser.add_argument('-n', '--number_of_generation', type=int, default=60)
parser.add_argument('-s', '--seed', type=int, default=42)

args = parser.parse_args()

## configurations
train_dataframe_path = '../../../../data1/donghoon/FederatedScopeData/ml-1m/split/train.csv'
user_column = 'user_id:token'
item_column = 'item_id:token'
timestamp_column = 'timestamp:float'
max_item_ids = 3952
max_sequence_length = 200
augmentation_column = 'augmentation_idx:token'
augmentation_prob = args.replace_probability
number_of_generation = args.number_of_generation
augmentation_type = args.augmentation_type
save_path_dir = f'../../../../data1/donghoon/FederatedScopeData/ml-1m/random_augmented_{augmentation_prob}'
seed = args.seed
user_gpu = True
gpu_id = 0
setup_seed(seed)

class SequenceDataset(Dataset) :
    
    def __init__(
        self,
        dataframe : pd.DataFrame,
        user_column : str,
        item_column : str,
        timestamp_column : str,
    ) :
        super(SequenceDataset, self).__init__()
        self.dataframe = dataframe
        self.user_column = user_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column
        
        
    def __len__(self) :
        return len(self.dataframe[self.user_column].unique())
    
    
    def __getitem__(
        self,
        idx : int
    ) :
        user_id = self.dataframe[self.user_column].unique()[idx]
        user_df = self.dataframe[self.dataframe[self.user_column] == user_id]
        
        user_interaction = user_df[self.item_column].values
        full_sequence = user_interaction
        
        return {
            'user_id' : user_id,
            'full_sequence' : full_sequence
        }
            
        
        
def replace_sequence(
    item_sequence : torch.Tensor,
    max_item_ids : int,
    probablity : float
) :
    ## Replace the item_sequence with random items
    mask = torch.rand(item_sequence.size()) < probablity
    random_values = torch.randint(1, max_item_ids+1, item_sequence.size())
    
    return torch.where(mask, random_values, item_sequence)


def random_replacing(
    full_sequence_dataset : SequenceDataset,
    replace_prob : float = 0.1,
    number_of_generation : int = 60,
    max_item_ids : int = 3952
) :
    resulted_dataset = {
        'user_id' : [],
        'original' : [],
        'list_of_augmented' :[]
    }
    
    ## Generate a augmented training sets
    full_sequence_loader = DataLoader(full_sequence_dataset, batch_size=1, shuffle=False)
    for data_batch in tqdm(full_sequence_loader, desc='Generating augmented dataset') :
        user_id = data_batch['user_id']
        full_sequence = data_batch['full_sequence'].squeeze(0)
        
        resulted_dataset['original'].append(full_sequence)
        resulted_dataset['user_id'].append(user_id)
        
        replaced_sequences = []
        for i in range(number_of_generation) :
            replaced_sequence = replace_sequence(full_sequence, max_item_ids, replace_prob)
            replaced_sequences.append(replaced_sequence)
        
        resulted_dataset['list_of_augmented'].append(replaced_sequences)

    return resulted_dataset


def cut_sequence(
    item_sequence : torch.Tensor,
    number_of_generation : int,
    max_sequence_length : int,
    direction : str = 'right'
) :
    ## Cut the sequence
    sequence_length = len(item_sequence)
    if sequence_length >= max_sequence_length :
        cut_range = list(range(sequence_length - max_sequence_length, sequence_length))
    else :
        cut_range = list(range(sequence_length))
    
    cut_sequences = []
    
    i= 0
    if direction == 'right' :
        while i < number_of_generation :
            cut_end = cut_range.pop()
            augmented_seq = item_sequence[:cut_end]
            cut_sequences.append(augmented_seq)
            i = i + 1
            if cut_range == [] :
                break
    else :  
        while i < number_of_generation :
            cut_start = cut_range.pop(0)
            augmented_seq = item_sequence[cut_start:]
            cut_sequences.append(augmented_seq)
            i = i + 1
            if cut_range == [] :
                break

    return cut_sequences



def cutting_augmentation(
    full_sequence_dataset : SequenceDataset,
    number_of_generation : int = 60,
    max_item_ids : int = 3952,
    max_sequence_length : int = 200
):
    resulted_dataset = {
        'user_id' : [],
        'original' : [],
        'list_of_augmented' :[]
    }
    
    ## Generate a augmented training sets
    full_sequence_loader = DataLoader(full_sequence_dataset, batch_size=1, shuffle=False)
    for data_batch in tqdm(full_sequence_loader, desc='Generating augmented dataset') :
        user_id = data_batch['user_id']
        full_sequence = data_batch['full_sequence'].squeeze(0)
        
        resulted_dataset['original'].append(full_sequence)
        resulted_dataset['user_id'].append(user_id)
        
        cut_sequences = cut_sequence(full_sequence,
                                     number_of_generation,
                                     max_sequence_length)
        
        resulted_dataset['list_of_augmented'].append(cut_sequences)

    return resulted_dataset


def sequence_to_tokens(
    user_id : int,
    full_sequence : List[int]
) :   
    full_sequence_np_array = np.array(full_sequence)
    ## reshape vertiaclly
    full_sequence_np_array = full_sequence_np_array.reshape(-1, 1)
    # concat user id and timestamp
    # timestamp's are incremented by 1
    user_id_np_array = np.full((full_sequence_np_array.shape[0], 1), user_id)
    timestamp_np_array = np.arange(1, full_sequence_np_array.shape[0] + 1).reshape(-1, 1)
    
    ## Concatenate the user_id, item_id, and timestamp
    resulted_np_array = np.concatenate([user_id_np_array, full_sequence_np_array, timestamp_np_array], axis=1)
    
    return resulted_np_array



def turn_result_into_dataframe(
    resulted_dataset : dict,
    user_column : str,
    item_column : str,
    timestamp_column : str,
    augmentation_column : str = "augmentation_idx:token"
) :
    ## Turn the resulted dataset into a dataframe
    user_id_per_tokenized_sequences = []
    
    for i in tqdm(range(len(resulted_dataset['user_id'])), desc='Converting to dataframe') :
        user_id = resulted_dataset['user_id'][i]
        original_sequence = resulted_dataset['original'][i]
        list_of_augmented = resulted_dataset['list_of_augmented'][i]
        
        ## initilize with original sequence
        tokenized_sequences = sequence_to_tokens(user_id, original_sequence)
        ## add column of augmentation index
        current_augmentation_index = 0
        tokenized_sequences = np.concatenate([tokenized_sequences, np.full((tokenized_sequences.shape[0], 1), current_augmentation_index)], axis=1)
        
        for current_augmented_sequence in list_of_augmented :
            current_augmentation_index = current_augmentation_index + 1
            tokenized_augmented_sequence = sequence_to_tokens(user_id, current_augmented_sequence)
            ## add column of augmentation index
            tokenized_augmented_sequence = np.concatenate([tokenized_augmented_sequence, np.full((tokenized_augmented_sequence.shape[0], 1), current_augmentation_index)], axis=1)
            ##concatenate to original vertically
            tokenized_sequences = np.concatenate([tokenized_sequences, tokenized_augmented_sequence], axis=0)
        
        user_id_per_tokenized_sequences.append(tokenized_sequences)
    
    ## Concatenate all the tokenized sequences
    user_id_per_tokenized_sequences = np.concatenate(user_id_per_tokenized_sequences, axis=0)
    
    ## Add columns to the dataframe
    return pd.DataFrame(user_id_per_tokenized_sequences, columns=[user_column, item_column, timestamp_column, augmentation_column])


def __main__():
    train_dataframe = pd.read_csv(train_dataframe_path)
    full_sequence_dataset = SequenceDataset(train_dataframe, user_column, item_column, timestamp_column)
    
    if augmentation_type == 'random_replacing' :
        resulted_dataset = random_replacing(full_sequence_dataset, replace_prob=augmentation_prob, number_of_generation=number_of_generation, max_item_ids=max_item_ids)
    elif augmentation_type == 'cutting' :
        resulted_dataset = cutting_augmentation(full_sequence_dataset, number_of_generation=number_of_generation, max_item_ids=max_item_ids, max_sequence_length=max_sequence_length)
    else :
        raise ValueError('Invalid augmentation type')
    
    resulted_dataframe = turn_result_into_dataframe(resulted_dataset, user_column, item_column, timestamp_column, augmentation_column)
    print("saving to : ", save_path_dir)
    os.makedirs(save_path_dir, exist_ok=True)
    save_path = os.path.join(save_path_dir, f'train.csv')
    resulted_dataframe.to_csv(save_path, index=False)

if __name__ == '__main__':
    __main__()
