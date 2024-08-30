import argparse
import os
import sys

import random
from typing import List

from federatedscope.core.auxiliaries.utils import setup_seed
from typing import List, Dict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--origianl_dataset', type=str, default='ml-1m', choices=['ml-1m', 
                                                                                    'amazon_beauty',
                                                                                    'amazon_sports'])
parser.add_argument('-r', '--result_path', type=str, default='')
parser.add_argument('-c', '--aug_column_name', type=str, default='augmentation_idx:token')
parser.add_argument('-t', '--augmentation_type', type=str, default='random_replacing', choices=['random_replacing', 
                                                                                                'cutting', 
                                                                                                'shuffle', 
                                                                                                'random_pushing', 
                                                                                                'self_sampled_pushing'])
parser.add_argument('-p', '--replace_probability', type=float, default=0.1)
parser.add_argument('-d', '--direction', type=str, default='left', choices=['right',
                                                                            'left'])
parser.add_argument('-pls', '--push_length_range_start', type = int, default = 1)
parser.add_argument('-ple', '--push_length_range_end', type = int, default = 4)
parser.add_argument('-n', '--number_of_generation', type=int, default=60)
parser.add_argument('-s', '--seed', type=int, default=42)

args = parser.parse_args()

## ML-1m configurations
ml_1m_configs = {
    "train_dataframe_path" : '../../../../data1/donghoon/FederatedScopeData/ml-1m/split/train.csv',
    "result_branch_path" : '../../../../data1/donghoon/FederatedScopeData/ml-1m/',
    "user_column" : 'user_id:token',
    "item_column" : 'item_id:token',
    "timestamp_column" : 'timestamp:float',
    "max_item_ids" : 3952,
    "max_sequence_length" : 200,
    "min_sequence_length" : 3
}

## Amazon_Beauty configurations
amazon_beauty_configs = {
    "train_dataframe_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Beauty_5core_mapped/split/train.csv',
    "result_branch_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Beauty_5core_mapped/',
    "user_column" : 'user_id:token',
    "item_column" : 'item_id:token',
    "timestamp_column" : 'timestamp:float',
    "max_item_ids" : 259204,
    "max_sequence_length" : 50,
    "min_sequence_length" : 3
}

## Amazon_Sports configurations
amazon_sports_configs = {
    "train_dataframe_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Sports_and_Outdoors_5core_mapped/split/train.csv',
    "result_branch_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Sports_and_Outdoors_5core_mapped/',
    "user_column" : 'user_id:token',
    "item_column" : 'item_id:token',
    "timestamp_column" : 'timestamp:float',
    "max_item_ids" : 532197,
    "max_sequence_length" : 50,
    "min_sequence_length" : 3
}


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
            


class AugmentationGenerator :
    
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
    ) :
        self.full_sequence_dataset = full_sequence_dataset
        self.number_of_generation = number_of_generation
        self.max_item_ids = max_item_ids
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
    
    
    def generate_for_sequence(self, sequence : torch.Tensor) :
        NotImplementedError('This method should be implemented in the subclass')

    
    def generate(self) :
        resulted_dataset = {
            'user_id' : [],
            'original' : [],
            'list_of_augmented' :[]
        }
        full_sequence_loader = DataLoader(self.full_sequence_dataset, batch_size=1, shuffle=False)
        ## Generate a augmented training sets
        for data_batch in tqdm(full_sequence_loader, desc='Generating augmented dataset') :
            user_id = data_batch['user_id']
            full_sequence = data_batch['full_sequence'].squeeze(0)

            resulted_dataset['original'].append(full_sequence)
            resulted_dataset['user_id'].append(user_id)

            augmented_sequences = self.generate_for_sequence(full_sequence)
            resulted_dataset['list_of_augmented'].append(augmented_sequences)
        
        return resulted_dataset

    
class RandomReplacingAugmentation(AugmentationGenerator) :
        
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
        replace_prob : float
    ) :
        self.replace_prob = replace_prob
        super(RandomReplacingAugmentation, self).__init__(full_sequence_dataset, number_of_generation, max_item_ids, max_sequence_length, min_sequence_length)
    
    
    def replace_sequence(
        self,
        item_sequence : torch.Tensor,
        max_item_ids : int,
        probablity : float
    ) -> torch.Tensor:
        ## Replace the item_sequence with random items
        mask = torch.rand(item_sequence.size()) < probablity
        random_values = torch.randint(1, max_item_ids+1, item_sequence.size())

        return torch.where(mask, random_values, item_sequence)
    
    
    def generate_for_sequence(self, sequence) :
        replaced_sequences = []
        for i in range(self.number_of_generation) :
            replaced_sequence = self.replace_sequence(sequence, self.max_item_ids, self.replace_prob)
            replaced_sequences.append(replaced_sequence)
        
        return replaced_sequences
    
        
class CuttingAugmentation(AugmentationGenerator) :
        
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
        direction : str
    ) :
        self.direction = direction
        
        super(CuttingAugmentation, self).__init__(full_sequence_dataset, number_of_generation, max_item_ids, max_sequence_length, min_sequence_length)
    
    
    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        ## Cut the sequence
        sequence_length = len(item_sequence)
        if sequence_length >= self.max_sequence_length and self.direction == 'left' :
            cut_range = list(range(sequence_length - self.max_sequence_length, sequence_length))
        else :
            cut_range = list(range(sequence_length))

        cut_sequences = []

        i= 0
        if self.direction == 'right' :
            while i < self.number_of_generation :
                if len(cut_range) <= self.min_sequence_length :
                    break
                cut_end = cut_range.pop()
                augmented_seq = item_sequence[:cut_end]
                cut_sequences.append(augmented_seq)
                i = i + 1
        else :  
            cut_range.pop(0) ## remove the first element
            while i < self.number_of_generation :
                if len(cut_range) < self.min_sequence_length :
                    break
                cut_start = cut_range.pop(0)
                augmented_seq = item_sequence[cut_start:]
                cut_sequences.append(augmented_seq)
                i = i + 1

        return cut_sequences


class ShuffleAugmentation(AugmentationGenerator) :


    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        ## Shuffle the sequence
        shuffled_sequences = []

        for i in range(self.number_of_generation) :
            shuffled_sequence = item_sequence[torch.randperm(item_sequence.size(0))]
            shuffled_sequences.append(shuffled_sequence)

        return shuffled_sequences


class RandomSequencePushing(AugmentationGenerator) :
        
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
        push_direction : str,
        push_length_range : List[int]
    ) :
        self.push_direction = push_direction
        self.push_length_range = push_length_range
        super(RandomSequencePushing, self).__init__(full_sequence_dataset, number_of_generation, max_item_ids, max_sequence_length, min_sequence_length)
    
    
    def generate_pushing_sequence(
        self,
        length : int,
        item_sequence : torch.Tensor
    ) -> torch.Tensor:
        return torch.randint(1, self.max_item_ids+1, (length,))
    
    
    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        ## Push the sequence
        sequence_length = len(item_sequence)
        if sequence_length >= self.max_sequence_length :
            return [] ## if the sequence is already at the maximum length, return empty list
        
        ## cut the push_length_range if neccessary
        minimum_result_length = sequence_length + self.push_length_range[0]
        maximum_result_length = sequence_length + self.push_length_range[1]
    
        if minimum_result_length > self.max_sequence_length :
            push_length_range_start = self.min_sequence_length - sequence_length
            push_length_range_end = self.min_sequence_length - sequence_length
        elif maximum_result_length > self.max_sequence_length :
            push_length_range_start = self.push_length_range[0]
            push_length_range_end = self.max_sequence_length - sequence_length
        else :
            push_length_range_start = self.push_length_range[0]
            push_length_range_end = self.push_length_range[1]

        push_sequences = []

        for i in range(self.number_of_generation) :
            push_length = random.randint(push_length_range_start, push_length_range_end + 1)
            pushing_sequence = self.generate_pushing_sequence(push_length, item_sequence)
            if self.push_direction == 'right' :
                push_sequence = torch.cat([item_sequence, pushing_sequence])
            else :
                push_sequence = torch.cat([pushing_sequence, item_sequence])

            push_sequences.append(push_sequence)

        return push_sequences


class SelfSampledSequencePushing(RandomSequencePushing) :
    
    def generate_pushing_sequence(
        self,
        length : int,
        item_sequence : torch.Tensor
    ) -> torch.Tensor:
        ## sample the item from the given sequence to make new sequence
       
        unique_items = item_sequence.unique()
        pushing_sequence_where = torch.randint(0, unique_items.size(0), (length,))
        pushing_sequence = unique_items[pushing_sequence_where]
        
        return pushing_sequence


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


def load_dataset_configs(
    dataset_name : str
): 
    if dataset_name == 'ml-1m' :
        return ml_1m_configs
    elif dataset_name == 'amazon_beauty' :
        return amazon_beauty_configs
    elif dataset_name == 'amazon_sports' :
        return amazon_sports_configs
    else :
        raise ValueError('Invalid dataset name')


def __main__():
    
    ## Set the configurations
    direction = args.direction
    augmentation_column = args.aug_column_name
    augmentation_prob = args.replace_probability
    number_of_generation = args.number_of_generation
    augmentation_type = args.augmentation_type
    push_length_range = [args.push_length_range_start, args.push_length_range_end]
    seed = args.seed
    user_gpu = True
    gpu_id = 0

    setup_seed(seed)
    
    dataset_configs = load_dataset_configs(args.origianl_dataset)
    
    train_dataframe_path = dataset_configs['train_dataframe_path']
    user_column = dataset_configs['user_column']
    item_column = dataset_configs['item_column']
    timestamp_column = dataset_configs['timestamp_column']
    max_item_ids = dataset_configs['max_item_ids']
    max_sequence_length = dataset_configs['max_sequence_length']
    min_sequence_length = dataset_configs['min_sequence_length']
    result_branch_path = dataset_configs['result_branch_path']
    
    if args.result_path != '' :
        save_path_dir = args.result_path
    else :
        leaf_folder = f"{augmentation_type}"
        if augmentation_type == 'random_replacing' :
            leaf_folder = f"{leaf_folder}_{augmentation_prob}"
        elif augmentation_type == 'cutting' :
            leaf_folder = f"{leaf_folder}_{direction}"
        elif augmentation_type == 'random_pushing' or augmentation_type == 'self_sampled_pushing' :
            leaf_folder = f"{leaf_folder}_{direction}_{push_length_range[0]}_{push_length_range[1]}"
        save_path_dir = result_branch_path + leaf_folder
    
    train_dataframe = pd.read_csv(train_dataframe_path)
    full_sequence_dataset = SequenceDataset(train_dataframe, user_column, item_column, timestamp_column)
    
    print("Augmentation type : ", augmentation_type)
    if augmentation_type == 'random_replacing' :
        print("Replace probability : ", augmentation_prob)
    elif augmentation_type == 'cutting' :
        print("Cut direction : ", direction)
        print("Min sequence length : ", min_sequence_length)
    elif augmentation_type == 'shuffle' :
        pass
    elif augmentation_type == 'random_pushing' :
        print("Push direction : ", direction)
        print("Push length range : ", push_length_range)
    elif augmentation_type == 'self_sampled_pushing' :
        print("Push direction : ", direction)
        print("Push length range : ", push_length_range)
    
    if augmentation_type == 'random_replacing' :
        augmentation_generator = RandomReplacingAugmentation(full_sequence_dataset,
                                                             number_of_generation,
                                                             max_item_ids,
                                                             max_sequence_length,
                                                             min_sequence_length,
                                                             augmentation_prob)
    elif augmentation_type == 'cutting' :
        augmentation_generator = CuttingAugmentation(full_sequence_dataset,
                                                     number_of_generation,
                                                     max_item_ids,
                                                     max_sequence_length,
                                                     min_sequence_length,
                                                     direction)
    elif augmentation_type == 'shuffle' :
        augmentation_generator = ShuffleAugmentation(full_sequence_dataset,
                                                     number_of_generation,
                                                     max_item_ids,
                                                     max_sequence_length,
                                                     min_sequence_length)
    elif augmentation_type == 'random_pushing' :
        augmentation_generator = RandomSequencePushing(full_sequence_dataset,
                                                       number_of_generation,
                                                       max_item_ids,
                                                       max_sequence_length,
                                                       min_sequence_length,
                                                       direction,
                                                       push_length_range)
    elif augmentation_type == 'self_sampled_pushing' :
        augmentation_generator = SelfSampledSequencePushing(full_sequence_dataset,
                                                            number_of_generation,
                                                            max_item_ids,
                                                            max_sequence_length,
                                                            min_sequence_length,
                                                            direction,
                                                            push_length_range)
    else :
        raise ValueError('Invalid augmentation type')
    
    resulted_dataset = augmentation_generator.generate()
    
    resulted_dataframe = turn_result_into_dataframe(resulted_dataset, user_column, item_column, timestamp_column, augmentation_column)
    print("saving to : ", save_path_dir)
    os.makedirs(save_path_dir, exist_ok=True)
    save_path = os.path.join(save_path_dir, f'train.csv')
    resulted_dataframe.to_csv(save_path, index=False)

if __name__ == '__main__':
    __main__()
