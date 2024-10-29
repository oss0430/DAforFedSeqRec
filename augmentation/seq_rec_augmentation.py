import argparse
import os
import sys

import random
from itertools import combinations

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
                                                                                                'self_sampled_pushing',
                                                                                                'random_masking',
                                                                                                'cut_middle',
                                                                                                'random_deletion'])
parser.add_argument('-p', '--replace_probability', type=float, default=0.1)
parser.add_argument('-d', '--direction', type=str, default='left', choices=['right',
                                                                            'left'])
parser.add_argument('-ls', '--length_range_start', type = int, default = 1) # both for cut_middle length min
parser.add_argument('-le', '--length_range_end', type = int, default = 4) # both for cut_middle length max
parser.add_argument('-is', '--item_perturb_range_start', type=int, default=1) # for random_deletion
parser.add_argument('-ie', '--item_perturb_range_end', type=int, default=1) # for random_deletion
parser.add_argument('-mi', '--mask_token_id', type=int, default=0) # only for random_masking
parser.add_argument('-n', '--number_of_generation', type=int, default=60)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-f', '--fix_length', type=bool, default=True) # for random deletion and cut_middle.

parser.add_argument('-no_org', '--no_original', type = bool, default = True)

args = parser.parse_args()

# ML-1m configurations
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

# Amazon_Beauty configurations
amazon_beauty_configs = {
    "train_dataframe_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Beauty_5core/split/train.csv',
    "result_branch_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Beauty_5core/',
    "user_column" : 'user_id:token',
    "item_column" : 'item_id:token',
    "timestamp_column" : 'timestamp:float',
    "max_item_ids" : 12101,
    "max_sequence_length" : 50,
    "min_sequence_length" : 3
}

# Amazon_Sports configurations
amazon_sports_configs = {
    "train_dataframe_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Sports_and_Outdoors_5core/split/train.csv',
    "result_branch_path" : '../../../../data1/donghoon/FederatedScopeData/Amazon_Sports_and_Outdoors_5core/',
    "user_column" : 'user_id:token',
    "item_column" : 'item_id:token',
    "timestamp_column" : 'timestamp:float',
    "max_item_ids" : 18357,
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
        # Generate a augmented training sets
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
        # Replace the item_sequence with random items
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
        
        super(CuttingAugmentation, self).__init__(full_sequence_dataset, 
                                                  number_of_generation, 
                                                  max_item_ids, 
                                                  max_sequence_length, 
                                                  min_sequence_length)
    
    
    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        # Cut the sequence
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
            cut_range.pop(0) # remove the first element
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
        # Shuffle the sequence
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
        # Push the sequence
        sequence_length = len(item_sequence)
        if sequence_length >= self.max_sequence_length :
            return [] # if the sequence is already at the maximum length, return empty list
        
        # cut the push_length_range if neccessary
        minimum_result_length = sequence_length + self.push_length_range[0]
        maximum_result_length = sequence_length + self.push_length_range[1]
    
        if minimum_result_length > self.max_sequence_length :
            push_length_range_start = self.max_sequence_length - sequence_length
            push_length_range_end = self.max_sequence_length - sequence_length
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
        # sample the item from the given sequence to make new sequence
       
        unique_items = item_sequence.unique()
        pushing_sequence_where = torch.randint(0, unique_items.size(0), (length,))
        pushing_sequence = unique_items[pushing_sequence_where]
        
        return pushing_sequence


class RandomMasking(AugmentationGenerator):
    
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
        min_mask_count : int,
        max_mask_count : int,
        mask_token_id : int = 0
    ) :
        self.min_mask_count = min_mask_count
        self.max_mask_count = max_mask_count
        self.mask_token_id = mask_token_id
        super(RandomMasking, self).__init__(full_sequence_dataset, number_of_generation, max_item_ids, max_sequence_length, min_sequence_length)
    
    def _short_sequence_deletion(self, item_sequence, sequence_length, max_mask_count):
        possible_combinations = []
        sequence_start_idx = 0
        if sequence_length > self.max_sequence_length :
            sequence_start_idx = sequence_length - self.max_sequence_length
        
        for i in range(self.min_mask_count, max_mask_count + 1) :
            possible_combinations = possible_combinations + list(combinations(range(sequence_start_idx, sequence_length), i))
        
        random.shuffle(possible_combinations)
        masked_sequences = []
        list_item_sequence = item_sequence.tolist()
        for combo in possible_combinations :    
            if len(masked_sequences) >= self.number_of_generation :
                break
            mask_sequence = []
            for idx, item in enumerate(list_item_sequence):
                if idx not in combo :
                    mask_sequence.append(item)
                else :
                    mask_sequence.append(self.mask_token_id)
            mask_sequence = torch.tensor(mask_sequence)
            masked_sequences.append(mask_sequence)
            
        return masked_sequences
    
    
    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        # Shuffle the sequence
        # Randomly Delete N items
        # first check if the sequence is already at the minimum length
        # and maximum deleteion count
        sequence_length = len(item_sequence)
        max_item_at_disposal = max(sequence_length - self.min_sequence_length, 0)
        if max_item_at_disposal < self.min_mask_count :
            return []
        max_mask_count = min(max_item_at_disposal, self.max_mask_count)
        
        sequence_start_idx = 0
        if sequence_length > self.max_sequence_length :
            sequence_start_idx = sequence_length - self.max_sequence_length
        
        seen_combinations = set()
        
        if sequence_length - self.min_mask_count < 10:
            return self._short_sequence_deletion(item_sequence, sequence_length, max_mask_count)
        
        masked_sequences = []
        max_continue = 20
        continue_count = 0
        list_item_sequence = item_sequence.tolist()
        while len(masked_sequences) < self.number_of_generation :
            # Randomly Select a number of items to delete within the range
            mask_count = random.randint(self.min_mask_count, max_mask_count)
            masking_indices = sorted(random.sample(range(sequence_start_idx, sequence_length), mask_count))
            
            if tuple(masking_indices) in seen_combinations :
                if continue_count > max_continue :
                    break
                continue_count = continue_count + 1
                continue
            else : 
                continue_count = 0
            seen_combinations.add(tuple(masking_indices))
            masked_seuqence = [self.mask_token_id if idx in masking_indices else item \
                                for idx, item in enumerate(list_item_sequence)]
            masked_seuqence = torch.tensor(masked_seuqence)        
            masked_sequences.append(masked_seuqence)
            
        return masked_sequences
    

class CutMiddleAugmentation(AugmentationGenerator):
    
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
        cut_count_min : int,
        cut_count_max : int,
        fix_length : bool
    ) :
        self.cut_count_min = cut_count_min
        self.cut_count_max = cut_count_max
        super(CutMiddleAugmentation, self).__init__(full_sequence_dataset, number_of_generation, max_item_ids, max_sequence_length, min_sequence_length)
    
    
    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        
        # check for appropriate cut range
        sequence_length = len(item_sequence)
        max_item_at_disposal = max(sequence_length - self.min_sequence_length, 0)
        if max_item_at_disposal < self.cut_count_min :
            return []
        
        cut_count_max = min(max_item_at_disposal, self.cut_count_max)
        
        seq_start = 0
        if sequence_length > self.max_sequence_length :
            seq_start = sequence_length - self.max_sequence_length
        
        possible_combinations = []
        for cut_length in range(self.cut_count_min, cut_count_max + 1) :
            possible_combinations = possible_combinations + [(start_idx, cut_length) for start_idx in range(seq_start, sequence_length - cut_length + 1)]
        
        random.shuffle(possible_combinations)
        cut_sequences = []
        for (cut_start, current_cut_length) in possible_combinations :
            if len(cut_sequences) >= self.number_of_generation :
                break
            cut_end = cut_start + current_cut_length
            cut_sequence = torch.cat([item_sequence[:cut_start], item_sequence[cut_end:]])
            cut_sequences.append(cut_sequence)
        
        return cut_sequences


class RandomDeletion(AugmentationGenerator):
    
    def __init__(
        self,
        full_sequence_dataset : SequenceDataset,
        number_of_generation : int,
        max_item_ids : int,
        max_sequence_length : int,
        min_sequence_length : int,
        delete_count_min : int,
        delete_count_max : int,
        fix_length : bool
    ) :
        self.delete_count_min = delete_count_min
        self.delete_count_max = delete_count_max
        super(RandomDeletion, self).__init__(full_sequence_dataset, number_of_generation, max_item_ids, max_sequence_length, min_sequence_length)

    
    def _short_sequence_deletion(self, item_sequence, sequence_length, delete_count_max):
        possible_combinations = []
        sequence_start_idx = 0
        if sequence_length > self.max_sequence_length :
            sequence_start_idx = sequence_length - self.max_sequence_length
        
        for i in range(self.delete_count_min, delete_count_max + 1) :
            possible_combinations = possible_combinations + list(combinations(range(sequence_start_idx, sequence_length), i))
        
        random.shuffle(possible_combinations)
        deleted_sequences = []
        list_item_sequence = item_sequence.tolist()
        for combo in possible_combinations :    
            if len(deleted_sequences) >= self.number_of_generation :
                break
            deleted_sequence = []
            for idx, item in enumerate(list_item_sequence):
                if idx not in combo :
                    deleted_sequence.append(item)
            deleted_sequence = torch.tensor(deleted_sequence)
            deleted_sequences.append(deleted_sequence)
            
        return deleted_sequences
    
    
    def generate_for_sequence(
        self,
        item_sequence : torch.Tensor
    ) -> List[torch.Tensor]:
        # Shuffle the sequence
        # Randomly Delete N items
        # first check if the sequence is already at the minimum length
        # and maximum deleteion count
        sequence_length = len(item_sequence)
        max_item_at_disposal = max(sequence_length - self.min_sequence_length, 0)
        if max_item_at_disposal < self.delete_count_min :
            return []
        delete_count_max = min(max_item_at_disposal, self.delete_count_max)
        
        sequence_start_idx = 0
        if sequence_length > self.max_sequence_length :
            sequence_start_idx = sequence_length - self.max_sequence_length
        
        seen_combinations = set()
        
        if sequence_length - self.delete_count_min < 10:
            return self._short_sequence_deletion(item_sequence, sequence_length, delete_count_max)
        
        deleted_sequences = []
        max_continue = 20
        continue_count = 0
        while len(deleted_sequences) < self.number_of_generation :
            # Randomly Select a number of items to delete within the range
            delete_count = random.randint(self.delete_count_min, delete_count_max)
            delete_indices = sorted(random.sample(range(sequence_start_idx, sequence_length), delete_count))
            
            if tuple(delete_indices) in seen_combinations :
                if continue_count > max_continue :
                    break
                continue_count = continue_count + 1
                continue
            else : 
                continue_count = 0
            seen_combinations.add(tuple(delete_indices))
            deleted_sequence = [item for idx, item in enumerate(item_sequence) if idx not in delete_indices]
            deleted_sequence = torch.tensor(deleted_sequence)        
            deleted_sequences.append(deleted_sequence)
            
        return deleted_sequences


def sequence_to_tokens(
    user_id : int,
    full_sequence : List[int]
) :   
    full_sequence_np_array = np.array(full_sequence)
    # reshape vertiaclly
    full_sequence_np_array = full_sequence_np_array.reshape(-1, 1)
    # concat user id and timestamp
    # timestamp's are incremented by 1
    user_id_np_array = np.full((full_sequence_np_array.shape[0], 1), user_id)
    timestamp_np_array = np.arange(1, full_sequence_np_array.shape[0] + 1).reshape(-1, 1)
    
    # Concatenate the user_id, item_id, and timestamp
    resulted_np_array = np.concatenate([user_id_np_array, full_sequence_np_array, timestamp_np_array], axis=1)
    
    return resulted_np_array



def turn_result_into_dataframe(
    resulted_dataset : dict,
    user_column : str,
    item_column : str,
    timestamp_column : str,
    augmentation_column : str = "augmentation_idx:token",
    remove_original : bool = True
) :
    # Turn the resulted dataset into a dataframe
    user_id_per_tokenized_sequences = []
    
    for i in tqdm(range(len(resulted_dataset['user_id'])), desc='Converting to dataframe') :
        users_tokenized_sequences = []
        
        user_id = resulted_dataset['user_id'][i]
        original_sequence = resulted_dataset['original'][i]
        list_of_augmented = resulted_dataset['list_of_augmented'][i]
        
        # Remove Original sequence when augmented sequence exists
        can_remove_org = False
        if remove_original :
            if len(list_of_augmented) > 0 :
                can_remove_org = True
            else :
                can_remove_org = False
        if can_remove_org :
            current_augmentation_index = -1
        else :
            # initilize with original sequence
            original_sequence_tokenized = sequence_to_tokens(user_id, original_sequence)
            current_augmentation_index = 0        
            original_sequence_tokenized = np.concatenate([original_sequence_tokenized, np.full((original_sequence_tokenized.shape[0], 1), current_augmentation_index)], axis=1)
            users_tokenized_sequences.append(original_sequence_tokenized)
        
        # column adding to augmented sequences tokenized
        for current_augmented_sequence in list_of_augmented :
            current_augmentation_index = current_augmentation_index + 1
            tokenized_augmented_sequence = sequence_to_tokens(user_id, current_augmented_sequence)
            # add column of augmentation index
            tokenized_augmented_sequence = np.concatenate([tokenized_augmented_sequence, np.full((tokenized_augmented_sequence.shape[0], 1), current_augmentation_index)], axis=1)
            users_tokenized_sequences.append(tokenized_augmented_sequence)
            #concatenate to original vertically
        
        tokenized_sequences = np.concatenate(users_tokenized_sequences, axis=0)
        user_id_per_tokenized_sequences.append(tokenized_sequences)
    
    # Concatenate all the tokenized sequences
    user_id_per_tokenized_sequences = np.concatenate(user_id_per_tokenized_sequences, axis=0)
    
    # Add columns to the dataframe
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
    
    # Set the configurations
    direction = args.direction
    augmentation_column = args.aug_column_name
    augmentation_prob = args.replace_probability
    number_of_generation = args.number_of_generation
    augmentation_type = args.augmentation_type
    length_range = [args.length_range_start, args.length_range_end]
    item_pertrub_range = [args.item_perturb_range_start, args.item_perturb_range_end]
    mask_token_id = args.mask_token_id
    fixed_length = args.fix_length
    
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
        elif augmentation_type == 'random_pushing' or augmentation_type == 'self_sampled_pushing':
            leaf_folder = f"{leaf_folder}_{length_range[0]}_{length_range[1]}"
        elif augmentation_type == 'cut_middle' :
            leaf_folder = f"{leaf_folder}_{length_range[0]}_{length_range[1]}"
            if fixed_length :
                leaf_folder = f"{leaf_folder}_fixed_length"
        elif  augmentation_type == 'random_masking' or augmentation_type == 'random_deletion':
            leaf_folder = f"{leaf_folder}_{item_pertrub_range[0]}_{item_pertrub_range[1]}"
            if fixed_length and augmentation_type == 'random_deletion':
                leaf_folder = f"{leaf_folder}_fixed_length"
        save_path_dir = result_branch_path + leaf_folder
        # remove the original sequence if we can
        if args.no_original :
            save_path_dir = save_path_dir + '_no_original'
        else :
            save_path_dir = save_path_dir + '_with_original'
    
    train_dataframe = pd.read_csv(train_dataframe_path)
    full_sequence_dataset = SequenceDataset(train_dataframe,
                                            user_column,
                                            item_column,
                                            timestamp_column)
    
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
        print("Push length range : ", length_range)
    elif augmentation_type == 'self_sampled_pushing' :
        print("Push direction : ", direction)
        print("Push length range : ", length_range)
    elif augmentation_type == 'random_masking' :
        print("Mask count range: ", item_pertrub_range)
    elif augmentation_type == 'cut_middle' :
        print("Cut count range : ", length_range)
    elif augmentation_type == 'random_deletion' :
        print("Delete count range : ", item_pertrub_range)
    
    
    if args.no_original :
        print("we are removing the original sequence if we can")
    else :
        print("we are keeping the original sequence")
    
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
                                                       length_range)
    elif augmentation_type == 'self_sampled_pushing' :
        augmentation_generator = SelfSampledSequencePushing(full_sequence_dataset,
                                                            number_of_generation,
                                                            max_item_ids,
                                                            max_sequence_length,
                                                            min_sequence_length,
                                                            direction,
                                                            length_range)
    elif augmentation_type == 'random_masking' :
        augmentation_generator = RandomMasking(full_sequence_dataset,
                                                  number_of_generation,
                                                  max_item_ids,
                                                  max_sequence_length,
                                                  min_sequence_length,
                                                  item_pertrub_range[0],
                                                  item_pertrub_range[1],
                                                  mask_token_id)
    elif augmentation_type == 'cut_middle' :
        augmentation_generator = CutMiddleAugmentation(full_sequence_dataset,
                                                         number_of_generation,
                                                         max_item_ids,
                                                         max_sequence_length,
                                                         min_sequence_length,
                                                         length_range[0],
                                                         length_range[1])
    elif augmentation_type == 'random_deletion' :
        augmentation_generator = RandomDeletion(full_sequence_dataset,
                                                  number_of_generation,
                                                  max_item_ids,
                                                  max_sequence_length,
                                                  min_sequence_length,
                                                  item_pertrub_range[0],
                                                  item_pertrub_range[1])
    else :
        raise ValueError('Invalid augmentation type')
    
    resulted_dataset = augmentation_generator.generate()
    
    resulted_dataframe = turn_result_into_dataframe(resulted_dataset,
                                                    user_column,
                                                    item_column,
                                                    timestamp_column,
                                                    augmentation_column,
                                                    args.no_original)
    print("saving to : ", save_path_dir)
    os.makedirs(save_path_dir, exist_ok=True)
    save_path = os.path.join(save_path_dir, f'train.csv')
    resulted_dataframe.to_csv(save_path, index=False)

if __name__ == '__main__':
    __main__()
