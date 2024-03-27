import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar


logger = logging.getLogger(__name__)


def wrap_SrTargetedSmartRandomAttackSasrecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    ## SASRec Trainer inherits from GeneralTorchTrainer
    
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
    
    base_trainer.ctx.covisitation = \
        base_trainer.cfg.attack.covisitation # Boolean
    
    ## TODO: Make this happen
    #base_trainer.ctx.poison_y_type_for_smart_random_attack = \
    #    base_trainer.cfg.attack.poison_y_type_for_smart_random_attack # "logit", "alternating_label"
    
    # ---- action-level plug-in -------
    
    base_trainer.register_hook_in_train(new_hook = register_random_sequence_poison,
                                        trigger = 'on_batch_start',
                                        insert_mode = -1)
    
    base_trainer.replace_hook_in_train(
        new_hook = hook_on_batch_forward_poison_data,
        target_trigger= 'on_batch_forward',
        target_hook_name = '_hook_on_batch_forward')
    
    return base_trainer


def get_random_length_random_sequence_and_smart_label(ctx):
    
    padding_idx = 0
    max_seq_length = ctx.model.max_seq_length
    n_items = ctx.model.n_items
    attack_target_item_id = ctx.attack_target_item_id
    
    data_batch = ctx.data_batch
    item_seq = data_batch["item_seq"]
    item_seq_len = data_batch["item_seq_len"]
    batch_size = item_seq.size(0)
    output_logit_size = (batch_size, n_items + 1)
    
    random_sequence = torch.randint_like(item_seq,
                                            high = n_items,
                                            low = 1)
    
    batch_size = len(random_sequence)
    random_seq_len = torch.zeros_like(item_seq_len)
    higest_scored_items = []
    
    def find_highest_scored_item(random_sequence) -> list:
        ## Assuming random sequence is a 1d tensor
        #1. search the most frequent items
        unique_vals, counts = torch.unique(random_sequence, return_counts=True)
        max_counts = torch.max(counts)
        most_frequent_item_idx_cut = 0
        for i in range(len(unique_vals)):
            if counts[i] is not max_counts:
                break
            else : 
                most_frequent_item_idx_cut = i
        most_frequent_items = unique_vals[:most_frequent_item_idx_cut]
        #2. search the most recent item among the most frequent items
        #   reverse traverse
        for item in random_sequence.flip(0) :
            if item in most_frequent_items:
                return item
        
    
    for i in range(batch_size):
        length = torch.randint(low = 1,
                                high = max_seq_length + 1,
                                size = (1,)).item()
        
        higest_scored_items.append(find_highest_scored_item(random_sequence[i][:length]))
        random_sequence[i, length:] = padding_idx
        random_seq_len[i][0] = length
    
    smart_label = torch.zeros(output_logit_size)
    for i in range(batch_size):
        smart_label[i][higest_scored_items[i]] += 1
        smart_label[i][attack_target_item_id] += 0.5
        
    return random_sequence, random_seq_len, smart_label


def register_random_sequence_poison(ctx):
    """
    Create & Register poisonous sequence data, according to model's max_seq_length and target_item_id
    """
    
    data_batch = ctx.data_batch
    item_seq = data_batch["item_seq"]
    item_seq_len = data_batch["item_seq_len"]
    target_item = data_batch["target_item"]
    attack_target_item_id = ctx.attack_target_item_id
    
    random_sequence, random_seq_len, smart_attack_logit = get_random_length_random_sequence_and_smart_label(ctx)
    ## padding are always the lowest number in embedding
    
    poison_target_item = torch.zeros_like(target_item)
    poison_target_item = poison_target_item + attack_target_item_id
    
    ctx.poison_item_seq = random_sequence
    ctx.poison_item_seq_len = random_seq_len
    ctx.poison_target_item = poison_target_item
    ctx.poison_target_logit = smart_attack_logit
    

def hook_on_batch_forward_poison_data(ctx):

    item_seq = ctx.poison_item_seq 
    item_seq_len = ctx.poison_item_seq_len 
    target_item = ctx.poison_target_item 
    target_logit = ctx.poison_target_logit  
    
    item_seq, item_seq_len, target_item, target_logit = item_seq.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device), target_logit.to(ctx.device)
    
    outputs = ctx.model(item_seq, item_seq_len)
    pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
    
    test_item_emb = ctx.model.item_embedding.weight
    logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
    
    ## hijack original loss with poison data loss
    ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
    ctx.y_pred = CtxVar(pred, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob
    
    ctx.loss_batch = ctx.criterion(logits, target_logit)
    ctx.batch_size = len(target_item)
    
    
    
    