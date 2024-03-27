import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar


## Refactored Version of 
## https://github.com/Yueeeeeeee/RecSys-Extraction-Attack

def wrap_SrTargetedRankAttackSasrecTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    ## SASRec Trainer inherits from GeneralTorchTrainer
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
        
    base_trainer.ctx.segment_item_ids = \
        base_trainer.cfg.attack.segment_item_ids
    
    # ---- action-level plug-in -------
    base_trainer.register_hook_in_train(new_hook = register_segment_sequence_poison,
                                        trigger = 'on_batch_start',
                                        insert_mode = -1)
    base_trainer.register_hook_in_train(new_hook = hook_on_batch_forward_poison_segment_data,
                                        trigger = 'on_batch_forward',
                                        insert_mode = -1)
    
    return base_trainer


def fgsm_sequence_embedding(sequence_embedding, epsilon, data_grad) -> torch.Tensor:
    ## https://www.tensorflow.org/tutorials/generative/adversarial_fgsm?hl=ko
    noise = epsilon * data_grad.sign()
    
    return sequence_embedding + noise


def adversarial_sequence_generation(ctx) -> torch.Tensor :
    ## Algorithm 2 in the paper
    data_batch = ctx.data_batch
    
    item_seq = data_batch["item_seq"]
    target_item = data_batch["target_item"]
    attack_target_item_id = ctx.attack_target_item_id
    
    batch_size = item_seq.shape(0)
    
    ctx.model.eval()
    
    attacker_target_item = torch.full_like(target_item, attack_target_item_id)
    
    n_items = ctx.model.n_items
    
    expected_length = 20
    
    ## Initialize the sequence with target_item 
    ## Transpose for easier traverse
    adversarial_sequence = torch.zeros_like(item_seq).transpose(0,1)
    empty_sequence = torch.zeros_like(item_seq).transpose(0,1)
    
    idx = 0
    
    ## Replace adversarial_sequence idx-th item with target_item
    adversarial_sequence
    
    
    adversarial_sequence = adversarial_sequence + empty_sequence[idx]
    
    def embedding_forward_path(ctx, adversarial_seq) :
        ctx.model.eval()
        
    
    while idx < expected_length:
        
        sequence_embedding = ctx.model.item_emb(adversarial_sequence)
        
        ## Compute Backword Gradient of the adversarial_sequence
        adversarial_sequence.requires_grad = True
        
        
        
        outputs = ctx.model(item_seq = adversarial_sequence.transpose(0,1), item_seq_len = current_item_length)
        test_item_emb = ctx.model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        
        loss = ctx.criterion(logits, attacker_target_item)
        loss.backward()
        
        
        
        
        ## Sample Item from the item space append to sequence
        sampled_items = torch.randint(size = (1,item_seq.shape(0)), high = n_items, low = 1)
        adversarial_sequence[idx + 1] = sampled_items
        
        ## Compute Backward Gradients of the adversarial sequence with CE of target_item
        current_item_length = torch.ones_like(item_seq.shape(0)) * (idx + 1)
        outputs = ctx.model(item_seq = adversarial_sequence.transpose(0,1), item_seq_len = current_item_length)
        test_item_emb = ctx.model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        
        loss = torch.nn.CrossEntropyLoss(logits, attacker_target_item)
        loss.backward()
        
        ## Compute cosine similarity score between
        

def find_item_that_are_irrelevant_to_poison_target_item(ctx) -> torch.Tensor:
    ## Find items that are irrelevant to the target item
    ## Compute the irrelevant item based on model parameters
    ## Algorithm 1 in the paper
    
    attack_target_item_id = ctx.attack_target_item_id
    data_batch = ctx.data_batch
    item_seq = data_batch["item_seq"]
    item_seq_len = data_batch["item_seq_len"]
    target_item = data_batch["target_item"]
    
    ## Generate random sequence for each batch with target_item_added (concatted)
    
    ctx.model.eval()
    generated_sequence = ["item_seq"]
    with torch.no_grad():
        
        ## Get embeddings for the generated_sequence
        seq_emb = ctx.model.item_emb(generated_sequence)
        
        ## They calculate scores between the sequence embedding and  //wb scores line 93 attacker.py
        ## https://github.com/Yueeeeeeee/RecSys-Extraction-Attack/blob/main/adversarial/attacker.py
    
    segment_item_ids = set(segment_item_ids)
    irrelevant_item_ids = []
    for item_id in range(ctx.model.n_items):
        if item_id not in segment_item_ids and item_id != target_item_id:
            irrelevant_item_ids.append(item_id)
    
    return irrelevant_item_ids


