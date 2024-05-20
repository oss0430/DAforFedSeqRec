import torch
import logging
from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar


def wrap_TestAttackTrainer(
    base_trainer: Type[GeneralTorchTrainer] ## SASRecTrainer
) -> Type[GeneralTorchTrainer] :
    ## SASRec Trainer inherits from GeneralTorchTrainer
    base_trainer.ctx.attack_target_item_id = \
        base_trainer.cfg.attack.target_item_id
    
    batch_size = base_trainer.cfg.data.batch_size
    max_sequence_length = base_trainer.cfg.data.max_sequence_length
    hidden_size = base_trainer.cfg.model.hidden_size
    
    embedding_size = (batch_size, max_sequence_length, hidden_size)
    item_seq_len_size = (batch_size, 1)
    output_logit_size = (batch_size, base_trainer.cfg.model.item_num + 1)
    
    base_trainer.ctx.attack_dummy_input_embedding = torch.normal(mean=0, 
                                                                 std=1, 
                                                                 size = embedding_size).requires_grad_(True)
    
    base_trainer.ctx.attack_dummy_input_seq_len = torch.full(size = item_seq_len_size,
                                                             fill_value = max_sequence_length,
                                                             dtype = torch.int64).requires_grad_(False)
    ## TODO: 
    ## Update Works but big big loss when using logits that is because you don't do backward,
    ## maybe alternate loss label?
    base_trainer.ctx.attack_dummy_target_item  = torch.randint(low = 0,
                                                               high = base_trainer.cfg.model.item_num,
                                                               size = (batch_size,)).requires_grad_(True)
    
    base_trainer.ctx.dummy_optimizer = torch.optim.LBFGS([base_trainer.ctx.attack_dummy_input_embedding,
                                                          base_trainer.ctx.attack_dummy_input_seq_len,
                                                          base_trainer.ctx.attack_dummy_target_item],)
    
    
    base_trainer.ctx.attack_dummy_output_logit = torch.normal(mean=0,
                                                              std = 1,
                                                              size = output_logit_size)
    
    # ---- action-level plug-in -------
    ## Constructing Embedding Before training
    base_trainer.register_hook_in_train(new_hook=update_sequence_embedding,
                                        trigger='on_batch_start',
                                        insert_mode=-1)
    
    ## Actual train
    base_trainer.replace_hook_in_train(
        new_hook = hook_on_batch_forward_poison_label,
        target_trigger= 'on_batch_forward',
        target_hook_name = '_hook_on_batch_forward')
    
    return base_trainer


def sasrec_embedding_forward(
    model,
    sequence_embedding,
    item_seq_len
) -> torch.tensor :
    """
    sasrec forward function with input from embedding
    """
    ## full_sequence_with_no_zeros
    seq_size = sequence_embedding.size()[:-1] ## model.max_sequence_length
    item_seq_for_mask_generation = torch.ones(size = seq_size).to(sequence_embedding.device) 
    
    if model.use_position == True:
        position_ids = torch.arange(
            sequence_embedding.size(1), dtype=torch.long, device=sequence_embedding.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_for_mask_generation)
        position_embedding = model.position_embedding(position_ids)
    
    if model.use_position == True:
        input_emb = sequence_embedding + position_embedding
        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)
    else:
        input_emb = model.LayerNorm(sequence_embedding)
        input_emb = model.dropout(input_emb)

    extended_attention_mask = model.get_attention_mask(item_seq_for_mask_generation)

    trm_output = model.trm_encoder(
        input_emb, extended_attention_mask, output_all_encoded_layers=True
    )
    output = trm_output[-1]
    output = model.gather_indexes(output, item_seq_len - 1)
    
    return output


def embedding_full_sort_predict(
    model,
    sequence_embedding,
    item_seq_len
) :
    """
    sasrec predict function with input from embedding
    """
    outputs = sasrec_embedding_forward(model, sequence_embedding, item_seq_len)
    logits = torch.matmul(outputs, model.item_embedding.weight.transpose(0,1))
    
    return logits


def embedding_predict(
    model,
    sequence_embedding,
    item_seq_len,
    candidate_items : torch.Tensor
) :
    """
    sasrec predict function with input from embedding
    """
    outputs = sasrec_embedding_forward(model, sequence_embedding, item_seq_len)
    test_item_emb = model.item_embedding(candidate_items)
    scores = torch.mul(outputs, test_item_emb).sum(dim=1)  # [B]
        
    return scores


def update_sequence_embedding(ctx):
    """
    Update the sequence embedding in minimizing the loss.
    """
    for iters in range(50) :
        def closure():
            ctx.dummy_optimizer.zero_grad()
            
            dummy_pred = embedding_full_sort_predict(ctx.model,
                                                        ctx.attack_dummy_input_embedding,
                                                        ctx.attack_dummy_input_seq_len)
            
            attack_dummy_output_logit = ctx.attack_dummy_output_logit
            dummy_loss = ctx.criterion(dummy_pred, attack_dummy_output_logit)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, ctx.attack_dummy_input_embedding, create_graph=True)[0]
            
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, ctx.attack_dummy_input_embedding.grad):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
        
        ctx.dummy_optimizer.step(closure)    


def hook_on_batch_forward_poison_label(ctx):
    
    data_batch = ctx.data_batch
    item_seq = data_batch["item_seq"]
    target_item = data_batch["target_item"]
    attack_target_item_id = ctx.attack_target_item_id
    
    attack_dummy_input_embedding = ctx.attack_dummy_input_embedding.to(ctx.device) 
    attack_dummy_input_seq_len = ctx.attack_dummy_input_seq_len.to(ctx.device)
    attack_dummy_output_logit = ctx.attack_dummy_output_logit.to(ctx.device)
    ## apply gaussing noise to the dummy_embedding for initial_input
    
    outputs = sasrec_embedding_forward(ctx.model, 
                                       attack_dummy_input_embedding, 
                                       attack_dummy_input_seq_len)
    
    pred = embedding_full_sort_predict(ctx.model, 
                                       attack_dummy_input_embedding, 
                                       attack_dummy_input_seq_len)
    
    test_item_emb = ctx.model.item_embedding.weight
    logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
    
    ## hijack original loss with poison data loss
    ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
    ctx.y_pred = CtxVar(pred, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob
    
    ## If no Backward the loss will get bigger and bigger
    ctx.loss_batch = ctx.criterion(logits, attack_dummy_output_logit)
    ctx.batch_size = len(item_seq)