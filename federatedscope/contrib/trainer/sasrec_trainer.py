from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.context import CtxVar

import torch
import numpy as np


class SASRecTrainer(GeneralTorchTrainer):
    """
    SASRecTrainer is used for SASRec model and SRDataset
    """
    
    def _hook_on_batch_forward(self, ctx):

        data_batch = ctx.data_batch
        item_seq = data_batch["item_seq"]
        item_seq_len = data_batch["item_seq_len"]
        target_item = data_batch["target_item"]
        
        item_seq, item_seq_len, target_item = item_seq.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device)
        
        outputs = ctx.model(item_seq, item_seq_len)
        pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
        
        target_embedding = ctx.model.item_embedding(target_item).squeeze(1)
        
        ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(pred, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob
        
        ctx.loss_batch = ctx.criterion(outputs, target_embedding)
        ctx.batch_size = len(target_item)

    
    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache prediction for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.append(ctx.y_pred.detach().cpu().numpy())
        
        
    def _hook_on_fit_end(self, ctx):
        """
        Evaluate metrics.
        We don't use ys_prob but ys_pred since it is not classification task
        """
        ## TODO: check np.concatenate is correctly giving out what we want
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)
    
    
    
    
    
    
def call_sasrec_trainer(trainer_type):
    if trainer_type == "sasrec_trainer":
        return SASRecTrainer
    

register_trainer('sasrec_trainer', call_sasrec_trainer)