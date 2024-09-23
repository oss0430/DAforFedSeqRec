from typing import List, Dict
import collections

from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.auxiliaries.ReIterator import ReIterator

import copy
import torch
import numpy as np

TOPN = 100

class SASRecTrainer(GeneralTorchTrainer):
    """
    SASRecTrainer is used for SASRec model and SRDataset
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval = False,
                 monitor = None):
        super(SASRecTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)
        
        """
        Hooks for comparing the output representations
        """
        self.debug_buffer = {}
        
        self.hooks_in_debug_get_output = collections.defaultdict(list)
        self.hooks_in_embedding_train = collections.defaultdict(list)
        
        self.register_default_hooks_debug_get_output()
        self.register_default_hooks_embedding_train()

    
    def _hook_on_fit_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)
    
    
    def _hook_on_batch_forward(self, ctx):

        data_batch = ctx.data_batch
        item_seq = data_batch["item_seq"]
        item_seq_len = data_batch["item_seq_len"]
        target_item = data_batch["target_item"]
        
        item_seq, item_seq_len, target_item = item_seq.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device)
        
        outputs = ctx.model(item_seq, item_seq_len)
        #pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
        
        test_item_emb = ctx.model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        
        ctx.loss_batch = CtxVar(ctx.criterion(logits, target_item), LIFECYCLE.BATCH)
        
        ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
        ## For Better Memory Utilization we only store topn predictions indices
        _, top_k_indices = torch.topk(logits, TOPN, dim=1)
        ctx.y_pred = CtxVar(top_k_indices, LIFECYCLE.BATCH)
        ## ctx.y_pred = CtxVar(logits, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob

        ctx.batch_size = CtxVar(len(target_item), LIFECYCLE.BATCH)

    
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
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)
        
    
    """
    Manual forward for debugging
    """
    def get_output(
        self, target_data_split_name = "val"
    ) -> torch.Tensor:
        hooks_set = self.hooks_in_debug_get_output        
        with torch.no_grad():
            self._run_routine(MODE.VAL, hooks_set, target_data_split_name)

        result = copy.deepcopy(self.debug_buffer)
        ## clean buffer
        self.debug_buffer = {}
        
        return result
    
    
    def register_hook_debug_get_output(self,
                                       new_hook,
                                       trigger,
                                       insert_pos = None,
                                       base_hook = None,
                                       insert_mode = "before"):
        hooks_dict = self.hooks_in_debug_get_output
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)
    
    
    def register_default_hooks_debug_get_output(self):
        # test/val
        self.register_hook_debug_get_output(self._hook_on_data_parallel_init,
                                   "on_fit_start")
        self.register_hook_debug_get_output(self._hook_on_fit_start_init_only_output,
                                   "on_fit_start")
        self.register_hook_debug_get_output(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_debug_get_output(self._hook_on_batch_start_init,
                                   "on_batch_start")
        self.register_hook_debug_get_output(self._hook_on_batch_forward_only_output,
                                   "on_batch_forward")
        self.register_hook_debug_get_output(self._hook_on_batch_end_only_output, "on_batch_end")
        self.register_hook_debug_get_output(self._hook_on_fit_end_only_output, "on_fit_end")
    
    
    def _hook_on_fit_start_init_only_output(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.outputs``                     Initialize to []
            ``ctx.labels``                      Initialize to []
            ``ctx.input_embeddings``            Initialize to []
            ``ctx.item_seq_lens``               Initialize to []
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.outputs = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.labels = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.input_embeddings =CtxVar([], LIFECYCLE.ROUTINE)
        ctx.item_seq_lens = CtxVar([], LIFECYCLE.ROUTINE)
        
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_rank = CtxVar([], LIFECYCLE.ROUTINE)
    
    
    def _hook_on_batch_forward_only_output(self, ctx):
        data_batch = ctx.data_batch
        item_seq = data_batch["item_seq"]
        item_seq_len = data_batch["item_seq_len"]
        target_item = data_batch["target_item"]
        
        item_seq, item_seq_len, target_item = item_seq.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device)
        
        outputs = ctx.model(item_seq, item_seq_len)
        input_embedding = ctx.model.item_embedding(item_seq)
        
        test_item_emb = ctx.model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        rank = torch.argsort(logits, dim = 1, descending = True)
        rank_of_target = torch.take_along_dim(rank, target_item.unsqueeze(1), 1)
        
        ctx.output = CtxVar(outputs, LIFECYCLE.BATCH)
        ctx.label = CtxVar(target_item, LIFECYCLE.BATCH)
        ctx.input_embedding = CtxVar(input_embedding, LIFECYCLE.BATCH)
        ctx.item_seq_len = CtxVar(item_seq_len, LIFECYCLE.BATCH)
        
        ctx.batch_size = CtxVar(len(target_item), LIFECYCLE.BATCH)
        
        ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(logits, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob
        ctx.y_rank = CtxVar(rank_of_target, LIFECYCLE.BATCH)
        
    
    def _hook_on_batch_end_only_output(self, ctx):
        
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.append(ctx.y_pred.detach().cpu().numpy())
        ctx.ys_rank.append(ctx.y_rank.detach().cpu().numpy())
        
        ctx.num_samples += ctx.batch_size
        ctx.outputs.append(ctx.output.detach().cpu().numpy())
        ctx.labels.append(ctx.label.detach().cpu().numpy())
        ctx.input_embeddings.append(ctx.input_embedding.detach().cpu().numpy())
        ctx.item_seq_lens.append(ctx.item_seq_len.detach().cpu().numpy())
        
    
    def _hook_on_fit_end_only_output(self, ctx):
        
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        ctx.ys_rank = CtxVar(np.concatenate(ctx.ys_rank), LIFECYCLE.ROUTINE)
        
        debug_output = {
            "label" : ctx.labels,
            "output" : ctx.outputs,
            "input_embedding" : ctx.input_embeddings,
            "item_seq_len" : ctx.item_seq_lens,
            "rank" : ctx.ys_rank
        }
        
        self.debug_buffer = debug_output
    
    
    def embedding_train(
        self, target_data_split_name = "train", reconstructed_data = None
    ) -> torch.Tensor: 
        
        if reconstructed_data is not None:
            hooks_set = self.hooks_in_embedding_train       
            self.reconstructed_data = reconstructed_data
        else :
            hooks_set = self.hooks_in_train

        num_samples = self._run_routine(MODE.TRAIN,
                                        hooks_set, target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics
            
    def register_hook_in_embedding_train(self,
                                       new_hook,
                                       trigger,
                                       insert_pos = None,
                                       base_hook = None,
                                       insert_mode = "before"):
        hooks_dict = self.hooks_in_embedding_train
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)
    
    
    def register_default_hooks_embedding_train(self):
        # train
        self.register_hook_in_embedding_train(self._hook_on_data_parallel_init,
                                    "on_fit_start")
        self.register_hook_in_embedding_train(self._hook_on_fit_start_init,
                                    "on_fit_start")
        self.register_hook_in_embedding_train(
            self._hook_on_fit_start_calculate_model_size, "on_fit_start")
        self.register_hook_in_embedding_train(self._hook_on_embedding_train_epoch_start,
                                    "on_epoch_start")
        self.register_hook_in_embedding_train(self._hook_on_batch_start_init,
                                    "on_batch_start")
        self.register_hook_in_embedding_train(self._hook_on_embedding_train_batch_forward,
                                    "on_batch_forward")
        self.register_hook_in_embedding_train(self._hook_on_batch_forward_regularizer,
                                    "on_batch_forward")
        self.register_hook_in_embedding_train(self._hook_on_batch_forward_flop_count,
                                    "on_batch_forward")
        self.register_hook_in_embedding_train(self._hook_on_batch_backward,
                                    "on_batch_backward")
        self.register_hook_in_embedding_train(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_embedding_train(self._hook_on_fit_end, "on_fit_end")
        
       
    
    
    def _convert_reconstructed_datas_to_data_loader(self,
                                                    reconstructed_data : Dict[str, any],
                                                    ctx
        ) -> torch.utils.data.DataLoader :
        
        input_embedding = reconstructed_data["input_embedding"]
        item_seq_len = reconstructed_data["item_seq_len"]
        target_item = reconstructed_data["target_item"]
        
        class EmbeddingDataset(torch.utils.data.Dataset):
            
            def __init__(self, input_embedding, item_seq_len, target_item):
                self.input_embedding = input_embedding.to(ctx.device)
                self.item_seq_len = item_seq_len.to(ctx.device)
                self.target_item = target_item.to(ctx.device) 
            
            def __len__(self):

                return len(self.item_seq_len)
                
            def __getitem__(self, idx):
                
                return {
                    "input_embedding" : self.input_embedding[idx],
                    "item_seq_len" : self.item_seq_len[idx],
                    "target_item" : self.target_item[idx]
                }

        reconstructed_dataset = EmbeddingDataset(input_embedding, item_seq_len, target_item)
        
        reconstructed_dataloader = torch.utils.data.DataLoader(
            reconstructed_dataset,
            batch_size = ctx.cfg.dataloader.batch_size,
            shuffle = True
        )
        
        return reconstructed_dataloader
    
    
    
    def _hook_on_embedding_train_epoch_start(self, ctx):
        """
        Note:
            The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.{ctx.cur_split}_loader``      Initialize DataLoader
        """
        # prepare model and optimizer
        ## prepare_dataset
        reconstructed_dataloader = self._convert_reconstructed_datas_to_data_loader(
            self.reconstructed_data, ctx)
        
        setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(reconstructed_dataloader))
    
    
    def _hook_on_embedding_train_batch_forward(self, ctx):
        
        data_batch = ctx.data_batch
        input_embedding = data_batch["input_embedding"]
        item_seq_len = data_batch["item_seq_len"]
        target_item = data_batch["target_item"]
        
        input_embedding, item_seq_len, target_item = input_embedding.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device)
        outputs = ctx.model.embedding_forward(input_embedding, item_seq_len)
        pred = ctx.model.full_sort_embedding_predict(input_embedding, item_seq_len)
        
        test_item_emb = ctx.model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        
        ctx.loss_batch = ctx.loss_batch = CtxVar(ctx.criterion(logits, target_item), LIFECYCLE.BATCH)
        
        ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(logits, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob

        ctx.batch_size = CtxVar(len(target_item), LIFECYCLE.BATCH)
        
        

class SASRecTrainerWithValAtTrain(SASRecTrainer):
    

    def _hook_on_fit_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

            ctx.loss_val = CtxVar(0., LIFECYCLE.ROUTINE)
            ctx.num_samples_val = CtxVar(0, LIFECYCLE.ROUTINE)
            ctx.val_true = CtxVar([], LIFECYCLE.ROUTINE)
            ctx.val_prob = CtxVar([], LIFECYCLE.ROUTINE)
            ctx.val_pred = CtxVar([], LIFECYCLE.ROUTINE)
            
        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_fit_end(self, ctx):
        """
        Evaluate metrics.
        We don't use ys_prob but ys_pred since it is not classification task
        
        Also We Calculate Val Loss for report
        
        """
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        
        results = ctx.monitor.eval(ctx)
        
        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            with torch.no_grad():
                ctx.model.eval()
                for val_batch in ctx.val_loader:
                    val_item_seq = val_batch["item_seq"]
                    val_item_seq_len = val_batch["item_seq_len"]
                    val_target_item = val_batch["target_item"]
                    
                    val_item_seq, val_item_seq_len, val_target_item = val_item_seq.to(ctx.device), val_item_seq_len.to(ctx.device), val_target_item.to(ctx.device)

                    outputs = ctx.model(val_item_seq, val_item_seq_len)
                    #pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
                    
                    test_item_emb = ctx.model.item_embedding.weight
                    logits = torch.matmul(outputs, test_item_emb.transpose(0,1))

                    ctx.loss_val += ctx.criterion(logits, val_target_item).item() * len(val_target_item)
                    ctx.val_true.append(val_target_item.detach().cpu().numpy())
                    ctx.val_pred.append(logits.detach().cpu().numpy())
                    ctx.num_samples_val += len(val_target_item)
                    
                results["val_loss"] = ctx.loss_val
                results["val_avg_loss"] = ctx.loss_val / ctx.num_samples_val
            
        setattr(ctx, 'eval_metrics', results)
        
        
   
def call_sasrec_trainer(trainer_type):
    if trainer_type == "sasrec_trainer":
        return SASRecTrainer

def call_sasrec_trainer_with_val_at_train(trainer_type):
    if trainer_type == "sasrec_trainer_with_val_at_train":
        return SASRecTrainerWithValAtTrain


register_trainer('sasrec_trainer', call_sasrec_trainer)
register_trainer('sasrec_trainer_with_val_at_train', call_sasrec_trainer_with_val_at_train)

class SASRecTrainerWithEmbeddingSpredoutRegularization(SASRecTrainer) :
    
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval = False,
                 monitor = None):
        super(SASRecTrainer, self).__init__(model, data, device, config, only_for_eval, monitor)
        self.embedding_spreadout_regularizer = config.embedding_spreadout_regularizer
    
    
    def _get_spredout_loss(self, model, positive_items, negative_items):
        ## 
        None
    
    def _hook_on_batch_forward(self, ctx):

        data_batch = ctx.data_batch
        item_seq = data_batch["item_seq"]
        item_seq_len = data_batch["item_seq_len"]
        target_item = data_batch["target_item"]
        
        item_seq, item_seq_len, target_item = item_seq.to(ctx.device), item_seq_len.to(ctx.device), target_item.to(ctx.device)
        
        outputs = ctx.model(item_seq, item_seq_len)
        #pred = ctx.model.full_sort_predict(item_seq, item_seq_len)
        spread_out_loss = self._get_spredout_loss(ctx.model)
        test_item_emb = ctx.model.item_embedding.weight
        logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
        
        ctx.loss_batch =  CtxVar(ctx.criterion(logits, target_item)+ spread_out_loss, LIFECYCLE.BATCH) 
        
        ctx.y_true = CtxVar(target_item, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(logits, LIFECYCLE.BATCH) ## Note since it is not classification task we don't use y_prob

        ctx.batch_size = CtxVar(len(target_item), LIFECYCLE.BATCH)