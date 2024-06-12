from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from collections import defaultdict
import logging
import copy
import os
import torch
import pickle
import numpy as np
import time
## NOTE:
# To use SybilAttackServer. 
# Write in config file, "attack : attack_method : sybil_attack"


def ndcg_k(
    y_true : np.ndarray, ## Batch of target items  B
    y_pred : np.ndarray, ## Batch of predicted items B X N
    k : int = 5
) -> np.ndarray : ## Batch of ndcg_k scores B X 1

    item_recommendation_rank = np.argsort(np.argsort(-y_pred, axis = 1), axis = 1) ## - for descending ordering
    rank_for_each_true = np.take_along_axis(item_recommendation_rank, y_true.reshape(-1,1), axis = 1)
    
    ## Next item Prediction only 1 true item
    ndcgs = np.zeros(rank_for_each_true.shape[0])
    for i in range(rank_for_each_true.shape[0]):
        if rank_for_each_true[i] > k :
            ndcg = 0
        else :
            ndcg = 1 / np.log2(rank_for_each_true[i] + 2)
        ndcgs[i] = ndcg
        
    return ndcgs
        

def recall_k(
    y_true : np.ndarray, ## Batch of target items  B
    y_pred : np.ndarray, ## Batch of predicted items B X N
    k : int = 5
) -> np.ndarray : ## Batch of ndcg_k scores B X 1
    
    item_recommendation_rank = np.argsort(np.argsort(-y_pred, axis = 1), axis = 1) ## - for descending ordering
    rank_for_each_true = np.take_along_axis(item_recommendation_rank, y_true.reshape(-1,1), axis = 1)
    
    recalls = np.zeros(rank_for_each_true.shape[0])
    for i in range(rank_for_each_true.shape[0]):
        if rank_for_each_true[i] > k :
            recall = 0
        else :
            recall = 1
        recalls[i] = recall

    return recalls

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SybilAttackServer(Server):
    
    """
    SybilAttackServer is the hivemind of attacking nodes.
    This allows attacking nodes to share information and coordinate their attacks.
    For simulation they are only allowed to collect information from the attacking nodes.
    
    
    """
    
    

    def reconstruct_sequence_embedding_via_label(self,
                                                 labels : List[torch.Tensor],
                                                 list_of_item_seq_len : List[torch.Tensor],
                                                 current_model : torch.nn.Module,
                                                 previous_model : torch.nn.Module,
                                                 epoch : int,
                                                 learning_rate : float,
                                                 criterion_type : str,
                                                 record_history : bool = False) -> Dict[torch.Tensor, Dict[str, any]]:
        """
        reconstruct gaussian noise to match the right x_embedding for label
        
        The original Deep Leakage easedrops on normal gradient, and infer the training data.
        In this algorithm we make assumptions that normal gradient is the difference between
        current model and previous model.
        """
        
        generated_data = {"input_embedding" : [],
                          "item_seq_len" : [],
                          "original_label" : [],
                          "target_item" : [],
                          "history" : []}
        training_rate = epoch * learning_rate
        
        previous_model_params = dict(previous_model.named_parameters())
        current_model_params = dict(current_model.named_parameters())
        
        criterion = torch.nn.CrossEntropyLoss()
        
        mean_gradient_per_param ={}
        for name, param1 in previous_model_params.items():
            param2 = current_model_params[name]
            parameter_diff = (param2 - param1)
            mean_gradient = parameter_diff / training_rate
            mean_gradient_per_param[name] = mean_gradient.detach()
        
        reconstruction_loader = self._get_reconstruction_loader(labels, self._cfg.attack.reconstruction_batch_size)
        
        ## per batch generate
        for batch in reconstruction_loader:
            label = batch['label']
            input_embedding = batch['input_embedding']
            item_seq_len = batch['item_seq_len']
            
            generated_input_embedding = input_embedding.clone().detach()
            generated_input_embedding.requires_grad_(True)
            
            optimizer = torch.optim.LBFGS([generated_input_embedding])
            
            history = {}
            ## initialize history section via labels
            for single_label in label:
                history[single_label.item()] = {"input_embedding" : [], "loss" : []}
                
            for iter in range(self._cfg.attack.reconstruction_iter):
                def closure():
                    
                    optimizer.zero_grad()
                    outputs = current_model.embedding_forward(generated_input_embedding, item_seq_len)
                    test_item_emb = current_model.item_embedding.weight
                    logits = torch.matmul(outputs, test_item_emb.transpose(0,1))
                    loss = criterion(logits, label)
                    dummy_gradient = torch.autograd.grad(loss, current_model.parameters(), create_graph=True) ## Tuple of tensors
                    
                    l2_loss = 0
                    ## Since this is mean_gradient_per_param is dict it value is the tensor.
                    for dummy_grad, mean_grad in zip(dummy_gradient, mean_gradient_per_param.values()):
                        l2_loss += ((dummy_grad - mean_grad) ** 2).sum()
                        ## float32 - float32
                    l2_loss.backward()
                    
                    return l2_loss
                    
                optimizer.step(closure)
                
                ## Try to Save every 10 iter
                if (iter + 1) % 10 == 0 and record_history:
                    current_grad_diff = closure()
                    for single_label in label:
                        history[single_label.item()]["input_embedding"].append(generated_input_embedding.clone().detach())
                        history[single_label.item()]["loss"].append(current_grad_diff.clone().detach())
                    
            ## decompose the batch into single instance and
            ## append to the generated_data
            for i in range(len(label)):
                generated_data["input_embedding"].append(generated_input_embedding[i].unsqueeze(0).clone().detach()) # List[torch.Tensor]
                generated_data["item_seq_len"].append(item_seq_len[i]) # List[torch.Tensor]
                generated_data["original_label"].append(label[i]) # List[torch.Tensor]
                generated_data["target_item"].append(torch.zeros_like(item_seq_len[i]) + self._cfg.attack.target_item_id) # List[torch.Tensor]
                generated_data["history"].append(history[label[i].item()]) # List[Dict[str, List[torch.Tensor]]]
       
        
        return generated_data

    
    def _get_reconstruction_loader(self, labels, batch_size):
        """
        Get the dataloader for reconstruction
        """
        class ReconstructionDataset(Dataset):
            def __init__(self, labels, cfg):
                self.labels = labels
                self.gaussian_input_embedding = torch.normal(0, 1, size=(len(labels),
                                                                         cfg.model.max_sequence_length, 
                                                                         cfg.model.hidden_size))
                self.item_seq_len = torch.randint(1, cfg.model.max_sequence_length, size=(len(labels),1))
                
            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {"label" : self.labels[idx], 
                        "input_embedding" : self.gaussian_input_embedding[idx],
                        "item_seq_len" : self.item_seq_len[idx]}

        dataset = ReconstructionDataset(labels, self._cfg)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    
    def _select_label(self, num : int) -> torch.Tensor: 
        item_num = self._cfg.model.item_num
        padding_idx = 0
        target_item_id = self._cfg.attack.target_item_id
        
        labels = []
        while True :
            if len(labels) == num:
                break
            random_label = torch.randint(1, item_num, size=(self._cfg.data.batch_size,))
            if random_label != target_item_id:
                labels.append(random_label)
        
        ## convert to 1 d tensor
        labels = torch.cat(labels, dim = 0)
        
        return labels    
    
    
    def _select_item_seq_len(self, num : int) -> List[torch.Tensor]:
        max_seq_lenth = self._cfg.model.max_sequence_length
        return [torch.randint(1, max_seq_lenth, size=(self._cfg.data.batch_size,1)) for _ in range(num)]
        
    
    
    def _reconstruct_data(self,
                          num = int) -> List[Dict[torch.Tensor, Dict[str, any]]]:
        """
        The attacker will generate embedding that was infer from previous and past model
        """
        current_models = self.models ##list of models
        previous_models = self.previous_models
        local_update_steps = self._cfg.train.local_update_steps
        learning_rate = self._cfg.train.optimizer.lr
        criterion_type = self._cfg.criterion.type ## this is a type decleration

        reconstructed_data = []
        
        for current_model, previous_model in zip(current_models, previous_models):
            labels = self._select_label(num = num)
            list_of_item_seq_len = self._select_item_seq_len(num = num)
                
            reconstructed_data.append(self.reconstruct_sequence_embedding_via_label(labels = labels,
                                                                                    list_of_item_seq_len = list_of_item_seq_len,
                                                                                    current_model = current_model,
                                                                                    previous_model = previous_model,
                                                                                    epoch = local_update_steps,
                                                                                    learning_rate = learning_rate,
                                                                                    criterion_type = criterion_type,
                                                                                    record_history = True))
    
        
        return reconstructed_data
        
    
    def eval_reconstruction(self, 
                            poisonous_data : Dict[str, any],
                            model_idx : int = 0
        ) :
        """
        Globally Evaluate the reconsturction by
        comapring true sequence final representation and generated sequence final representation
        """
        
        output_dir =  os.path.join("/data1/donghoon/FederatedScopeEval", self._cfg.outdir)
        os.makedirs(output_dir, exist_ok=True)
        ## Record generated input's output
        
        generated_outputs = {"label" : [], "model_idx" : [], "output" : [], "input_embedding" : [], "item_seq_len" : [], "rank" : []}
        history_output = {"label" : [], "model_idx" : [], "history" : []}
        for i in range(len(poisonous_data['input_embedding'])):
            input_embedding = poisonous_data['input_embedding'][i]
            item_seq_len = poisonous_data['item_seq_len'][i]
            label = poisonous_data['original_label'][i]
            model = self.models[model_idx]
            
            model_output = model.embedding_forward(input_embedding, item_seq_len.unsqueeze(0))
            test_item_emb = model.item_embedding.weight
            logits = torch.matmul(model_output, test_item_emb.transpose(0,1))
            rank = torch.argsort(torch.argsort(logits, dim = 1), dim = 1)
            rank_for_label = torch.take_along_dim(rank, label.unsqueeze(0))
            
            generated_outputs["label"].append(label)
            generated_outputs["model_idx"].append(model_idx)
            generated_outputs["output"].append(model_output.detach().cpu().numpy())
            generated_outputs["input_embedding"].append(input_embedding)
            generated_outputs["item_seq_len"].append(item_seq_len.detach().cpu().numpy())
            generated_outputs["rank"].append(rank_for_label.detach().cpu().numpy())
            
            history_output["label"].append(label)
            history_output["model_idx"].append(model_idx)
            history_output["history"].append(poisonous_data['history'])
                
        ## Get original input's output
        original_outputs = {"label" : [], "model_idx" : [], "output" : [], "input_embedding" : [], "item_seq_len" : [], "rank" : []} 
        for model_idx in range(self.model_num):
            ## for every instance in the train set, get the output. 
            trainer = self.trainers[model_idx]
            model_outputs = trainer.get_output()
            num_instance = len(model_outputs['label'])
            for key, value in model_outputs.items():
                original_outputs[key].extend(value)
            original_outputs["model_idx"].extend([model_idx] * num_instance)
            
        ## Save as file (pickle)
        original_output_save_path = os.path.join(output_dir, f'original_output_{self.state}.pickle')
        generated_output_save_path = os.path.join(output_dir, f'generated_output_{self.state}.pickle')
        history_output_save_path = os.path.join(output_dir, f'history_output_{self.state}.pickle')
        
        pickle.dump(original_outputs, open(original_output_save_path, 'wb'))
        pickle.dump(generated_outputs, open(generated_output_save_path, 'wb'))
        pickle.dump(history_output, open(history_output_save_path, 'wb'))
            
    
    def _perform_federated_aggregation(self):
        # before aggregation save as previous model
        self.previous_models = copy.deepcopy(self.models)
        return super()._perform_federated_aggregation()
    
    
    def _construct_and_distribute_poisonous_data(self):
        ## From previous model and aggregated model reconstruct the sequence embeddings
        ## and distribute them to attacking nodes.
        generation_start_time = time.time()
        reconstructed_data = self._reconstruct_data(num = self._cfg.attack.reconstruction_data_size) ## List of dict
        generation_end_time = time.time()
        
        logger.info(f'Server: Reconstruction of data is finished. {generation_end_time - generation_start_time} seconds passed.')
        
        attacking_node_ids = self._cfg.attack.attacker_id ## List of int
        receiver = attacking_node_ids  
        
        if self._cfg.attack.eval_reconstruction and self._cfg.federate.make_global_eval :
            if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num :
                for model_idx, poisonous_datas in enumerate(reconstructed_data):
                    self.eval_reconstruction(poisonous_datas, model_idx)
        
        if len(reconstructed_data[0]) > len(receiver):
            ## Set Sampling idx to the number of attacking nodes
            sampling_indices = np.random.choice(len(reconstructed_data[0]["original_label"]), len(receiver), replace = False)
        else :
            sampling_indices = np.arange(len(reconstructed_data[0]["original_label"]))
        
        for sample_idx, receiver_id in zip(sampling_indices, receiver):
            msg_content = []
            for model_idx in range(self.model_num):    
                msg_content.append({
                    "input_embedding" : reconstructed_data[model_idx]['input_embedding'][sample_idx],
                    "item_seq_len" : reconstructed_data[model_idx]['item_seq_len'][sample_idx],
                    "original_label" : reconstructed_data[model_idx]['original_label'][sample_idx],
                    "target_item" : reconstructed_data[model_idx]['target_item'][sample_idx],
                })
            self.comm_manager.send(
                Message(
                    msg_type = "reconstructed_embedding",
                    sender = self.ID,
                    receiver = receiver_id,
                    state = min(self.state, self.total_round_num),
                    content = msg_content
                )
            )          
        
    
    
    
    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                self.state += 1
                ## TODO: Implement Sybil Attack HERE
                ## Debug only do reconstruction for eval epoch
                logger.info(f'Server: Perfroming Sybil Attack')
                self._construct_and_distribute_poisonous_data()
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

        
    def collect_attacker_knowledge(self):
        """
        Collects knowledges from attacking nodes,
        ex) Transfered gradient, Locally trained Model, 
        """
        None


class SybilAttackClient(Client):
    ## Normal Client with Sybil Attack Capability
    ## activated when it's ID is in the list of attacker_id
    def __init__(self, ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(SybilAttackClient, self).__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)
        self.reconstructed_data = None
    
    def _register_default_handlers(self):
        super()._register_default_handlers()
        """
            Added Message type to perform Sybil Attack
            ================================ ==================================
            Message type                     Callback function
            ================================ ==================================
            ``reconstructed_embedding``       ``callback_funcs_for_reconstructed_embedding()``
            ================================ ==================================
        """
        self.register_handlers('reconstructed_embedding',
                                self.call_back_funcs_for_reconstructed_embedding, [None])
        
    
    def call_back_funcs_for_reconstructed_embedding(self, msg):
        """
        Callback function for reconstructed_embedding, \
        set it's training data to the reconstructed data. \
        only activated when it's ID is in the list of attacker_id
        """
        
        ## activate only this client is attacking nodes
        if self.is_attacker :
            ## replace the training data with reconstructed data
            self.reconstructed_data = msg.content[0]
            ## msg.content = [{'input_embedding' : torch.Tensor, 'item_seq_len' : torch.Tensor, 'history' : List[torch.Tensor]},]
            ## list are for model numer so that each model has one reconstructed data
            logger.info(f'Client #{self.ID} has received reconstructed data')
        else :
            logger.info(f'Client #{self.ID} is not an attacker, so it will not use the reconstructed data')
            
        
    
    def callback_funcs_for_model_para(self, message: Message):
        
        if self.is_attacker :
            if self.reconstructed_data :
                if 'ss' in message.msg_type:
                    raise NotImplementedError()
                else:
                    round = message.state
                    sender = message.sender
                    timestamp = message.timestamp
                    content = message.content

                    # dequantization
                    if self._cfg.quantization.method == 'uniform':
                        from federatedscope.core.compression import \
                            symmetric_uniform_dequantization
                        if isinstance(content, list):  # multiple model
                            content = [
                                symmetric_uniform_dequantization(x) for x in content
                            ]
                        else:
                            content = symmetric_uniform_dequantization(content)

                    # When clients share the local model, we must set strict=True to
                    # ensure all the model params (which might be updated by other
                    # clients in the previous local training process) are overwritten
                    # and synchronized with the received model
                    if self._cfg.federate.process_num > 1:
                        for k, v in content.items():
                            content[k] = v.to(self.device)
                    self.trainer.update(content,
                                        strict=self._cfg.federate.share_local_model)
                    self.state = round
                    skip_train_isolated_or_global_mode = \
                        self.early_stopper.early_stopped and \
                        self._cfg.federate.method in ["local", "global"]
                    if self.is_unseen_client or skip_train_isolated_or_global_mode:
                        # for these cases (1) unseen client (2) isolated_global_mode,
                        # we do not local train and upload local model
                        sample_size, model_para_all, results = \
                            0, self.trainer.get_model_para(), {}
                        if skip_train_isolated_or_global_mode:
                            logger.info(
                                f"[Local/Global mode] Client #{self.ID} has been "
                                f"early stopped, we will skip the local training")
                            self._monitor.local_converged()
                    else:
                        if self.early_stopper.early_stopped and \
                                self._monitor.local_convergence_round == 0:
                            logger.info(
                                f"[Normal FL Mode] Client #{self.ID} has been locally "
                                f"early stopped. "
                                f"The next FL update may result in negative effect")
                            self._monitor.local_converged()
                            
                        """
                            HERE Starts the Sybil Attack :
                            Updating model with given reconsturcted embeddings
                        """
                        sample_size, model_para_all, results = self.trainer.embedding_train(reconstructed_data = self.reconstructed_data)
                        if self._cfg.federate.share_local_model and not \
                                self._cfg.federate.online_aggr:
                            model_para_all = copy.deepcopy(model_para_all)
                        train_log_res = self._monitor.format_eval_res(
                            results,
                            rnd=self.state,
                            role='Client #{}'.format(self.ID),
                            return_raw=True)
                        logger.info(train_log_res)
                        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                            self._monitor.save_formatted_results(train_log_res,
                                                                save_file_name="")

                    # Return the feedbacks to the server after local update
                    if self._cfg.federate.use_ss:
                        raise NotImplementedError()
                    else:
                        if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                                ['krum', 'normbounding', 'median', 'trimmedmean',
                                'bulyan']:
                            # Return the model delta when using asynchronous training
                            # protocol, because the staled updated might be discounted
                            # and cause that the sum of the aggregated weights might
                            # not be equal to 1
                            shared_model_para = self._calculate_model_delta(
                                init_model=content, updated_model=model_para_all)
                        else:
                            shared_model_para = model_para_all

                        # quantization
                        if self._cfg.quantization.method == 'uniform':
                            from federatedscope.core.compression import \
                                symmetric_uniform_quantization
                            nbits = self._cfg.quantization.nbits
                            if isinstance(shared_model_para, list):
                                shared_model_para = [
                                    symmetric_uniform_quantization(x, nbits)
                                    for x in shared_model_para
                                ]
                            else:
                                shared_model_para = symmetric_uniform_quantization(
                                    shared_model_para, nbits)
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self._gen_timestamp(
                                init_timestamp=timestamp,
                                instance_number=sample_size),
                            content=(sample_size, shared_model_para)))
            else :
                super().callback_funcs_for_model_para(message)
        else :
            super().callback_funcs_for_model_para(message)
    


def call_my_worker(method):
    
    if method == 'sybil_attack':
        worker_builder = {'client': SybilAttackClient, 'server': SybilAttackServer}
        return worker_builder


register_worker('sybil_attack', call_my_worker)
