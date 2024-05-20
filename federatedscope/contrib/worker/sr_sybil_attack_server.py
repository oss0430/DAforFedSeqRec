from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
import logging
from copy import deepcopy
import pickle
import os
import torch
## NOTE:
# To use SybilAttackServer. 
# Write in config file, "attack : attack_method : sybil_attack"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SybilAttackServer(Server):
    
    """
    SybilAttackServer is the hivemind of attacking nodes.
    This allows attacking nodes to share information and coordinate their attacks.
    For simulation they are only allowed to collect information from the attacking nodes.
    
    
    """

    def __init__(self,
           ID=-1,
           state=0,
           config=None,
           data=None,
           model=None,
           client_num=5,
           total_round_num=10,
           device='cpu',
           strategy=None,
           unseen_clients_id=None,
           **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)
        self.attacking_nodes = []
        self.attacking_nodes_info = []
        self.attacking_nodes_data = []
        self.attacking_nodes_models = []
        self.attacking_nodes_gradients = []
        self.attacking_nodes_metrics = []
        self.attacking_nodes_losses
        self.max_iter = 200
        self.criterion = torch.nn.CrossEntropyLoss()
    
    
    
    
    def reconsturct_sequence_embedding_via_label(self, y, current_model, previous_model, epoch, learning_rate, loss_fn) -> torch.Tensor:
        """
        reconstruct gaussian noise to match the right x_embedding for label y
        """
        batch_size = y.size(0)
        x_embedding = torch.normal(0, 1, size=(batch_size, self.model.embedding_dim))
        
        optimizer = torch.optim.Adam([x_embedding], lr=learning_rate)
        
        training_rate = epoch * learning_rate
        mean_gradient = (current_model.weight.grad - previous_model.weight.grad) / training_rate
        
        for iter in range(self.max_iter):
            current_model.zero_grad()
            y_pred = current_model.forward_by_embedding(x_embedding)
            loss = loss_fn(y_pred, y)
            
            dy_dx = torch.autograd(loss, current_model.parameters(), create_graph=True)
            
            grad_diff = 0
            for dy_dxi in dy_dx:
                grad_diff += ((dy_dxi - mean_gradient) ** 2).sum()
            
            grad_diff.backward()
            
            
        return x_embedding
    
    
    def save_generated_embedding_as_file(self, x_embedding, file_path):
        """
        Save generated embedding as file
        """
        torch.save(x_embedding, file_path)
    
    
    
    def _get_y_by_number_of_attacking_nodes(self) -> torch.Tensor:
        """
            number_of_ys * 1 (id)
        """
        attacking_node_num = len(self._cfg.attack.attacker_id)
        number_of_ys = attacking_node_num 
        
        item_num = self._cfg.data.item_num
        y = torch.randint(0, item_num + 1, (number_of_ys, 1))
        
        return y
        
    
    def _reconstruct_sequence_embedding(self):
        """
        The attacker will generate embedding that was infer from previous and past model
        """
        current_models = self.model ##list of models
        previous_models = self.previous_model
        local_update_steps = self._cfg.train.local_update_steps
        learning_rate = self._cfg.train.optimizer.lr
        criterion_type = self._cfg.criterion.type ## this is a type decleration
        loss_fn = self._get_criterion(criterion_type)
        
        for current_model, previous_model in zip(current_models, previous_models):
            y = self._get_y_by_number_of_attacking_nodes()
            x_embedding = self.reconsturct_sequence_embedding_via_label(y, current_model, previous_model, local_update_steps, learning_rate, loss_fn)
            self.attacking_nodes_info.append(x_embedding)
    
    
    def eval_reconstruction(self) :
        # Globally Evaluate the reconsturction by
        # comapring true sequence embedding and constructed sequence embedding
        ## save sequence representation with corresponding label
        
        output_dir = self._cfg.outdir
        
        target_data_split_name = 'train'
        
        label_via_embedding = {}
        
        if self._cfg.federate.make_global_eval:
            
            for i in range(self.model_num):
                trainer = self.trainers[i]
                
                label_via_input_emb = trainer.get_true_input_emb(target_data_split_name = target_data_split_name) ## Dict {'label' : ['input_emb', ...]}
                
                ys = label_via_input_emb.keys()
                
                label_via_generated_item_emb = self.reconsturct_sequence_embedding_via_label(
                    y = ys,
                    current_model = self.model,
                    previous_model = self.previous_model,
                    epoch = self._cfg.train.local_update_steps,
                    learning_rate = self._cfg.train.optimizer.lr,
                    loss_fn = self.criterion
                )  ## Dict {'label' : ['generated_emb']} only one
                
                cosine_sim_score = {'cosine_score' : []}
                
                for label in label_via_generated_item_emb.keys():
                    input_embs = label_via_input_emb[label] ## list of input embeddings
                    generated_emb = label_via_generated_item_emb[label][0] ## generated embedding
                    
                    for input_emb in input_embs:
                        ## Calculate Cosine Similairty between input_emb and generated_emb
                        cosine_sim_score['cosine_score'].append(torch.nn.functional.cosine_similarity(input_emb, generated_emb))
                    
                ## output as file
                original_embedding_save_path = os.path.join(output_dir, 'original_embedding.pickle')
                generated_embedding_save_path = os.path.join(output_dir, 'generated_embedding.pickle')
                cosine_similarity_save_path = os.path.join(output_dir, 'cosine_similarity.pickle')
                
                ## Save as pickle
                with open(original_embedding_save_path, 'wb') as f:
                    pickle.dump(label_via_input_emb, f)
                    
                with open(generated_embedding_save_path, 'wb') as f:
                    pickle.dump(label_via_generated_item_emb, f)
                    
                with open(cosine_similarity_save_path, 'wb') as f:
                    pickle.dump(cosine_sim_score, f)
                    
            
    
    def _perform_federated_aggregation(self):
        # before aggregation save as previous model
        self.previous_model = deepcopy(self.model)
        return super()._perform_federated_aggregation()
    
        
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
                ## TODO: Implement Sybil Attack HERE
                # aggregated_num = self._performe_sybil_attack()
                logger.info(f'Server: Perfroming Sybil Attack')
                self.state += 1
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

def call_my_worker(method):
    
    if method == 'sybil_attack':
        worker_builder = {'client': Client, 'server': SybilAttackServer}
        return worker_builder


register_worker('sybil_attack', call_my_worker)
