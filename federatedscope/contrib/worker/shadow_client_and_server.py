import logging
import copy
import os
import sys

import numpy as np
import pickle
import time
from torch.utils.data import Subset

from typing import List

from federatedscope.augmentation_control.controller import get_aug_controller
from federatedscope.register import register_worker
from federatedscope.core.message import Message
from federatedscope.core.workers import Server, Client
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
from federatedscope.core.data.base_data import ClientData
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ShadowServer(Server):
    
    
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)
        
        ## Change Sampler to assign Shadow Clients
        assert self._cfg.federate.mode == 'standalone', \
            "Standalone mode is not enabled try : \
                federate.mode = 'standalone'"
        
        assert self._cfg.federate.standalone_args.use_shadow == True, \
            "Shadow Server is not enabled try : \
                federate.standalone_args.use_shadow = True"
        
        ## client_num is the proxy client number
        ## shadow_client_num is the real client number
        self.client_num = self._cfg.federate.client_num
        self.shadow_client_num = self._cfg.federate.standalone_args.shadow_client_num
        if self._cfg.federate.sampler in ['uniform']:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num = self._cfg.federate.standalone_args.shadow_client_num,
                client_info= None)
        else:
            self.sampler = None
        
        self.seen_data_indices = kwargs.get('seen_data_indices', [])
        self.unseen_data_indices = kwargs.get('unseen_data_indices', [])
        if len(self.unseen_data_indices) > 0 and self._cfg.federate.make_global_eval:
            self._init_seen_and_unseen_data()
    
    
    def _init_seen_and_unseen_data(self):
        
        unseen_data = ClientData(
            self._cfg,
            train=Subset(self.data['train'].dataset,
                         self.unseen_data_indices['train']),
            test=Subset(self.data['test'].dataset,
                        self.unseen_data_indices['test']),
            val=Subset(self.data['val'].dataset,
                           self.unseen_data_indices['val'])    
        )
        seen_data = ClientData(
            self._cfg,
            train=Subset(self.data['train'].dataset,
                            self.seen_data_indices['train']),
            test=Subset(self.data['test'].dataset,
                            self.seen_data_indices['test']),
            val=Subset(self.data['val'].dataset,
                            self.seen_data_indices['val'])
        )
        self.seen_and_unseen_data = {
            'seen': seen_data,
            'unseen': unseen_data
        }
        
        self.seen_and_unseen_trainer = {
            'seen' : get_trainer(
                model= self.models[0],
                data= self.seen_and_unseen_data['seen'],
                device= self.device,
                config= self._cfg,
                only_for_eval=True,
                monitor=self._monitor
            ),
            'unseen' : get_trainer(
                model= self.models[0],
                data= self.seen_and_unseen_data['unseen'],
                device= self.device,
                config= self._cfg,
                only_for_eval=True,
                monitor=self._monitor
            )
        }


    def eval_seen_and_unseen(self):
        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics_seen = self.seen_and_unseen_trainer['seen'].evaluate(
                target_data_split_name = split
            )
            eval_metric_unseen = self.seen_and_unseen_trainer['unseen'].evaluate(
                target_data_split_name = split
            )
            for key in eval_metrics_seen.keys():
                metrics.update({f'{key}_seen': eval_metrics_seen[key]})
                metrics.update({f'{key}_unseen': eval_metric_unseen[key]})
            ## add the merged_metrics
            ## only avialble for sequential recommendation
            metrics_seen_count = metrics.get(f'{split}_total_seen', 0)
            metrics_unseen_count = metrics.get(f'{split}_total_unseen', 0)
            metrics_total_count = metrics_seen_count + metrics_unseen_count
            for metrics_buildin in self._monitor.metric_calculator.eval_metric.keys():
                sum_metrics = metrics[f'{split}_{metrics_buildin}_seen'] * metrics_seen_count + \
                    metrics[f'{split}_{metrics_buildin}_unseen'] * metrics_unseen_count
                metrics[f'{split}_{metrics_buildin}'] = \
                    sum_metrics / metrics_total_count                    
            metrics[f'{split}_total'] = metrics_total_count
        formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Server #',
                forms=self._cfg.eval.report,
                return_raw=self._cfg.federate.make_global_eval)
        self._monitor.update_best_result(
            self.best_results,
            formatted_eval_res['Results_raw'],
            results_type="server_global_eval")
        self.history_results = merge_dict_of_results(
            self.history_results, formatted_eval_res)
        self._monitor.save_formatted_results(formatted_eval_res)
        logger.info(formatted_eval_res)
        
        
    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """
        split_eval_seen_and_unseen = True
        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models;
            # for other cases such as ensemble, override the eval function
            if split_eval_seen_and_unseen :
                  self.eval_seen_and_unseen()
                  self.check_and_save()
            else :
                for i in range(self.model_num):
                    trainer = self.trainers[i]
                    # Preform evaluation in server
                    metrics = {}
                    for split in self._cfg.eval.split:
                        eval_metrics = trainer.evaluate(
                            target_data_split_name=split)
                        metrics.update(**eval_metrics)
                    formatted_eval_res = self._monitor.format_eval_res(
                        metrics,
                        rnd=self.state,
                        role='Server #',
                        forms=self._cfg.eval.report,
                        return_raw=self._cfg.federate.make_global_eval)
                    self._monitor.update_best_result(
                        self.best_results,
                        formatted_eval_res['Results_raw'],
                        results_type="server_global_eval")
                    self.history_results = merge_dict_of_results(
                        self.history_results, formatted_eval_res)
                    self._monitor.save_formatted_results(formatted_eval_res)
                    logger.info(formatted_eval_res)
                self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate',
                                      sample_client_num=-1, ##broadcast to all clients when evaluating locally
                                      filter_unseen_clients=False)
    
    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """

        if self.check_client_join_in():
            if self._cfg.federate.use_ss or self._cfg.vertical.use:
                self.broadcast_client_address()

            # get sampler
            if 'client_resource' in self._cfg.federate.join_in_info:
                client_resource = [
                    self.join_in_info[client_index]['client_resource']
                    for client_index in np.arange(1, self.client_num + 1)
                ]
            else:
                if self._cfg.backend == 'torch':
                    model_size = sys.getsizeof(pickle.dumps(
                        self.models[0])) / 1024.0 * 8.
                else:
                    # TODO: calculate model size for TF Model
                    model_size = 1.0
                    logger.warning(f'The calculation of model size in backend:'
                                   f'{self._cfg.backend} is not provided.')

                client_resource = [
                    model_size / float(x['communication']) +
                    float(x['computation']) / 1000.
                    for x in self.client_resource_info
                ] if self.client_resource_info is not None else None

            if self.sampler is None:
                self.sampler = get_sampler(
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.shadow_client_num,
                    client_info=client_resource)

            # change the deadline if the asyn.aggregator is `time up`
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                self.deadline_for_cur_round = self.cur_timestamp + \
                                               self._cfg.asyn.time_budget

            # start feature engineering
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))
    
    def check_client_join_in(self):
        if len(self._cfg.federate.join_in_info) != 0:
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num
    
    
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
            min_received_num = self.shadow_client_num
            #min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
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
    
        
    def callback_funcs_for_metrics(self, message: Message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """

        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()

        self.msg_buffer['eval'][rnd][sender] = content

        return self.check_and_move_on(check_eval_result=True)


class ShadowServerWithAugSelection(ShadowServer):
    
    
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)
        self.define_comm_content()
        self._get_controller()
    
    
    def define_comm_content(self) -> None:
        """
        Aug Controller Might require different Client & Server Feedbacks
        Here we define the content of communications
        Sent in call_back_for_model_para
        """
        self.server_side_content_keys = ["aug_control", "sample_size", "model_para"]
        self.client_side_content_keys = ["reward", "sample_size", "model_para"]
        
    
    def _handle_addtional_feedback_content_for_aug_controller(self, msg_buffer) -> None:
        
        """
        Handles Dynamic Content Size for the Aggregation
        Requires the content keys to be defined first.
        """
         
        content_length = len(self.client_side_content_keys)
        ## sample_size and model_para always appears
        additional_content_num = content_length - 2 
        
        ## define dict for additional content placeholders
        ## utilize only first (additional_content_num) elements
        additional_contents = {key : [] for key in self.client_side_content_keys[:additional_content_num]}
        
        for client_id in msg_buffer.keys():
            client_additional_contents = list(msg_buffer[client_id])[:additional_content_num]
            for idx in range(0,len(client_additional_contents)):
                content_key = self.client_side_content_keys[idx]
                additional_contents[content_key].append(client_additional_contents[idx])
        
        self.additional_contents = additional_contents
                

    def _get_controller(self):
        controller = get_aug_controller(num_augs = self._cfg.data.augmentation_args.aug_types_count,
                                        controller_type = self._cfg.data.augmentation_controller.type,
                                        controller_args = self._cfg.data.augmentation_controller.args, 
                                        is_weighted = False)
        self.aug_controller = controller
        
    
    def get_aug_control(self) :
        arm = self.aug_controller.get_control()
        if type(arm) == np.array : arm_str = f"Augmentation Control : {arm.tolist()}"
        else : arm_str = f"Augmentation Control : {arm}"
        
        logger.info("{"+arm_str+ "}")
        return arm
    
    
    def eval_controller(self) -> None:
        """
        Evaluate the MAB
        """
        controller_evaluation = self.aug_controller.evaluate()
        eval_str =  "{'Controller Evaluation' : {"
        for key, value in controller_evaluation.items():
            eval_str += f"'{key}' : {value}, "
        eval_str = eval_str[:-2] + "}}"
        logger.info(eval_str)
    
    
    def update_aug_controller(self) -> None :
        """
        Evaluate the training loss and determine
        augmentation should enforce generalization or personalization
        """
        # Evaluate the training loss for each client
        ## default rewards are the negative of the losses
        
        reward = self.additional_contents['reward']
        
        reward = -np.mean(reward)
        self.aug_controller.update(self.aug_controller.get_control(), reward)
        
        ## Evaluate the MAB
        controller_evaluation = self.aug_controller.evaluate()
    
    
    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            avg_train_loss_list = list()
            msg_list = list()
            staleness = list()

            self._handle_addtional_feedback_content_for_aug_controller(train_msg_buffer)
            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    essentail_contents = list(train_msg_buffer[client_id])[-2:]
                    train_data_size, msg_content = essentail_contents[0], essentail_contents[1]
                    
                    msg_list.append((train_data_size, msg_content))
                    #avg_train_loss_list.append(avg_train_loss)
                else:
                    essentail_contents = list(train_msg_buffer[client_id])[-2:]
                    train_data_size, model_para_multiple = \
                        essentail_contents[0], essentail_contents[1]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))
                    #avg_train_loss_list.append(avg_train_loss)
                    
                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            #self._handle_addtional_feedback_content_for_aug_controller(self.staled_msg_buffer)
            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    essentail_contents = list(content)[-2:]
                    train_data_size, msg_content = essentail_contents[0], essentail_contents[1]
                    msg_list.append((train_data_size, msg_content))
                    #avg_train_loss_list.append(avg_train_loss)
                else:
                    essentail_contents = list(content)[-2:]
                    train_data_size, msg_content = essentail_contents[0], essentail_contents[1]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))
                    #avg_train_loss_list.append(avg_train_loss)
                    
                staleness.append((client_id, self.state - state))

            ## Aug Controller Update
            self.update_aug_controller()
            
            ## Evaluate the controller
            self.eval_controller()
            
            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return aggregated_num

 
    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        ## same as super but added flag at message content
        """
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        # quantization
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        ## Added Flag in the content for augmentation
        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=(self.get_aug_control(), model_para)))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')
    

class ShadowServerWithWeightedAugSelection(ShadowServerWithAugSelection):

    def _get_controller(self):
        controller = get_aug_controller(num_augs = self._cfg.data.augmentation_args.aug_types_count,
                                        controller_type = self._cfg.data.augmentation_controller.type,
                                        controller_args = self._cfg.data.augmentation_controller.args, 
                                        is_weighted = True)
        self.aug_controller = controller
        
    
    def update_aug_controller(self) -> None:
        """
        To minimize the train losses received from the clients
        Find the best weights for the augmentations
        BlackBox Optimization
        """
        reward = self.additional_contents['reward']
        
        used_weights = self.aug_controller.get_control()
        
        ## average the round losses and give it a negative sign
        ## as the reward is the negative of the loss
        reward = -np.mean(reward)
        self.aug_controller.update(used_weights, reward) 
    
    
    def get_aug_control(self) :
        """
        Get the augmentation control
        """
        weight = self.aug_controller.get_control()
        logger.info(f"Augmentation Controll : {weight}")
        return weight



class ShadowClient(Client):
    
    def __init__(self,
                 ID=-1,
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
        ## In proxy clients there is no need for unseen clients
        ## relies on server side unseen client masking
        is_unseen_client = False
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)

        ## Change Sampler to assign Shadow Clients
        assert self._cfg.federate.mode == 'standalone', \
            "Standalone mode is not enabled try : \
                federate.mode = 'standalone'"
        
        assert self._cfg.federate.standalone_args.use_shadow == True, \
            "Shadow Server is not enabled try : \
                federate.standalone_args.use_shadow = True"
        
        ## shadow_client_id is the real client id
        self.shadow_client_id = self.ID
        
    
    def update_shadow_client(self, data, ID):
        self.shadow_client_id = ID
        self.data = data
        self.trainer.ctx.data = data
        new_init_dict = self.trainer.parse_data(data)
        self.trainer.ctx.merge_from_dict(new_init_dict)

    
    """
    replace the sender address for each send
    except for joining in. with
    simple trick of switching client id temporary
    """
    def callback_funcs_for_model_para(self, message: Message):
        original_id = self.ID
        self.ID = self.shadow_client_id
        super().callback_funcs_for_model_para(message)
        self.ID = original_id
        
    
    def callback_funcs_for_evaluate(self, message: Message):
        original_id = self.ID
        self.ID = self.shadow_client_id
        super().callback_funcs_for_evaluate(message)
        self.ID = original_id
        
        
    
class ShadowClientWithAugSelection(ShadowClient):
    
    
    def _from_df_get_user_full_sequences(self, user_id_value, df):
        
        user_df = df[df[self._cfg.data.user_column] == user_id_value]
        ## group by augmentation_column
        
        full_inputs = []
        import torch
        max_sequence_length = self._cfg.data.max_sequence_length
        padding_value = self._cfg.data.padding_value
        ## convert each sequence to tensor
        for aug_idx, group in user_df.groupby(self._cfg.data.augmentation_args.augmentation_column):
            item_seq = group[self._cfg.data.item_column].values
            item_seq_len = np.array([len(item_seq)])
            
            if max_sequence_length is not None :
                sequence_length = len(item_seq)
                if  sequence_length > max_sequence_length :
                    start_index = sequence_length - max_sequence_length
                    
                    item_seq = item_seq[start_index:]
                    item_seq_len = np.array([max_sequence_length])
                else :
                    item_seq = np.pad(
                        item_seq,
                        (0, max_sequence_length - sequence_length),
                        constant_values = padding_value
                    )
            full_inputs.append(
                {'item_seq' : torch.tensor(item_seq, dtype=torch.int64),
                 'item_seq_len' : torch.tensor(item_seq_len, dtype=torch.int64)})
        
        return full_inputs

    
    def _from_aug_idx_get_train_data(self, aug_idx):
        full_trainset = self.trainer.ctx.data['train'].dataset.dataset
        selected_aug_set = full_trainset.augmentation_datasets[aug_idx]
        ## client id starts from 1
        ## self.ID is a user value self.ID - 1 is the index
        subset_range = selected_aug_set._from_user_idx_get_user_subset_range(self.ID - 1)

        import torch
        subset = torch.utils.data.Subset(selected_aug_set, subset_range)
        new_train_data = torch.utils.data.DataLoader(subset,
                                                     batch_size=self._cfg.dataloader.batch_size,
                                                     shuffle=True)
        
        return new_train_data
    
    
    def _from_flag_get_train_data(self, flag):
        selected_train_data = self._from_aug_idx_get_train_data(flag)
        
        return selected_train_data
    
    
    def handle_aug_controls(self, message : Message) -> None:
        ## refer to the callback_funcs_for_model_para in the server
        flag, content = message.content 
        # dequantization
        selected_train_data = self._from_flag_get_train_data(flag) 
        
        self.original_train_data = self.trainer.ctx.data['train']
        data = self.trainer.ctx.data
        data['train'] = selected_train_data
        self.update_shadow_client(data, self.ID)
        
    
    def restore_original_train_data(self) -> None:
        data = self.trainer.ctx.data
        data['train'] = self.original_train_data
        self.update_shadow_client(data, self.ID)
    
    
    def get_reward_from_results(self, results) -> any:
        """
        Get the reward from the results
        """
        try :
            reward_key = self._cfg.data.augmentation_controller.args.reward_type
        except KeyError:
            raise KeyError("reward_type is not defined in the config")
            #reward_key = "train_avg_loss"
        try :
            reward = results[reward_key]
        except KeyError:
            reward = 0
        return reward
    
    
    def formulate_sending_content(self, received_message, shared_model_para, sample_size, results) -> set:
        
        reward = self.get_reward_from_results(results)
        return (reward, sample_size, shared_model_para)
        
    
    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        original_id = self.ID
        self.ID = self.shadow_client_id
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content, timestamp = message.state, message.content, \
                                        message.timestamp
            ## refer to the callback_funcs_for_model_para in the server
            flag, content = content 
            self.handle_aug_controls(message)
            
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should " \
                        "a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(None, sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))
            ## restore the original train data
            self.restore_original_train_data()
        else:
            round = message.state
            sender = message.sender
            timestamp = message.timestamp
            content = message.content

            ## refer to the callback_funcs_for_model_para in the server
            flag, content = content 
            self.handle_aug_controls(message)
            
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
                sample_size, model_para_all, results = self.trainer.train()
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

            ## restore the original train data
            self.restore_original_train_data()
            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss:
                raise NotImplementedError(
                    "Secret sharing is not supported in the "
                    "ShadowClientWithAugSelection")
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
                            content=self.formulate_sending_content(
                                received_message = message,
                                shared_model_para=shared_model_para,
                                sample_size = sample_size,
                                results=results)))
            self.ID = original_id



class ShadowServerWithFedEx(ShadowServerWithAugSelection):
    
    """
    A weighted sampling control
    Also Receives Client Side Aug Configurations
    """
    def define_comm_content(self) -> None:
        """
        added client side content keys
        """
        self.server_side_content_keys = ["aug_control", "sample_size", "model_para"]
        self.client_side_content_keys = ["aug_control", "reward", "sample_size", "model_para"]
    
    def update_aug_controller(self) -> None:
        reward = self.additional_contents['reward']
        client_side_aug_control = self.additional_contents['aug_control']
        
        ## concat the controls
        control_vector = np.stack(client_side_aug_control)
        reward = -np.array(reward)

        self.aug_controller.update(reward, control_vector)
        
        ## Evaluate the MAB
        controller_evaluation = self.aug_controller.evaluate()


class ShadowServerWithWeightedFedEx(ShadowServerWithFedEx):
    
    def _get_controller(self):
        controller = get_aug_controller(num_augs = self._cfg.data.augmentation_args.aug_types_count,
                                        controller_type = self._cfg.data.augmentation_controller.type,
                                        controller_args = self._cfg.data.augmentation_controller.args, 
                                        is_weighted = True)
        self.aug_controller = controller


class ShadowClientWithWeightedAugSelection(ShadowClientWithAugSelection):
    
    def _get_int_allocation(self, weights : np.ndarray, total_item : int):
        raw_allocation = weights * total_item
        
        int_allocation = np.floor(raw_allocation).astype(int)
        fractional_part = raw_allocation - int_allocation
        
        remaining_items = total_item - np.sum(int_allocation)
        
        sorted_indices = np.argsort(-fractional_part)
        for i in range(int(remaining_items)):
            int_allocation[sorted_indices[i]] += 1
        
        return int_allocation
    
    
    def _from_flag_get_train_data(self, flag):
        ## flag is weight
        full_aug_dataset = self.trainer.ctx.data['train'].dataset.dataset
        
        batch_size = self._cfg.dataloader.batch_size
        batch_num = self._cfg.train.local_update_steps
        
        sample_size = batch_size * batch_num
        int_allocation = self._get_int_allocation(flag, sample_size)
        sampled_indices = []
        
        for aug_idx, allocated in enumerate(int_allocation):
            user_aug_sub_range = full_aug_dataset._from_user_idx_and_augmentation_type_idx_get_subset_range(self.ID - 1, aug_idx)
            sampled_sub_range = np.random.choice(user_aug_sub_range, allocated)
            sampled_indices += list(sampled_sub_range)
            
        ## create a new subset
        import torch
        subset = torch.utils.data.Subset(full_aug_dataset, sampled_indices)
        new_train_data = torch.utils.data.DataLoader(subset,
                                                     batch_size=self._cfg.dataloader.batch_size,
                                                     shuffle=True)
        
        return new_train_data


class ShadowClientWithFedEx(ShadowClientWithAugSelection):
    """
    The clients perturbs the sampling distribution
    Reports the perturbed distribution and valid Loss
    """ 
    
    def set_perturb_given_flag(self, flag):
        self.perturbed_flag = flag
        
    
    
    def handle_aug_controls(self, message: Message) -> None:
        ## refer to the callback_funcs_for_model_para in the server
        flag, content = message.content 
        
        self.set_perturb_given_flag(flag)
        selected_train_data = self._from_flag_get_train_data(self.perturbed_flag) 
        
        self.original_train_data = self.trainer.ctx.data['train']
        data = self.trainer.ctx.data
        data['train'] = selected_train_data
        self.update_shadow_client(data, self.ID)
    
    
    def formulate_sending_content(self, received_message, shared_model_para, sample_size, results) -> set:
        ## content includes val_loss and perturebed flag.
        reward = self.get_reward_from_results(results)
        return (self.perturbed_flag, reward, sample_size, shared_model_para)


class ShadowClientWithFedExWeighted(ShadowClientWithWeightedAugSelection):
    
    
    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None, is_unseen_client=False, *args, **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)
        self.max_addative_noise = self._cfg.data.augmentation_controller.args.max_addative_noise
        
    
    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    
    def set_perturb_given_flag(self, flag):
        ## addative noise
        random_noise = np.random.normal(0, self.max_addative_noise, len(flag))
        perturbed_flag = self._softmax(flag + random_noise)
        
        self.perturbed_flag = perturbed_flag
        
    
    def handle_aug_controls(self, message: Message) -> None:
        ## refer to the callback_funcs_for_model_para in the server
        flag, content = message.content 
        
        self.set_perturb_given_flag(flag)
        selected_train_data = self._from_flag_get_train_data(self.perturbed_flag) 
        
        self.original_train_data = self.trainer.ctx.data['train']
        data = self.trainer.ctx.data
        data['train'] = selected_train_data
        self.update_shadow_client(data, self.ID)
    
    
    def formulate_sending_content(self, received_message, shared_model_para, sample_size, results) -> set:
        ## content includes val_loss and perturebed flag.
        reward = self.get_reward_from_results(results)
        return (self.perturbed_flag, reward, sample_size, shared_model_para)


class ShadowClientWithDirichletSampling(ShadowClientWithFedExWeighted):
    
    
    def set_perturb_given_flag(self, flag):
        self.perturbed_flag = np.random.dirichlet(flag)

    
def call_shadow_server(method):
    if method == 'shadow':
        worker_builder = {'server': ShadowServer, 'client': ShadowClient}
        return worker_builder

def call_shadow_with_aug_selection(method):
    if method == 'shadow_with_aug_selection':
        worker_builder = {'server': ShadowServerWithAugSelection, 'client': ShadowClientWithAugSelection}
        return worker_builder

def call_shadow_with_weighted_aug_selection(method):
    if method == 'shadow_with_weighted_aug_selection':
        worker_builder = {'server': ShadowServerWithWeightedAugSelection, 'client': ShadowClientWithWeightedAugSelection}
        return worker_builder

def call_shadow_with_fedex(method):
    if method == 'shadow_with_fedex':
        worker_builder = {'server': ShadowServerWithFedEx, 'client': ShadowClientWithFedEx}
        return worker_builder

def call_shadow_with_fedex_weighted(method):
    if method == 'shadow_with_fedex_weighted':
        worker_builder = {'server': ShadowServerWithWeightedFedEx, 'client': ShadowClientWithFedExWeighted}
        return worker_builder

def call_shadow_with_weighted_dirichlet(method):
    if method == 'shadow_with_weighted_dirichlet':
        worker_builder = {'server': ShadowServerWithWeightedFedEx, 'client': ShadowClientWithDirichletSampling}
        return worker_builder


register_worker('shadow', call_shadow_server)
register_worker('shadow_with_aug_selection', call_shadow_with_aug_selection)
register_worker('shadow_with_weighted_aug_selection', call_shadow_with_weighted_aug_selection)
register_worker('shadow_with_fedex', call_shadow_with_fedex)
register_worker('shadow_with_fedex_weighted', call_shadow_with_fedex_weighted)
register_worker('shadow_with_weighted_dirichlet', call_shadow_with_weighted_dirichlet)