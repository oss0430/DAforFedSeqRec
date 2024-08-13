import logging
import copy
import os
import sys

import numpy as np
import pickle
import time

from federatedscope.register import register_worker
from federatedscope.core.message import Message
from federatedscope.core.workers import Server, Client
from federatedscope.core.auxiliaries.sampler_builder import get_sampler

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
        
        
    
        
    
    
    
def call_shadow_server(method):
    if method == 'shadow':
        worker_builder = {'server': ShadowServer, 'client': ShadowClient}
        return worker_builder
    
register_worker('shadow', call_shadow_server)
