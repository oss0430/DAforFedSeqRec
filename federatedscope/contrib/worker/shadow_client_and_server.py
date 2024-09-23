import logging
import copy
import os
import sys

import numpy as np
import pickle
import time

from typing import List

from federatedscope.register import register_worker
from federatedscope.core.message import Message
from federatedscope.core.workers import Server, Client
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_param_dict
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


def get_aug_controller(
    num_augs : int,
    controller_type : str,
    controller_args : any,
    is_weighted : bool = False):
    if is_weighted :
        if controller_type == 'weighted_multi_armed_bandit':
            return WeightedMultiArmedBandit(num_augs,
                                            controller_args.decay_rate,
                                            controller_args.temperature)
    else :
        if controller_type == 'mab':
            return MultiArmedBandit(num_augs,
                                    controller_args.decay_rate,
                                    controller_args.alpha)
        elif controller_type == 'epsilon_greedy':
            return EpsilonGreedyBandit(num_augs,
                                        controller_args.epsilon)
        elif controller_type == 'thompson':
            return ThompsonSamplingGaussianBandit(num_augs)
        elif controller_type == 'sliding_window':
            return SlidingWindowUCB(num_augs, 
                                    controller_args.decay_rate,
                                    controller_args.alpha,
                                    controller_args.window_size)
        elif controller_type == 'discounted_ucb':
            return DiscountedUCB(num_augs, 
                                controller_args.alpha,
                                controller_args.discount_factor)
        elif controller_type == 'mdp':
            return MDPController(num_augs, 
                                controller_args.lr,
                                controller_args.gamma,
                                controller_args.epsilon)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")


class AbstractAugController:
    
    def __init__(self, num_augs):
        self.num_augs = num_augs
        self.total_reward = 0
        self.total_runs = 0
    
    def get_control(self):
        raise NotImplementedError
    
    
    def update(self, reward) -> None:
        self.total_reward += reward
        self.total_runs += 1
        raise NotImplementedError
    
    
    def evaluate(self) -> dict:
        if self.total_runs == 0:
            return {"avg_reward" : 0}
        else :
            return {"avg_reward" : self.total_reward / self.total_runs}


class MultiArmedBandit(AbstractAugController):
    
    def __init__(self, num_augs, decay_rate = 0.99, alpha = 2.0):
        
        self.num_augs = num_augs
        self.decay_rate = decay_rate
        self.alpha = alpha
        
        self.n_pulls = np.zeros(num_augs)
        self.rewards = np.zeros(num_augs)
        self.total_runs = 0
        
        self.cumulative_regret = []
        self.total_reward = 0
        self.action_distribution = np.zeros(num_augs)
        
            
    def get_control(self):
        ucb_values = np.zeros(self.num_augs)
        for arm in range(self.num_augs):
            if self.n_pulls[arm] == 0:
                return arm
            avg_reward = self.rewards[arm] / self.n_pulls[arm]
            exploration = np.sqrt(self.alpha * np.log(self.total_runs) / self.n_pulls[arm])
            ucb_values[arm] = avg_reward + exploration
        return np.argmax(ucb_values)


    def update(self, arm, reward):
        
        self.n_pulls[arm] += 1
        self.rewards[arm] = self.decay_rate * self.rewards[arm] + (1 - self.decay_rate) * reward
        self.total_runs += 1

        ## evaluation purposes
        self.total_reward += reward
        self.action_distribution[arm] += 1
        
        
    # Method to calculate and track cumulative regret
    # NOTE : requires optimal reward to be passed
    def track_cumulative_regret(self, optimal_reward):
        regret = optimal_reward - self.total_reward / self.total_runs if self.total_runs > 0 else 0
        self.cumulative_regret.append(regret)
    
    # Method to calculate average reward
    def calculate_average_reward(self):
        return self.total_reward / self.total_runs if self.total_runs > 0 else 0
    
    # Method to calculate action distribution
    def calculate_action_distribution(self):
        return self.action_distribution / self.total_runs if self.total_runs > 0 else np.zeros(self.num_augs)
    
    
    def evaluate(self) -> dict:
        
        avg_reward = self.calculate_average_reward()
        action_distribution = self.calculate_action_distribution()
        # cumulative_regret = self.cumulative_regret()
        
        return {"avg_reward" : avg_reward, "action_distribution" : action_distribution}
    

class EpsilonGreedyBandit(AbstractAugController):
    
    def __init__(self, num_augs, epsilon=0.1):
        super().__init__(num_augs)
        self.epsilon = epsilon
        self.n_pulls = np.zeros(num_augs)
        self.rewards = np.zeros(num_augs)
        self.total_runs = 0
        self.total_reward = 0
        self.action_distribution = np.zeros(num_augs)

    def get_control(self):
        if np.random.random() < self.epsilon:
            # Exploration: randomly select an arm
            return np.random.randint(self.num_augs)
        else:
            # Exploitation: select the arm with the highest average reward
            avg_rewards = np.divide(self.rewards, self.n_pulls, where=self.n_pulls > 0, out=np.zeros_like(self.rewards))
            return np.argmax(avg_rewards)

    def update(self, arm, reward):
        self.n_pulls[arm] += 1
        self.rewards[arm] += reward
        self.total_runs += 1
        self.total_reward += reward
        self.action_distribution[arm] += 1

    def evaluate(self) -> dict:
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        action_distribution = self.action_distribution / self.total_runs if self.total_runs > 0 else np.zeros(self.num_augs)
        return {"avg_reward": avg_reward, "action_distribution": action_distribution}


class ThompsonSamplingGaussianBandit(AbstractAugController):
    
    def __init__(self, num_augs):
        super().__init__(num_augs)
        # Keep track of the sum of rewards and sum of squared rewards to calculate mean and variance
        self.mean_rewards = np.zeros(num_augs)  # Mean of rewards for each arm
        self.sum_squares = np.zeros(num_augs)   # Sum of squared rewards for variance calculation
        self.n_pulls = np.zeros(num_augs)       # Number of pulls for each arm
        self.total_runs = 0
        self.total_reward = 0
        self.action_distribution = np.zeros(num_augs)
    
    def get_control(self):
        # Sample a reward estimate for each arm from a normal distribution with the estimated mean and variance
        sampled_values = []
        for arm in range(self.num_augs):
            if self.n_pulls[arm] == 0:
                return arm  # If an arm hasn't been pulled yet, explore it first
            variance = self.sum_squares[arm] / self.n_pulls[arm] if self.n_pulls[arm] > 1 else 1.0
            sampled_value = np.random.normal(self.mean_rewards[arm], np.sqrt(variance))
            sampled_values.append(sampled_value)
        
        # Select the arm with the highest sampled value
        return np.argmax(sampled_values)
    
    def update(self, arm, reward):
        self.n_pulls[arm] += 1
        old_mean = self.mean_rewards[arm]
        # Update mean reward using online update formula
        self.mean_rewards[arm] = old_mean + (reward - old_mean) / self.n_pulls[arm]
        # Update the sum of squares for variance calculation
        self.sum_squares[arm] += (reward - old_mean) * (reward - self.mean_rewards[arm])
        
        # Update total reward and other metrics
        self.total_reward += reward
        self.total_runs += 1
        self.action_distribution[arm] += 1

    def evaluate(self) -> dict:
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        action_distribution = self.action_distribution / self.total_runs if self.total_runs > 0 else np.zeros(self.num_augs)
        return {"avg_reward": avg_reward, "action_distribution": action_distribution}


class SlidingWindowUCB(MultiArmedBandit):
    
    def __init__(self, num_augs, decay_rate=0.99, alpha=2.0, window_size=100):
        super().__init__(num_augs, decay_rate, alpha)
        self.window_size = window_size
        self.rewards_history = [[] for _ in range(num_augs)]  # Track recent rewards per arm

    def update(self, arm, reward):
        # Maintain a sliding window for rewards
        if len(self.rewards_history[arm]) >= self.window_size:
            self.rewards_history[arm].pop(0)  # Remove the oldest reward if window is full
        self.rewards_history[arm].append(reward)  # Add the new reward

        # Recalculate mean and rewards using only the sliding window
        recent_rewards = self.rewards_history[arm]
        self.rewards[arm] = np.mean(recent_rewards)
        self.n_pulls[arm] = len(recent_rewards)
        
        self.total_runs += 1
        self.total_reward += reward
        self.action_distribution[arm] += 1
    

class DiscountedUCB(MultiArmedBandit):
    
    def __init__(self, num_augs, alpha=2.0, discount_factor=0.9):
        super().__init__(num_augs, alpha=alpha)
        self.discount_factor = discount_factor
    
    def update(self, arm, reward):
        # Apply a discount to the previous rewards
        self.rewards[arm] = self.discount_factor * self.rewards[arm] + (1 - self.discount_factor) * reward
        self.n_pulls[arm] += 1
        self.total_runs += 1
        self.total_reward += reward
        self.action_distribution[arm] += 1


class LinUCBAugController(AbstractAugController):
    
    def __init__(self, num_augs, context_dim, alpha=1.0):
        super().__init__(num_augs)
        self.context_dim = context_dim
        self.alpha = alpha
        
        # For each augmentation, we maintain a covariance matrix (d x d) and reward vector (d)
        self.A = [np.identity(context_dim) for _ in range(num_augs)]
        self.b = [np.zeros(context_dim) for _ in range(num_augs)]
        self.total_runs = 0
        self.n_pulls = np.zeros(num_augs)
        self.action_distribution = np.zeros(num_augs)

    
    def get_control(self, context):
        """
        Select the augmentation to apply based on the current context.
        """
        ucb_values = np.zeros(self.num_augs)
        
        for aug in range(self.num_augs):
            A_inv = np.linalg.inv(self.A[aug])  # Inverse of covariance matrix
            theta = A_inv @ self.b[aug]         # Estimated reward parameters for the augmentation
            ucb_values[aug] = theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)  # UCB Score

        # Return the augmentation with the highest UCB value
        return np.argmax(ucb_values)


    def update(self, chosen_aug, reward, context):
        """
        Update the model parameters after receiving the reward for the chosen augmentation.
        """
        self.n_pulls[chosen_aug] += 1
        self.total_runs += 1
        self.total_reward += reward
        self.action_distribution[chosen_aug] += 1

        # Update A and b for the chosen augmentation
        self.A[chosen_aug] += np.outer(context, context)
        self.b[chosen_aug] += reward * context


    def evaluate(self):
        """
        Evaluate the average reward and action distribution so far.
        """
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        action_dist = self.action_distribution / self.total_runs if self.total_runs > 0 else np.zeros(self.num_augs)
        
        return {
            "avg_reward": avg_reward,
            "action_distribution": action_dist
        }


class MDPController(AbstractAugController):
    
    def __init__(self, num_augs, lr=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(num_augs)
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-values for each state-action pair
        self.Q = np.zeros((num_augs, num_augs))  # Q-table, states are arms, actions are arms
        self.state = None  # Current state (arm pulled last)
        self.total_reward = 0
        self.total_runs = 0
        self.action_distribution = np.zeros(num_augs)

    def get_control(self):
        """ Choose an action using epsilon-greedy strategy """
        if self.state is None:  # If no prior state, choose randomly
            return np.random.randint(self.num_augs)
        
        # Epsilon-greedy policy for exploration vs exploitation
        if np.random.random() < self.epsilon:
            # Exploration: Randomly select an action
            return np.random.randint(self.num_augs)
        else:
            # Exploitation: Select the action with the highest Q-value for the current state
            return np.argmax(self.Q[self.state])

    def update(self, action, reward):
        """ Update the Q-values based on the action and reward received """
        if self.state is None:
            # First update, set the current state to the action taken
            self.state = action
            return
        
        # Get the best possible Q-value for the new state
        best_next_q = np.max(self.Q[action])
        
        # Q-learning update rule
        self.Q[self.state, action] = self.Q[self.state, action] + \
                                     self.lr * (reward + self.gamma * best_next_q - self.Q[self.state, action])
        
        # Update internal state and statistics
        self.state = action
        self.total_runs += 1
        self.total_reward += reward
        self.action_distribution[action] += 1

    def evaluate(self):
        """ Evaluate the current performance """
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        action_distribution = self.action_distribution / self.total_runs if self.total_runs > 0 else np.zeros(self.num_augs)
        return {"avg_reward": avg_reward, "action_distribution": action_distribution}



class ShadowServerWithAugSelection(ShadowServer):
    
    
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)
        self._get_controller()
        
    
    def _get_controller(self):
        controller = get_aug_controller(num_augs = self._cfg.data.augmentation_args.aug_types_count,
                                        controller_type = self._cfg.data.augmentation_controller.type,
                                        controller_args = self._cfg.data.augmentation_controller.args, 
                                        is_weighted = False)
        self.aug_controller = controller
        
    
    def get_aug_control(self) :
        """
        Get the augmentation control
        """
        arm = self.aug_controller.get_control()
        logger.info(f"Augmentation Controll : {arm}")
        return arm
    
    
    def eval_controller(self) -> None:
        """
        Evaluate the MAB
        """
        controller_evaluation = self.aug_controller.evaluate()
        eval_str =  "'Controller Evaluation' : {"
        for key, value in controller_evaluation.items():
            eval_str += f"'{key}' : {value}, "
        eval_str = eval_str[:-2] + "}"
        logger.info(eval_str)
    
    
    def update_aug_controller(self, round_losses : List[float]) -> None :
        """
        Evaluate the training loss and determine
        augmentation should enforce generalization or personalization
        """
        # Evaluate the training loss for each client
        ## default rewards are the negative of the losses
        reward = -np.mean(round_losses)
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

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    avg_train_loss, train_data_size, msg_content = train_msg_buffer[client_id]
                    msg_list.append((train_data_size, msg_content))
                    avg_train_loss_list.append(avg_train_loss)
                else:
                    ## Added AvgTrainloss
                    avg_train_loss, train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))
                    avg_train_loss_list.append(avg_train_loss)
                    
                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    avg_train_loss, train_data_size, msg_content = content
                    msg_list.append((train_data_size, msg_content))
                    avg_train_loss_list.append(avg_train_loss)
                else:
                    avg_train_loss, train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))
                    avg_train_loss_list.append(avg_train_loss)
                    
                staleness.append((client_id, self.state - state))

            ## Aug Controller Update
            self.update_aug_controller(avg_train_loss_list)
            
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


class WeightedMultiArmedBandit(AbstractAugController):
    
    def __init__(self, num_augs, decay_rate = 0.99, temperature = 1.0):
        
        self.num_augs = num_augs
        self.decay_rate = decay_rate
        self.temperature = temperature
        
        self.weight_records = []
        self.rewards = np.zeros(num_augs)
        
            
    def get_control(self):
        """
        Get the weight of each arm
        """
        exp_value = np.exp(self.rewards / self.temperature)
        weights = exp_value / np.sum(exp_value)
        return weights
    

    def update(self, weight, reward):
        
        self.weight_records.append(weight)
        self.rewards = self.decay_rate * self.rewards + (1 - self.decay_rate) * reward

    
    def evaluate(self) -> dict:
        
        avg_reward = self.calculate_average_reward()
        action_distribution = self.calculate_action_distribution()
        # cumulative_regret = self.cumulative_regret()
        
        return {"Average_Reward" : avg_reward, "Action_Distribution" : action_distribution}
    

class ShadowServerWithWeightedAugSelection(ShadowServerWithAugSelection):

    def _get_controller(self):
        controller = get_aug_controller(num_augs = self._cfg.data.augmentation_args.aug_types_count,
                                        controller_type = self._cfg.data.augmentation_controller.type,
                                        controller_args = self._cfg.data.augmentation_controller.args, 
                                        is_weighted = True)
        self.aug_controller = controller
        
    
    def update_aug_controller(self, round_losses : List[float]) -> None:
        """
        To minimize the train losses received from the clients
        Find the best weights for the augmentations
        BlackBox Optimization
        """
        userd_weights = self.aug_controller.get_control()
        
        ## average the round losses and give it a negative sign
        ## as the reward is the negative of the loss
        reward = -np.mean(round_losses)
        self.aug_controller.update(userd_weights, reward) 
    
    
    def get_aug_control(self) :
        """
        Get the augmentation control
        """
        weight = self.aug_controller.get_control()
        logger.info(f"Augmentation Controll : {weight}")
        return weight


class AutoAug(AbstractAugController):
    
    def __init__(self, num_augs, decay_rate = 0.99, temperature = 1.0):
        
        self.num_augs = num_augs
        self.decay_rate = decay_rate
        self.temperature = temperature
        
        self.weight_records = []
        self.rewards = np.zeros(num_augs)

import torch
class AugSearcher(torch.nn.Module):
    
    """
    inspired by the 
    AutoAugment : Learning Augmentation Strategies from Data (Cubuk et al., 2019)
    Updates the augmentation policy based on, validation results
    
    One Layer LSTM with Softmax Activation
    """
    
    def __init__(self, num_arms, hidden_size = 128):
        super(AugSearcher, self).__init__()
        self.num_arms = num_arms
        self.lstm = torch.nn.LSTM(1, hidden_size, batch_first = True)
        self.fc1 = torch.nn.Linear(hidden_size, num_arms)
        
        
    def forward(self, x, hidden = None):
        """
        Forward pass for the AugSearcher model.
        
        Args:
        - x (torch.Tensor): Input reward sequence of shape (batch_size, seq_len, 1)
        - hidden (tuple): Hidden states for LSTM (optional)
        
        Returns:
        - probs (torch.Tensor): Output probabilities for choosing each augmentation
        - hidden (tuple): Updated hidden state
        """
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get the hidden state from the last time step (sequence length is assumed > 0)
        lstm_out_last = lstm_out[:, -1, :]
        
        # Pass through the fully connected layer to get raw logits for each augmentation
        logits = self.fc(lstm_out_last)
        
        # Apply softmax to get probabilities over augmentations (arms)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        return probs, hidden
    

class AutoAugController(AbstractAugController):
    
    def __init__(self, num_augs, device = 'cpu'):
        self.num_augs = num_augs
        self.policy = AugSearcher(num_augs)
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters())

        self.total_reward = 0
        self.total_runs = 0
        self.reward_history = []
        
        self.last_control = None
        self.control_history = []
        
        self.last_hidden = None
        self.hidden_history = []
        
        
    def get_control(self):
        control, hidden = self.policy(torch.tensor([self.total_reward], device = self.device).unsqueeze(0), self.last_hidden)
        
        self.last_control = control
        self.control_history.append(control)
        
        self.last_hidden = hidden
        self.hidden_history.append(hidden)        
        
        return control
    
    
    def update(self, reward : np.ndarray) -> None:
        ## Update every time when a new reward is received
        
        self.total_reward += reward
        self.total_runs += 1
        
        self.reward_history.append(reward)
        
        last_reward = torch.tensor([reward], device = self.device).unsqueeze(0)
        self.optimizer.zero_grad()
        control, hidden = self.policy(last_reward, self.last_hidden)
        
        ## TODO ! : Implement the loss function
        
        
        


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
        ## previous pervion where we measured the cosine similarity
        #selected_aug_idx = self._select_training_set(flag)
        #selected_train_data = self._from_aug_idx_get_train_data(selected_aug_idx)
        #augmentation_type = "personalized" if flag else "generalized"
        #logger.info(f"Selected the most {augmentation_type} augmentation set for training")
        
        ## flag is aug_idx
        selected_train_data = self._from_aug_idx_get_train_data(flag)
        
        return selected_train_data
    
    
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
            selected_train_data = self._from_flag_get_train_data(flag) 
            
            original_train_data = self.trainer.ctx.data['train']
            temp_data = self.trainer.ctx.data
            temp_data['train'] = selected_train_data
            self.update_shadow_client(temp_data, self.ID)
            
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
                            content=(avg_train_loss, sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))
            ## restore the original train data
            temp_data['train'] = original_train_data
            self.update_shadow_client(temp_data, self.ID)
        else:
            round = message.state
            sender = message.sender
            timestamp = message.timestamp
            content = message.content

            ## refer to the callback_funcs_for_model_para in the server
            flag, content = content 
            # dequantization
            selected_train_data = self._from_flag_get_train_data(flag) 
            
            original_train_data = self.trainer.ctx.data['train']
            temp_data = self.trainer.ctx.data
            temp_data['train'] = selected_train_data
            self.update_shadow_client(temp_data, self.ID)
            
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
                avg_train_loss, sample_size, model_para_all, results = \
                    None, 0, self.trainer.get_model_para(), {}
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
                avg_train_loss = results["train_avg_loss"]
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
            temp_data['train'] = original_train_data
            self.update_shadow_client(temp_data, self.ID)
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
                            content=(avg_train_loss, sample_size, shared_model_para)))
            self.ID = original_id



class ShadowClientWithAugSelectionValidVer(ShadowClientWithAugSelection):
    """
    Instead of communicating train losses
    It communicates Valid Losses
    """
    
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
            selected_train_data = self._from_flag_get_train_data(flag) 
            
            original_train_data = self.trainer.ctx.data['train']
            temp_data = self.trainer.ctx.data
            temp_data['train'] = selected_train_data
            self.update_shadow_client(temp_data, self.ID)
            
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
                            content=(avg_train_loss, sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))
            ## restore the original train data
            temp_data['train'] = original_train_data
            self.update_shadow_client(temp_data, self.ID)
        else:
            round = message.state
            sender = message.sender
            timestamp = message.timestamp
            content = message.content

            ## refer to the callback_funcs_for_model_para in the server
            flag, content = content 
            # dequantization
            selected_train_data = self._from_flag_get_train_data(flag) 
            
            original_train_data = self.trainer.ctx.data['train']
            temp_data = self.trainer.ctx.data
            temp_data['train'] = selected_train_data
            self.update_shadow_client(temp_data, self.ID)
            
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
                avg_train_loss, sample_size, model_para_all, results = \
                    None, 0, self.trainer.get_model_para(), {}
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
                avg_train_loss = results["val_avg_loss"]
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
            temp_data['train'] = original_train_data
            self.update_shadow_client(temp_data, self.ID)
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
                            content=(avg_train_loss, sample_size, shared_model_para)))
            self.ID = original_id


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


def call_shadow_with_MAP_VALID(method):
    if method == 'shadow_with_map_valid':
        worker_builder = {'server': ShadowServerWithAugSelection, 'client': ShadowClientWithAugSelectionValidVer}
        return worker_builder


register_worker('shadow', call_shadow_server)
register_worker('shadow_with_aug_selection', call_shadow_with_aug_selection)
register_worker('shadow_with_map_valid', call_shadow_with_MAP_VALID)
register_worker('shadow_with_weighted_aug_selection', call_shadow_with_weighted_aug_selection)