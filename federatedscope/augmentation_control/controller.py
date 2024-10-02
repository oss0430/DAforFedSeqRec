import numpy as np
import torch

from typing import Callable
from scipy.stats import norm
from scipy.optimize import minimize

def get_aug_controller(
    num_augs : int,
    controller_type : str,
    controller_args : any,
    is_weighted : bool = False):
    if is_weighted :
        if controller_type == "rs":
            return RSController(num_augs, controller_args.lr, controller_args.decay_rate)
        elif controller_type == "bayesian_optimization":
            return BayesianOptimizationController(num_augs,
                                                  controller_args.max_addative_noise,
                                                  controller_args.perturbation_num,
                                                  controller_args.window_size)
        elif controller_type == "cumulative_reward":
            return CumulativeRewardController(num_augs)
        elif controller_type == "dirichlet_prior":
            return DirichletPriorController(num_augs, controller_args.lr, controller_args.decay_rate)
                                            
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
        
        
class AbstractWeightedAugController(AbstractAugController):
    
    def __init__(self, num_augs):
        self.num_augs = num_augs
        self.total_reward = 0
        self.total_runs = 0
        self.weight_records = []
        self.rewards = np.zeros(num_augs)
    
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    
    def get_control(self) -> np.array:
        raise NotImplementedError
    
    
    def update(self, reward, weight) -> None:
        raise NotImplementedError
    
    
    def evaluate(self) -> dict:
        raise NotImplementedError
    


class RSController(AbstractWeightedAugController):
    """
    Random Search Controller
    """
    def __init__(self, num_augs, lr = 0.1, decay_rate = 1):
        super().__init__(num_augs)
        self.weights = np.ones(num_augs) / num_augs
        self.original_lr = lr
        self.current_lr = lr
        self.decay_rate = decay_rate
        
        self.total_reward = 0
        self.total_runs = 0

        
    def get_control(self):
        return self.weights
    
    def update(self, reward : np.array, weight : np.array) -> None:
        ## Given Weight are randomly sampled (perturbed from the original weights)
        # reward 1d, weight 2d
        self.total_reward += np.mean(reward)
        self.total_runs += 1
        
        ## select the best descent direction
        best_direction_idx = np.argmax(reward)
        
        best_direction = weight[best_direction_idx] - self.weights
        best_direction = best_direction / np.linalg.norm(best_direction)
        
        self.weights = self.weights + self.current_lr * best_direction
        #self.weights = self._softmax(new_weights)
        self.current_lr = self.current_lr * self.decay_rate
    
    
    def evaluate(self) -> dict:
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        return {"avg_reward": avg_reward}
    
    
    
def squared_exponential_kernel(x1, x2, l=1.0, sigma_f=1.0):
    """
    Squared Exponential Kernel
    """
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
    


class GaussianProcess:
    
    def __init__(self, 
                 kernel : Callable = None,
                 noise : float = 1e-3,
                 window_size : int = 100):
        self.kernel = kernel
        self.noise = noise
        self.window_size = window_size
        self.X = None
        self.y = None
    
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        
    def fit_new(self, X_new, y_new):
        if self.X is None or self.y is None:
            self.X = X_new
            self.y = y_new
        else:
            self.X = np.vstack((self.X, X_new))
            self.y = np.concatenate((self.y, y_new))

        if len(self.X) > self.window_size:
            self.X = self.X[-self.window_size:]
            self.y = self.y[-self.window_size:]
        
    
    def predict(self, X_new):
        if self.X is None or self.y is None:
            raise ValueError("Model not fitted yet")
        
        K = self.kernel(self.X, self.X) + self.noise * np.eye(len(self.X))
        K_new = self.kernel(self.X, X_new)
        K_inv = np.linalg.inv(K)
        
        mu = K_new.T.dot(K_inv).dot(self.y)
        sigma = self.kernel(X_new, X_new) - K_new.T.dot(K_inv).dot(K_new)
        
        return mu.flatten(), np.diag(sigma)
        
        
    def expected_improvement(self, X, y_best):
        mu, sigma = self.predict(X)
        sigma = np.sqrt(sigma)
        
        with np.errstate(divide='ignore'):
            Z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
        

class BayesianOptimizationController(AbstractWeightedAugController):
    
    def __init__(self, 
                 num_augs : int,
                 max_addative_noise : float = 0.1,
                 perturbation_num : int = 100,
                 window_size : int = 100):
        super().__init__(num_augs)
        self.weights = np.ones(num_augs) / num_augs
        self.max_addative_noise = max_addative_noise
        self.perturbation_num = perturbation_num
        
        self.total_reward = 0
        self.total_runs = 0
        
        self.best_weight = None
        self.best_reward = -np.inf
        self.gp = GaussianProcess(kernel = squared_exponential_kernel,
                                  window_size=window_size)
        
    
    def get_control(self):
        return self.weights
        
    
    def update(self, reward, weight):
        self.total_reward += np.mean(reward)
        self.total_runs += 1
        self.best_reward = np.max([np.mean(reward), self.best_reward])
        
        self.gp.fit_new(weight, reward)
        ## perturb addativly the weights and return the
        ## best performing weight
        ## Too slow after several concatenation, use sliding window
        
        ## note that rewards all all negative
        ## thus the maximum reward is 0
        
        addative_noise = np.random.normal(0, self.max_addative_noise, (self.num_augs, self.perturbation_num))
        ## (num_augs, perturbation_num) + (num_augs, 1)
        perturbed_weights = self.weights + addative_noise.T
        
        ## simplex
        perturbed_weights = np.clip(perturbed_weights, 1e-8, 1)
        perturbed_weights = perturbed_weights / np.sum(perturbed_weights, axis = 1, keepdims = True)

        ## perturb and add the original weights
        perturbed_weights = np.vstack((perturbed_weights, self.weights))
        
        ## Check EI for each perturbed weights and select the best
        ei = self.gp.expected_improvement(perturbed_weights, self.best_reward)
        ## since all rewards are negative, the best reward is 0
        best_idx = np.argmax(ei)
        
        self.weights = perturbed_weights[best_idx]
        
        
    def evaluate(self) -> dict:
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        return {"avg_reward": avg_reward}


class CumulativeRewardController(AbstractWeightedAugController):
        
    def __init__(self, num_augs, discount_factor=0.9):
        super().__init__(num_augs)
        self.weights = np.ones(num_augs) / num_augs
        self.discount_factor = discount_factor
        
        self.total_reward = 0
        self.total_runs = 0
        self.cumulative_reward = np.zeros(num_augs)
    
    
    def get_control(self):
        return self.weights
    
    
    def update(self, reward, weight):
        self.total_reward += np.mean(reward)
        self.total_runs += 1
        
        new_cumulative_reward = np.dot(reward, weight)
        
        self.cumulative_reward += self.discount_factor * self.cumulative_reward +\
                                    (1 - self.discount_factor) * new_cumulative_reward
        
        ## min max normalization than softmax
        ## error case gives nan
        self.weights = self._softmax((self.cumulative_reward - np.min(self.cumulative_reward)) / \
                        (np.max(self.cumulative_reward) - np.min(self.cumulative_reward)))
        
    
    def evaluate(self) -> dict:
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        return {"avg_reward": avg_reward}


class DirichletPriorController(AbstractWeightedAugController):
    """
    Optimize to the best performing prior
    Bayesian Manner
    """
    def __init__(self, num_augs, lr=0.1, decay_rate=1):
        self.num_augs = num_augs
        self.prior = np.ones(num_augs)  # Initial prior is uniform
        self.original_lr = lr
        self.current_lr = lr
        self.decay_rate = decay_rate
        self.total_reward = 0
        self.total_runs = 0

    
    def get_control(self) -> np.array:
        return self.prior
    
    
    def update(self, reward, weight_vector):
        """
        Update the prior distribution based on the reward and weight vector.
        Args:
            reward: The observed reward for the sampled weight_vector.
            weight_vector: The weight vector sampled from the prior.
        """
        # Update the total reward and number of runs
        self.total_reward += np.mean(reward)
        self.total_runs += 1

        # Normalize the weight vector (in case it isn't already)
        weight_vector = weight_vector / np.linalg.norm(weight_vector)

        # Scale the weight vector by its corresponding reward (Softmax-weighted update)
        reward_importance = np.exp(reward) / np.sum(np.exp(reward))

        # Update the prior with a learning rate and decay mechanism
        new_prior = self.prior + self.current_lr * np.dot(reward_importance, weight_vector)
        self.prior = self._normalize_prior(new_prior)

        # Decay the learning rate
        self.current_lr *= self.decay_rate


    def _normalize_prior(self, prior):
        """
        Normalize the prior so that it can still represent a valid Dirichlet distribution.
        """
        return np.clip(prior, 1e-8, None)  # Avoid zero probabilities by clipping values

        
    def evaluate(self):
        """
        Evaluate the controller's performance.
        Returns a dictionary with the average reward.
        """
        avg_reward = self.total_reward / self.total_runs if self.total_runs > 0 else 0
        return {"avg_reward": avg_reward, "current_prior": self.prior.tolist()}
