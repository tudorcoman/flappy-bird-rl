import random, logging, torch 
import numpy as np 
from typing import Tuple, Any 
from collections import deque 

import torch.nn as nn 
import torch.nn.functional as fct 
import torch.optim as optim
import time 
import scipy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

## Create empty array

def create_empty_array(shape, type=np.float32):
    return np.zeros(shape, dtype=type)

## Create the replay memory 

def create_replay_memory(memory_size, state_size):
    states = create_empty_array((memory_size, state_size))
    actions = create_empty_array((memory_size, 1), type=np.int32)
    rewards = create_empty_array((memory_size, 1))
    advantages = create_empty_array((memory_size, 1))
    returns = create_empty_array((memory_size, 1))
    values = create_empty_array((memory_size, 1))
    logs = create_empty_array((memory_size, 1))
    return states, actions, rewards, advantages, returns, values, logs

def add_to_replay_memory(memory, position, state, action, reward, value, log):
    memory[0][position] = state
    memory[1][position] = action
    memory[2][position] = reward
    memory[4][position] = value
    memory[5][position] = log
    position += 1

def discounted_cumulative_sum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# def compute_advantages_and_returns(rewards, values, last_value, start_index, gamma, lamda, advantages, returns, pointer):
#     rewards_extended = rewards[start_index:pointer] + [last_value]
#     values_extended = values[start_index:pointer] + [last_value]

#     # Calculate deltas
#     deltas = []
#     for i in range(len(rewards_extended) - 1):
#         delta = rewards_extended[i] + gamma * values_extended[i + 1] - values_extended[i]
#         deltas.append(delta)

#     # Calculate advantages
#     advantages[start_index:pointer] = discounted_cumulative_sum(deltas, gamma * lamda)

#     # Calculate returns
#     returns[start_index:pointer] = discounted_cumulative_sum(rewards_extended, gamma)[:-1]

#     # Update pointer
#     start_index = pointer

def compute_advantages_and_returns(start_index, pointer, memory, gamma, lamda, last_value=0):
    rewards = memory[2][start_index:pointer]
    values = memory[4][start_index:pointer]
    last_value = memory[4][pointer-1]
    advantages = memory[3][start_index:pointer]
    returns = memory[5][start_index:pointer]

    rewards_extended = rewards + [last_value]
    values_extended = values + [last_value]

    # Calculate deltas
    deltas = []
    for i in range(len(rewards_extended) - 1):
        delta = rewards_extended[i] + gamma * values_extended[i + 1] - values_extended[i]
        deltas.append(delta)

    # Calculate advantages
    advantages = discounted_cumulative_sum(deltas, gamma * lamda)

    # Calculate returns
    returns = discounted_cumulative_sum(rewards_extended, gamma)[:-1]

    # Update pointer
    start_index = pointer
    
    memory[3][start_index:pointer] = advantages
    memory[5][start_index:pointer] = returns

    return memory

# def get_experiences(start_index, pointer, states, actions, rewards, returns, logs):
    
def get_experiences(start_index, pointer, memory):
    states = memory[0][start_index:pointer]
    actions = memory[1][start_index:pointer]
    rewards = memory[2][start_index:pointer]
    advantages = memory[3][start_index:pointer]
    returns = memory[4][start_index:pointer]
    values = memory[5][start_index:pointer]
    logs = memory[6][start_index:pointer]

    advantages = (advantages - advantages.mean()) / advantages.std()

    return states, actions, rewards, advantages, returns, values, logs

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size 
        self.gamma = 0.99 # discount factor
        self.lamda = 0.97 # GAE parameter
        self.memory = create_replay_memory(100000, state_size)
        self.target_divergence = 0.01
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99995 
        self.policy_learning_rate = 0.0005
        self.value_learning_rate = 0.001
        self.memory_start, self.memory_end = 0, 0

        self.actor, self.critic, self.policy_opt, self.value_opt = self.build_actor_critic(state_size, action_size, self.policy_learning_rate, self.value_learning_rate)

    def build_mlp(state_size, action_size):
        model = nn.Sequential(
            nn.Linear(state_size, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 64, bias=False),
            nn.Tanh(),
            # nn.Linear(hidden_sizes[1], hidden_sizes[2], bias=False),
            # nn.Tanh
            nn.Linear(64, action_size, bias=False)
        )
        return model

    def build_actor_critic(self, state_size, action_size, policy_lr, value_lr):
        # Define the actor model
        actor = self.build_mlp(state_size, action_size)

        # Define the critic model
        critic = self.build_mlp(state_size, action_size)

        # Initialize the policy and the value function optimizers
        policy_opt = optim.Adam(actor.parameters(), lr=policy_lr)
        value_opt = optim.Adam(critic.parameters(), lr=value_lr)

        return actor, critic, policy_opt, value_opt
    
    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        # if random.random() < self.epsilon:
        #     action = np.random.randint(self.action_size)
        # else:
        #     action = self.actor(state).sample()
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # return action

        logits = self.actor(state)
        action = logits.sample()
        log_prob = logits.log_prob(action)
        value = self.critic(state)

        if random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = action.item()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action, log_prob, value
    
    def step(self, state, action, reward, done, log_prob, value):
        state = torch.from_numpy(state).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        log_prob = self.actor(state).log_prob(action)
        value = self.critic(state)

        add_to_replay_memory(self.memory, self.memory_end, state, action, reward, value, log_prob)
        if done:
            compute_advantages_and_returns(self.memory_start, self.memory_end, self.memory, self.gamma, self.lamda)
            experiences = get_experiences(self.memory_start, self.memory_end, self.memory)
            self.learn(experiences)            

    def train_value_function(self, states, returns):
        self.value_opt.zero_grad()
        values = self.critic(states)
        value_loss = fct.mse_loss(values, returns)
        value_loss.backward()
        self.value_opt.step()

    def train_policy(self, states, actions, advantages, old_log_probs):
        self.policy_opt.zero_grad()
        log_probs = self.actor(states).log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.target_divergence, 1 + self.target_divergence)
        policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        policy_loss.backward()
        self.policy_opt.step()

    def learn(self, experiences):
        states, actions, rewards, advantages, returns, values, logs = experiences
        for _ in range(80):
            self.train_policy(states, actions, advantages, logs)

        for _ in range(80):
            self.train_value_function(states, returns)
        

def train_agent(agent, env_no_render, env_render, episodes, early_stop=200):
    rewards = []
    scores = []
    rewards_window = deque(maxlen=100)
    scores_window = deque(maxlen=100)

    for episode in range(1, episodes+1):
        # if episode < 5000 or episode % 200:
        #     env = env_no_render
        # else:
        #     env = env_render
        env = env_no_render
        state, _ = env.reset()
        total_reward = 0 
        done = False 
        score_object = None 

        while not done:
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _, score_object = env.step(action)
            agent.step(state, action, reward, done, log_prob, value)
            state = next_state.copy()
            total_reward += reward
            if done:
                break 

        rewards_window.append(total_reward)
        rewards.append(total_reward)
        scores_window.append(score_object['score'])
        scores.append(score_object['score'])

        print(f"\rEpisode {episode}\tAverage Reward: {np.mean(rewards_window):.2f}\tAverage Score: {np.mean(scores_window):.2f}", end="")
        if episode % 100 == 0:
            print(f"\rEpisode {episode}\tAverage Reward: {np.mean(rewards_window):.2f}\tAverage Score: {np.mean(scores_window):.2f}")
        if np.mean(scores_window) >= early_stop:
            print(f"\nEnvironment solved in {episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            break 

    return scores