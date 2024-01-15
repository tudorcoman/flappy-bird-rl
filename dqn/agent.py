import random, logging, torch 
import numpy as np 
from typing import Tuple, Any 
from collections import deque 

import torch.nn as nn 
import torch.nn.functional as fct 
import torch.optim as optim
import time 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Create the Q-Network 
def create_q_network(state_size, action_size, seed):
    #torch.cuda.manual_seed(seed)
    #print(torch.seed())
    #print(torch.seed())
    model = nn.Sequential(
        nn.Linear(state_size, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, action_size, bias=False)
    )
    return model

## Create the replay memory 

def create_replay_memory(memory_size):
    return deque(maxlen=memory_size)

def add_to_replay_memory(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_from_replay_memory(memory, batch_size):
    sample = random.sample(memory, batch_size)

    states = []
    actions = []
    rewards = []
    next_states = [] 
    dones = [] 

    for s in sample:
        states.append(s[0])
        actions.append(s[1])
        rewards.append(s[2])
        next_states.append(s[3])
        dones.append(s[4])

    torch_states = torch.from_numpy(np.vstack(states)).float().to(device)
    torch_actions = torch.from_numpy(np.vstack(actions)).long().to(device)
    torch_rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
    torch_next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
    torch_dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

    return (torch_states, torch_actions, torch_rewards, torch_next_states, torch_dones)

class DQNAgent: 
    def __init__(self, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size 
        self.learning_rate = 0.0005
        self.gamma = 0.99 # discount factor
        self.tau = 0.001 # interpolation parameter
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99995 

        self.losses = []
        self.local_qnet = create_q_network(state_size, action_size, 1621229999698273942).to(device)
        self.target_qnet = create_q_network(state_size, action_size, 1473805256038137402).to(device)
        self.optimizer = optim.Adam(self.local_qnet.parameters(), lr=self.learning_rate)

        self.batch_size = 64
        self.memory = create_replay_memory(100000)

        self.time_step = 0 

    def step(self, state, action, reward, next_state, done, episode): 
        self.memory.append((state, action, reward, next_state, done))

        self.time_step = (self.time_step + 1) % 4
        if self.time_step == 0: 
            if len(self.memory) > self.batch_size:
                experiences = sample_from_replay_memory(self.memory, self.batch_size)
                self.learn(experiences)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_qnet.eval()
        with torch.no_grad():
            action_values = self.local_qnet(state)
        self.local_qnet.train()

        # Action selection (epsilon-greedy policy)
        if random.random() > self.epsilon:
            action = np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            action = random.choice(np.arange(self.action_size))

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.local_qnet(states).gather(1, actions)

        loss = fct.mse_loss(Q_expected, Q_targets)
        self.losses.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(self.target_qnet.parameters(), self.local_qnet.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


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
            action = agent.choose_action(state)
            next_state, reward, done, _, score_object = env.step(action)
            agent.step(state, action, reward, next_state, done, episode)
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