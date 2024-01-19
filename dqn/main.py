import numpy as np
import gymnasium as gym 
import flappy_bird_gymnasium 

from agent import DQNAgent, train_agent

env_1 = gym.make("FlappyBird-v0", render_mode="human")
env_2 = gym.make("FlappyBird-v0")
env = env_2 

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print(f"Action size: {action_size}")
print(f"State size: {state_size}")

env.reset()
agent = DQNAgent(state_size, action_size)

train_agent(agent, env_2, env_1, 25000)

