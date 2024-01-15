# Import necessary libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import flappy_bird_gymnasium
import gymnasium as gym
from sklearn.model_selection import train_test_split

# Set up the environment
env = gym.make("FlappyBird-v0", render_mode="human")
state_size = env.observation_space.shape[0] - 2
action_size = env.action_space.n

# Initialize replay memory
replay_buffer = deque(maxlen=2000)

# Define hyperparameters
gamma = 0.95  # Discount rate
epsilon = 0.1  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
target_update_frequency = 10

### 
CSV_PATH = 'flappy_bird.csv'
WEIGHTS = 'weights.h5'
df = pd.read_csv(CSV_PATH)

# Drop all frames that led the bird to crash
for index, row in df.iterrows():
    if row['reward'] < 0:
        df.drop(df[(df['score'] == row['score']) & (df['game'] == row['game']) & (df.index <= index)].index,
                inplace=True)

# We don't need the action, score, game and reward. Also, 'player_s_vertical_velocity' and 'player_s_rotation' make it get "addicted" to the last action taken, which makes the bird to crash all the time
data = df.drop(columns=['player_s_vertical_velocity',
                        'player_s_rotation',
                        'score', 'game'])

new_data = data.drop(columns=['action', 'reward'])

# Initialize DQN model
model = Sequential()
model.add(Dense(32, input_shape=(new_data.shape[1:]), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#model.load_weights(WEIGHTS)
    
print('Done')

def get_replay_buffer_row(index, row, data):
    # separate action and reward
    action = row['action']
    reward = row['reward']
    new_row = row.copy()
    new_row.drop(['action', 'reward'], inplace=True)
    state = new_row.values

    new_new_row = data.iloc[index + 1].drop(['action', 'reward']) if index < len(data) - 1 else None
    next_state = new_new_row.values if index < len(data) - 1 else np.zeros(state_size)
    done = True if reward < 0 else False

    return state, action, reward, next_state, done

# Define function to add experience to replay buffer
def add_to_replay_buffer(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def initialize_replay_buffer(data):
    # Go through the data and add each frame to the replay buffer
    for index, row in data.iterrows():
        # separate action and reward
        state, action, reward, next_state, done = get_replay_buffer_row(index, row, data)
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        add_to_replay_buffer(state, action, reward, next_state, done)

initialize_replay_buffer(data)

# Define training function
def train_policy_network():
    minibatch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma * model.predict(next_state, verbose=0)[0][0])
        target_f = model.predict(state, verbose=0)
        if action == 1:
            target_f[0][0] = target
        else:
            target_f[0][0] = 1 - target
        model.fit(state, target_f, epochs=1, verbose=0)

# Main training loop
total_episodes = 1000
for episode in range(total_episodes):
    state, _ = env.reset()
    #state = np.reshape(state, [1, state_size])
    done = False

    state = state[:-2]
    state = np.expand_dims(state, axis=0)
    print(episode)

    threshold = 0.2
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = model.predict(state, verbose=0)[0][0]
            action = 1 if action >= 0.6 else 0

        next_state, reward, done, _, score = env.step(action)
        next_state = next_state[:-2]
        next_state = np.expand_dims(next_state, axis=0)

        # Add experience to replay buffer
        add_to_replay_buffer(state, action, reward, next_state, done)

        state = next_state

        if done:
            print(f"episode: {episode}/{total_episodes}, score: {score}")
            break

    train_policy_network()

    # Update epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

# Save the final model
model.save("final_model.h5")