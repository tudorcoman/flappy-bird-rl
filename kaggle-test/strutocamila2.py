import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
import flappy_bird_gymnasium
import gymnasium
from collections import deque 
import random 


CSV_PATH = 'flappy_bird.csv'
WEIGHTS = 'weights.h5'

df = pd.read_csv(CSV_PATH)

# Drop all frames that led the bird to crash
for index, row in df.iterrows():
    if row['reward'] < 0:
        df.drop(df[(df['score'] == row['score']) & (df['game'] == row['game']) & (df.index <= index)].index,
                inplace=True)

# We don't need the action, score, game and reward. Also, 'player_s_vertical_velocity' and 'player_s_rotation' make it get "addicted" to the last action taken, which makes the bird to crash all the time
data = df.drop(columns=['action',
                        'player_s_vertical_velocity',
                        'player_s_rotation',
                        'score',
                        'game',
                        'reward'])

y_data = df['action']

X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.2)

model = Sequential()
model.add(Dense(32, input_shape=(X_train.shape[1:]), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model.load_weights(WEIGHTS)
    
print('Done')
env = gymnasium.make("FlappyBird-v0")

replay_buffer = deque(maxlen=2000)
gamma = 0.95  # Discount rate
epsilon = 0.3  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
target_update_frequency = 10

def add_to_replay_buffer(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def train_policy_network():
    minibatch = random.sample(replay_buffer, batch_size)
    
best_score = 0
game = 0
while game < 100:
    state, _ = env.reset()
    done = False
    game += 1
    game_score = 0

    while not done:
        state = state[:-2]
        state = np.expand_dims(state, axis=0)
        action = model.predict(state, verbose=0)
        action = action[0][0]
        action = 1 if action >= 0.2 else 0

        state, _, done, _, info = env.step(action)
        if info['score'] > game_score:
            game_score = info['score']
        if game_score > best_score:
            best_score = game_score        
            print('New best score:', best_score)
        action = None

        if done:
            print(f'Game {game} finished with score {game_score}')
            break


