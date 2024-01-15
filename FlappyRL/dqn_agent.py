import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from flappy_bird_game import FlappyBirdGame
import random, pygame
from threading import Thread
from collections import deque 

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(state, verbose = 0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Avoid training if we don't have enough memory

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])

        states = np.array([t[0].reshape((self.state_size,)) for t in minibatch])
        next_states = np.array([t[3].reshape((self.state_size,)) for t in minibatch])

        # Predict Q-values in batch for efficiency
        q_values = self.model.predict(states, verbose=0)
        q_values_next = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(q_values_next[i])
            q_values[i][action] = target

        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def execute_episode(agent, game, e):
    state = np.reshape(game.get_state(), [1, state_size])
    for _ in range(1000):  # Maximum time steps in an episode
        action = agent.choose_action(state)
        game.push_action(action)
        # for _ in range(20):
        #     if not game.running:
        #         break
        #     game.run_game_once()
        game.run_game_once()
        #next_state, reward, done = game.step([action])
        next_state, reward, done = game.get_state(), game.calculate_reward(), not game.running
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{total_episodes}, score: {game.score}")
            break
        #if len(agent.memory) > batch_size:
        #   agent.replay(batch_size)

def continue_pygame_loop():
    pygame.mainloop(0.1)
    yield

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.reduce_sum(tf.random.normal([10, 10])))

    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    batch_size = 8
    total_episodes = 50

    for e in range(total_episodes):
        game = FlappyBirdGame(headless=True, use_keyboard=False)
        thread = Thread(target=game.run_game, args=(agent, game, e))
        #game.run_game_thread(thread)
        execute_episode(agent, game, e)
    