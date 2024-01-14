import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from flappy_bird_game import FlappyBirdGame
import random, pygame
from threading import Thread

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = [] 
        self.gamma = 0.95
        self.epsilon = 1.0 
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
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def execute_episode(agent, game, e):
    state = np.reshape(game.get_state(), [1, state_size])
    for _ in range(1000):  # Maximum time steps in an episode
        action = agent.choose_action(state)
        next_state, reward, done = game.step([action])
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{total_episodes}, score: {game.score}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

if __name__ == "__main__":
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    batch_size = 32
    total_episodes = 50

    for e in range(total_episodes):
        game = FlappyBirdGame(headless=False, use_keyboard=False)
        thread = Thread(target=execute_episode, args=(agent, game, e))
        game.run_game_thread(thread)
    