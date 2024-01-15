import numpy as np
import random
from flappy_bird_game import FlappyBirdGame

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((167, 50, 50, 2))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def update_q_table(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        # Update the Q-value for the state-action pair
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

        # If the game is done, reset the epsilon to explore more in the next episode
        if done:
            self.epsilon = self.epsilon * 0.99  # Decay epsilon


if __name__ == "__main__":
    # Set the parameters
    total_episodes = 1000
    learning_rate = 0.1
    max_steps = 100
    gamma = 0.99

    # Exploration parameters
    epsilon = 1.0  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.01  # Exponential decay rate for exploration prob

    # Create the agent
    agent = QLearningAgent(20000, 2, alpha=learning_rate, gamma=gamma, epsilon=epsilon)
    
    # Run the game
    for episode in range(total_episodes):
        env = FlappyBirdGame(headless=False, use_keyboard=False)
        state = env.get_state()
        step = 0
        done = False

        for step in range(max_steps):
            print(state)
            action = agent.choose_action(state)
            next_state, reward, done = env.step([action])
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        print(f"episode: {episode}/{total_episodes}, score: {env.score}")