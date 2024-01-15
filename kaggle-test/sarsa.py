
import flappy_bird_gymnasium 
import numpy as np 
import gymnasium as gym 

env = gym.make("FlappyBird-v0", render_mode="human")

state_size = [800, 576, 3]
q_table = np.random.uniform(low=-2, high=0, size=(state_size + [env.action_space.n]))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
total_episodes = 20

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore action space
    return np.argmax(q_table[state])     # Exploit learned values

# 3. Define the SARSA Algorithm
def update_q_table(state, action, reward, new_state, new_action):
    predict = q_table[state][action]
    target = reward + gamma * q_table[new_state][new_action]
    q_table[state][action] += alpha * (target - predict)

# 4. Training Loop
for episode in range(total_episodes):
    state, _ = env.reset()
    state = state[:-2]
    state = np.expand_dims(state, axis=0)
    #state = discretize_state(state)  # You need to define this function
    action = choose_action(state)

    done = False
    while not done:
        new_state, reward, done, _, score = env.step(action)
        new_state = new_state[:-2]
        new_state = np.expand_dims(new_state, axis=0)
        #new_state = discretize_state(new_state)
        new_action = choose_action(new_state)

        update_q_table(state, action, reward, new_state, new_action)

        state = new_state
        action = new_action

    if episode % 100 == 0:
        print(f"Episode: {episode}")

# 5. Testing
# Test the performance of the agent after training

env.close()