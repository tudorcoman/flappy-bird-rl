import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym 

# Actor-Critic Network
class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Common layers
        self.common_layers = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='linear')
        ])

        # Actor layers
        self.policy_logits = layers.Dense(action_size)

        # Critic layers
        self.values = layers.Dense(1)

    def call(self, inputs):
        x = self.common_layers(inputs)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values

# PPO Parameters
learning_rate = 0.001
gamma = 0.99
clip_ratio = 0.2
update_steps = 2048
batch_size = 64
num_epochs = 10
max_episodes = 20

# Environment setup
env = gym.make('FlappyBird-v0')  # Replace with the correct environment name
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the Actor-Critic model
model = ActorCritic(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def compute_loss(old_policy_logits, advantages, actions, rewards, values, clip_ratio):
    # Convert everything to tensors
    advantages, actions, rewards, old_policy_logits = [tf.convert_to_tensor(x, dtype=tf.float32) for x in [advantages, actions, rewards, old_policy_logits]]

    # Compute the ratio (pi_theta / pi_theta_old)
    policy_logits, new_values = model(states)
    new_policy_actions = tf.gather_nd(policy_logits, tf.expand_dims(actions, axis=-1), batch_dims=1)
    old_policy_actions = tf.gather_nd(old_policy_logits, tf.expand_dims(actions, axis=-1), batch_dims=1)
    ratio = tf.exp(tf.math.log(new_policy_actions) - tf.math.log(old_policy_actions))

    # Clipped surrogate function
    clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
    clipped_surrogate = tf.minimum(ratio * advantages, clipped_ratio * advantages)

    # Critic loss (value loss)
    critic_loss = tf.reduce_mean(tf.square(rewards - new_values))

    # Total loss (negative because we want to maximize the objective)
    loss = -tf.reduce_mean(clipped_surrogate) + critic_loss

    return loss

@tf.function
def train_step(states, actions, rewards, old_policy_logits, advantages, values):
    with tf.GradientTape() as tape:
        loss = compute_loss(old_policy_logits, advantages, actions, rewards, values)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Main Training Loop
for episode in range(max_episodes):
    # Reset environment
    state = env.reset()
    episode_reward = 0

    # Arrays to keep track of states, actions, rewards
    states, actions, rewards, old_policy_logits, advantages, values = [], [], [], [], [], []

    for _ in range(update_steps):
        state = state[:-2]
        state = np.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        policy_logits, value = model.predict(state)

        # Sample action
        action = tf.random.categorical(policy_logits, 1)[0, 0]
        next_state, reward, done, _, score = env.step(action.numpy())

        # Store state, action, reward, and logits
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_policy_logits.append(policy_logits)
        values.append(value)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            print(f"Episode: {episode}, Reward: {episode_reward}, Score: {score}")
            episode_reward = 0

    # After update_steps, perform PPO update
    states, actions, rewards, old_policy_logits, advantages, values = map(np.array, [states, actions, rewards, old_policy_logits, advantages, values])

    # Compute advantages and targets
    # Add your logic to compute advantages and targets

    inds = np.arange(update_steps)
    for _ in range(num_epochs):
        np.random.shuffle(inds)
        for start in range(0, update_steps, batch_size):
            end = start + batch_size
            mbinds = inds[start:end]
            batch_states, batch_actions, batch_rewards, batch_old_policy_logits, batch_advantages, batch_values = map(lambda arr: arr[mbinds], [states, actions, rewards, old_policy_logits, advantages, values])
        train_step(batch_states, batch_actions, batch_rewards, batch_old_policy_logits, batch_advantages, batch_values)

    env.close()