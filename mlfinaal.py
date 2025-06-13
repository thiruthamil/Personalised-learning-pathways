import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# Custom Environment for Demo
class CustomEnv:
    def __init__(self):
        self.state_size = 8
        self.action_size = 3
        self.reset()

    def reset(self):
        self.state = np.random.rand(self.state_size)
        return self.state

    def step(self, action):
        next_state = np.random.rand(self.state_size)
        reward = 1 if action == np.argmax(self.state) else -1
        done = random.random() > 0.8  # End episode randomly
        return next_state, reward, done

# Double DQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """Update target network weights."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0  # Skip replay if insufficient data

        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                action_next = np.argmax(self.model.predict(next_state, verbose=0)[0])
                target += self.gamma * self.target_model.predict(next_state, verbose=0)[0][action_next]

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            total_loss += history.history['loss'][0]

        self.update_target_model()
        return total_loss / self.batch_size

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_ddqn(env, agent, episodes=300):
    rewards_log, epsilon_log, loss_log, q_values_log = [], [], [], []

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        total_reward = 0

        for time in range(100):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        avg_loss = agent.replay()
        agent.update_epsilon()

        # Logging
        rewards_log.append(total_reward)
        epsilon_log.append(agent.epsilon)
        loss_log.append(avg_loss)
        q_values_log.append(np.mean(agent.model.predict(state, verbose=0)))

        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward}, "
              f"Epsilon: {agent.epsilon:.2f}, Loss: {avg_loss:.4f}")

    return rewards_log, epsilon_log, loss_log, q_values_log

def plot_results(rewards, epsilon, loss, q_values):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Rewards Plot
    axs[0, 0].plot(rewards)
    axs[0, 0].set_title("Total Rewards per Episode")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")

    # Epsilon Decay Plot
    axs[0, 1].plot(epsilon)
    axs[0, 1].set_title("Epsilon Decay")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Epsilon Value")

    # Loss Plot
    axs[1, 0].plot(loss)
    axs[1, 0].set_title("Loss per Episode")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Loss")

    # Q-Values Plot
    axs[1, 1].plot(q_values)
    axs[1, 1].set_title("Average Q-Values per Episode")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Q-Value")

    plt.tight_layout()
    plt.show()

# Main Execution
env = CustomEnv()
agent = DDQNAgent(env.state_size, env.action_size)

rewards, epsilon, loss, q_values = train_ddqn(env, agent)
plot_results(rewards, epsilon, loss, q_values)
