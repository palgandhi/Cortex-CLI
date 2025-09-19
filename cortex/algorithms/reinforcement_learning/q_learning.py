# cortex/algorithms/reinforcement_learning/q_learning.py
import numpy as np
import gymnasium as gym
from ..base import BaseModel

class QLearningAgent(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Q-Learning Agent"
        self.model = None
        self.hyperparameters = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon_greedy': 0.1,
            'episodes': 5000
        }
        self.q_table = None

    def train(self, env):
        """Trains the agent in the given environment."""
        state_space_size = env.observation_space.n
        action_space_size = env.action_space.n
        
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        for episode in range(self.hyperparameters['episodes']):
            state, info = env.reset()
            terminated, truncated = False, False
            while not terminated and not truncated:
                if np.random.uniform(0, 1) < self.hyperparameters['epsilon_greedy']:
                    action = env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state, :])  # Exploit

                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Q-Learning update rule
                self.q_table[state, action] = self.q_table[state, action] + self.hyperparameters['learning_rate'] * (
                    reward + self.hyperparameters['discount_factor'] * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
                )
                state = next_state
        
        self.model = self.q_table
        print(f"Training complete after {self.hyperparameters['episodes']} episodes.")

    def evaluate(self, env):
        """Evaluates the agent's performance by running a few test episodes."""
        total_rewards = 0
        test_episodes = 100
        
        for _ in range(test_episodes):
            state, info = env.reset()
            terminated, truncated = False, False
            episode_rewards = 0
            while not terminated and not truncated:
                action = np.argmax(self.model[state, :])
                state, reward, terminated, truncated, _ = env.step(action)
                episode_rewards += reward
            total_rewards += episode_rewards
        
        average_reward = total_rewards / test_episodes
        return {"Average Reward": average_reward}

    def save(self, file_path):
        """Saves the trained Q-table."""
        if self.model is not None:
            np.save(file_path, self.model)
            print(f"Q-table saved to {file_path}.npy")
        else:
            print("Model not trained yet. Cannot save.")