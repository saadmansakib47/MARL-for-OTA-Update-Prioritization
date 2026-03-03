import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class OTAManagerEnv:
    def __init__(self, num_vehicles=5):
        self.num_vehicles = num_vehicles
        self.reset()
        
    def reset(self):
        # State: [Veh1_Crit, Veh1_Size, Veh2_Crit, Veh2_Size, ...]
        # Criticality 1-10, Size 1-100MB
        self.state = np.random.randint(1, 11, size=(self.num_vehicles, 2))
        self.done_mask = np.zeros(self.num_vehicles, dtype=bool)
        return self.state.flatten()
    
    def step(self, vehicle_index):
        if self.done_mask[vehicle_index]:
            return self.state.flatten(), 0, np.all(self.done_mask), {"error": "Already updated"}
            
        # Simple reward: Criticality / (Size / 10)
        crit = self.state[vehicle_index, 0]
        size = self.state[vehicle_index, 1]
        reward = crit / (size / 10)
        
        # Mark as done
        self.done_mask[vehicle_index] = True
        self.state[vehicle_index] = [0, 0]
        
        done = np.all(self.done_mask)
        return self.state.flatten(), reward, done, {}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state, done_mask):
        if np.random.rand() < self.epsilon:
            available_actions = np.where(~done_mask)[0]
            if len(available_actions) == 0: return 0
            return np.random.choice(available_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t).numpy()[0]
        
        # Mask already updated vehicles by setting their Q-value very low
        q_values_masked = q_values.copy()
        q_values_masked[done_mask] = -1e9
        return np.argmax(q_values_masked)
    
    def train(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)
        reward_t = torch.tensor(reward, dtype=torch.float32)
        
        current_q = self.model(state_t)[action]
        with torch.no_grad():
            max_next_q = self.model(next_state_t).max()
            target_q = reward_t + (self.gamma * max_next_q if not done else 0)
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_fifo_baseline(env):
    env.reset()
    total_reward = 0
    for i in range(env.num_vehicles):
        _, reward, _, _ = env.step(i)
        total_reward += reward
    return total_reward

def run_random_baseline(env):
    env.reset()
    total_reward = 0
    available = list(range(env.num_vehicles))
    np.random.shuffle(available)
    
    for i in available:
        _, reward, _, _ = env.step(i)
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    env = OTAManagerEnv(num_vehicles=5)
    agent = DQNAgent(state_dim=10, action_dim=5)
    num_episodes = 200
    fifo_rewards = []
    random_rewards = []
    rl_rewards = []

    print("Starting Training...")
    for e in range(num_episodes):
        # Baselines
        fifo_rewards.append(run_fifo_baseline(env))
        random_rewards.append(run_random_baseline(env))
        
        # RL Agent Training
        state = env.reset()
        total_rl_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, env.done_mask)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_rl_reward += reward
        rl_rewards.append(total_rl_reward)
        
        if (e + 1) % 50 == 0:
            print(f"Episode {e+1}/{num_episodes} complete. Epsilon: {agent.epsilon:.3f}")

    print(f"\nResults:")
    print(f"Avg FIFO Reward: {np.mean(fifo_rewards):.2f}")
    print(f"Avg Random Reward: {np.mean(random_rewards):.2f}")
    print(f"Avg RL Reward (Last 50): {np.mean(rl_rewards[-50:]):.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(fifo_rewards, label='FIFO')
    plt.plot(random_rewards, label='Random', alpha=0.5)
    plt.plot(rl_rewards, label='DQN RL Agent', linewidth=2)
    plt.title("OTA Prioritization: RL vs Baselines")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_plot.png")
    print("\nPlot saved as comparison_plot.png")
