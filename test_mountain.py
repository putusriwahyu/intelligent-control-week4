import gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

# Inisialisasi environment dan agen
env = gym.make("MountainCar-v0", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Set epsilon rendah supaya pakai policy terlatih (bukan eksplorasi acak)
agent.epsilon = 0.01

# List untuk menyimpan skor tiap episode
scores = []
moving_avg_scores = []
window_size = 10

# Jalankan beberapa episode
for e in range(20):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(200):  # max steps di MountainCar
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        if done:
            print(f"Episode {e+1}: Score = {total_reward}")
            break

    scores.append(total_reward)

    # Hitung moving average
    if len(scores) >= window_size:
        moving_avg = np.mean(scores[-window_size:])
        moving_avg_scores.append(moving_avg)
    else:
        moving_avg_scores.append(np.mean(scores))

env.close()

# Plot hasil
plt.plot(moving_avg_scores)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Testing Performance on MountainCar-v0 (Moving Average)')
plt.grid(True)
plt.show()
