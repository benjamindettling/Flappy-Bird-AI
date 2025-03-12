# ================================================================================
# Imports
# ================================================================================

# !pip install gymnasium pygame imageio flappy-bird-gymnasium
import numpy as np
import gymnasium as gym
import pygame
import os
import torch
import torch.nn as nn
import random
import math
import time
import flappy_bird_gymnasium
import csv
import imageio
import shutil
from collections import deque, namedtuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("FlappyBird-v0", use_lidar=True, pipe_gap=140, render_mode='rgb_array')
gamma = 0.99

class TensorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TensorWrapper, self).__init__(env)

    def observation(self, obs):
        return torch.tensor(obs).float().to(device)

class QNet(nn.Module):
    def __init__(self, observation_size, n_actions, hidden_size=[128, 128], activation=nn.ReLU):
        super(QNet, self).__init__()
        layers = []
        input_size = observation_size

        for size in hidden_size:
            layers.append(nn.Linear(input_size, size))
            layers.append(activation())
            input_size = size

        layers.append(nn.Linear(input_size, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self):
        assert len(self.buffer) >= self.batch_size
        batch = random.sample(self.buffer, self.batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


Experience = namedtuple('Experience', ('observation', 'action', 'reward', 'next_observation', 'terminated'))


def _compute_DQN_loss(observations, actions, rewards, next_observations, terminateds, net, gamma, criterion):
    q_values = net(observations)
    actions = actions.unsqueeze(1)
    current_q_values = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = net(next_observations)
        max_next_q_values = next_q_values.max(1)[0]
        max_next_q_values[terminateds] = 0.0
        target_q_values = rewards + gamma * max_next_q_values

    loss = criterion(current_q_values, target_q_values)
    return loss


class DQNAgent:
    def __init__(self, net, n_actions, compute_loss, gamma=0.99, lr=0.001, memory_capacity=10000, batch_size=64):
        self.net = net
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = ExperienceReplay(memory_capacity, batch_size)
        self.compute_loss = compute_loss
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.step_count = 0

    def act(self, observation, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            observation = observation.to(device)
            with torch.no_grad():
                return self.net(observation).argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        observations, actions, rewards, next_observations, terminateds = self.memory.sample()
        observations = torch.stack(observations).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_observations = torch.stack(next_observations).to(device)
        terminateds = torch.BoolTensor(terminateds).to(device)

        loss = self.compute_loss(observations, actions, rewards, next_observations, terminateds, self.net, self.gamma, self.criterion)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.optimizer.step()


def epsilon_scheduler(episode, epsilon_start=1.0, epsilon_end=0.3, n_episodes=1000):
    return epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / n_episodes)


def run_episode(env, agent, epsilon, train=True, add_to_memory=True):
    observation, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(observation, epsilon)
        next_observation, reward, terminated, truncated, _ = env.step(action)

        if add_to_memory:
            agent.memory.push(observation, action, reward, next_observation, terminated)

        if train:
            agent.optimize_model()

        total_reward += reward
        observation = next_observation
        done = terminated or truncated

    return total_reward


def train_DQN_agent(env, agent, n_episodes=1000, test_interval=100, n_test_episodes=10, epsilon_scheduler=None):
    best_test_reward = -float('inf')

    max_time_minutes=30
    start_time = time.time()  # Record the start time
    max_time_seconds = max_time_minutes * 60  # Convert max time to seconds

    for episode in range(n_episodes):

        # Check if the elapsed time exceeds the maximum allowed time
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_seconds:
            print("Training stopped due to time limit.")
            break

        if episode % test_interval == 0:
            test_rewards = [run_episode(env, agent, epsilon=0, train=False, add_to_memory=False) for _ in range(n_test_episodes)]
            average_test_reward = np.mean(test_rewards)
            print(f"Episode {episode}, Average Test Reward: {average_test_reward}")

            if average_test_reward > best_test_reward:
                best_test_reward = average_test_reward
                torch.save(agent.net.state_dict(), 'policy_net.pth')
                print("New best model saved!")

        else:
            epsilon = epsilon_scheduler(episode)
            run_episode(env, agent, epsilon, train=True)


# Set modus:
# 0 = Train from scratch
# 1 = Load model and continue training
# 2 = Load model and only evaluate (no training)
modus = 2  # Change this to 1 or 2 as needed


# Define environment and agent
env = gym.make("FlappyBird-v0", use_lidar=True, pipe_gap=140, render_mode='rgb_array')
env = TensorWrapper(env)

observation_size = env.observation_space.shape[0]
n_actions = env.action_space.n

net = QNet(observation_size, n_actions).to(device)
if modus in [1,2] and os.path.exists("policy_net.pth"):
    print("Loading existing model...")
    net.load_state_dict(torch.load("policy_net.pth", map_location=device))
else:
    print("Training from scratch...")

agent = DQNAgent(net, n_actions, _compute_DQN_loss, gamma=gamma, lr=0.0001, memory_capacity=10000, batch_size=64)

if modus in [0,1]:
  n_episodes=500  # do 1000 or more (total)
  test_interval=10
  n_test_episodes=30

  train_DQN_agent(env, agent, n_episodes, test_interval, n_test_episodes, epsilon_scheduler=lambda ep: epsilon_scheduler(ep, n_episodes))

  # Save model after training
  torch.save(agent.net.state_dict(), "policy_net.pth")
  print("Model saved as policy_net.pth")





def generate_flappybird_gif(agent, filename="flappybird_play", n_frames=6000, frame_skip=2):
    """Generates a GIF of the agent playing Flappy Bird, skipping frames for speed-up."""
    print("\nGenerating Flappy Bird gameplay GIF...")

    image_folder = "flappy_frames_temp"
    os.makedirs(image_folder, exist_ok=True)

    env = gym.make("FlappyBird-v0", use_lidar=True, pipe_gap=140, render_mode='rgb_array')
    env = TensorWrapper(env)

    observation, _ = env.reset()
    frames = []
    total_reward = 0
    done = False

    for i in range(n_frames * frame_skip):  # Capture more frames, but save only some
        action = agent.act(observation, epsilon=0)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Save only every `frame_skip` frame to speed up the GIF
        if i % frame_skip == 0:
            frame = env.render()
            frames.append(frame)

            image_path = f"{image_folder}/frame_{len(frames)-1:04d}.png"
            imageio.imwrite(image_path, frame)

        if done:
            break  # Stop recording if game is over

    env.close()
    print(f"Game over! Total Reward: {total_reward}")

    # Create GIF, ensuring faster playback by reducing the number of frames
    image_files = [f"{image_folder}/frame_{i:04d}.png" for i in range(len(frames))]
    images = [imageio.imread(image_file) for image_file in image_files]
    imageio.mimsave(f"{filename}.gif", images, duration=0.05)  # Standard GIF speed (20 FPS)

    # Cleanup temporary images
    shutil.rmtree(image_folder)

    print(f"GIF saved as {filename}.gif")

    # Display the GIF in Jupyter Notebook (optional)
    try:
        from IPython.display import display, Image
        display(Image(filename=f"{filename}.gif"))
    except ImportError:
        pass  # Skip display if not running in Jupyter


# Watch the AI play after evaluation
generate_flappybird_gif(agent)
