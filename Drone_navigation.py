import math
import random
import numpy as np
import matplotlib.pyplot as plt
# Changed to gymnasium for modern compatibility and to address warnings
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# -----------------------------
# Custom 2D Drone Navigation Env
# -----------------------------
# Inherit from gymnasium.Env
class Drone2DNavEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200):
        super().__init__()
        # world bounds
        self.bound = 5.0
        self.dt = 0.2
        self.max_steps = max_steps
        # action: accelerations
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # observation: x,y,vx,vy, gx,gy
        obs_low = np.array([-self.bound, -self.bound, -10.0, -10.0, -self.bound, -self.bound], dtype=np.float32)
        obs_high = np.array([self.bound, self.bound, 10.0, 10.0, self.bound, self.bound], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # obstacles: list of (x,y,r)
        self.obstacles = []
        self.goal = np.array([0.0, 0.0])
        self.goal_threshold = 0.25
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def _randomize_scene(self):
        self.goal = np.random.uniform(-3.5, 3.5, size=(2,))
        self.start = np.random.uniform(-3.5, 3.5, size=(2,))
        # simple obstacles: 3 circles
        self.obstacles = []
        for _ in range(3):
            ox, oy = np.random.uniform(-3.0, 3.0, size=(2,))
            r = np.random.uniform(0.3, 0.8)
            self.obstacles.append((ox, oy, r))
        # ensure start and goal not inside obstacles: if inside, move slightly
        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(self.start - np.array([ox, oy])) < (r + 0.2):
                self.start += np.sign(self.start - np.array([ox, oy])) * (r + 0.3)
            if np.linalg.norm(self.goal - np.array([ox, oy])) < (r + 0.2):
                self.goal += np.sign(self.goal - np.array([ox, oy])) * (r + 0.3)

    def reset(self, seed=None, options=None):
        # Gymnasium compliant reset method
        super().reset(seed=seed)
        self._randomize_scene()
        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.pos - self.goal)
        obs = np.concatenate([self.pos, self.vel, self.goal]).astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # simple double-integrator dynamics
        acc = np.array(action)  # ax, ay
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

        self.pos = np.clip(self.pos, -self.bound, self.bound)

        dist = np.linalg.norm(self.pos - self.goal)

        reward = (self.prev_dist - dist) * 10.0
        reward -= 0.01  # time penalty
        self.prev_dist = dist
        terminated = False
        truncated = False
        info = {}

        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(self.pos - np.array([ox, oy])) <= r:
                reward -= 100.0
                terminated = True
                info['reason'] = 'collision'
                break


        if dist <= self.goal_threshold:
            reward += 100.0
            terminated = True
            info['reason'] = 'goal'

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            info['reason'] = 'timeout'

        obs = np.concatenate([self.pos, self.vel, self.goal]).astype(np.float32)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

def train_and_save(model_path="ppo_drone_2d.zip", timesteps=200_000):
    # DummyVecEnv from stable_baselines3 will automatically wrap gymnasium environments
    env = DummyVecEnv([lambda: Drone2DNavEnv(max_steps=200)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_drone_2d_tb/")
    print("Starting training...")
    model.learn(total_timesteps=timesteps)
    print("Training complete; saving model to", model_path)
    model.save(model_path)
    env.close()
    return model_path

def evaluate_and_plot(model_path="ppo_drone_2d.zip", episodes=5):
    # Load model and run a few episodes; collect trajectories and plot
    env = Drone2DNavEnv(max_steps=400)
    model = PPO.load(model_path)
    fig, ax = plt.subplots(figsize=(6,6))

    success_count = 0
    for ep in range(episodes):
        obs, _ = env.reset() # Gymnasium reset returns (observation, info)
        traj = [env.pos.copy()]
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(env.pos.copy())
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], label=f"ep{ep+1}")
        # mark start and goal
        ax.scatter(env.start[0], env.start[1], marker='o', s=40, label=f"start{ep+1}" if ep==0 else None)
        ax.scatter(env.goal[0], env.goal[1], marker='*', s=80, label='goal' if ep==0 else None)
        # plot obstacles
        for (ox, oy, r) in env.obstacles:
            circle = plt.Circle((ox,oy), r, fill=True, alpha=0.3)
            ax.add_patch(circle)

        if 'reason' in info and info['reason'] == 'goal':
            success_count += 1

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_title("Drone trajectories (trained policy)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')
    plt.show()

    success_rate = (success_count / episodes) * 100
    print(f"\nSuccess Rate over {episodes} episodes: {success_rate:.2f}%")
