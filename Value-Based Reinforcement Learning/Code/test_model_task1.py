import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import imageio
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),  # 4 is the state dimension for CartPole
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)  # num_actions is the number of actions in the environment
        )

    def forward(self, x):
        return self.network(x)

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def reset(self, obs):
        self.frames = deque([obs for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # Set the seed using reset method
    obs, _ = env.reset(seed=args.seed)

    preprocessor = AtariPreprocessor(frame_stack=1)  # No frame stack required for CartPole
    num_actions = env.action_space.n

    # Load the trained model
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Output directory for video
    os.makedirs(args.output_dir, exist_ok=True)

    total_rewards = []  # List to store rewards for each episode

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)  # Ensure a different seed for each episode
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []

        while not done:
            # Render and store frames for video generation
            frame = env.render()
            frames.append(frame)

            # Convert state to tensor and select action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            # Take a step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        # Save the generated video
        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        total_rewards.append(total_reward)  # Add the reward of this episode to the list

    # Calculate and print the average reward
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
