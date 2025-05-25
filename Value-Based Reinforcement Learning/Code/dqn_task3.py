# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DuelingDQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # shared projection
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True)
        )
        
        # Value and Advantage heads for dueling
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.features(x.float() / 255.0)
        x = self.fc(x)
        v = self.value(x)         # V(s)
        a = self.advantage(x)     # A(s,a)
        # Q(s,a) = V(s) + (A(s,a) – mean_a A(s,a))
        return v + a - a.mean(dim=1, keepdim=True)

class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # Compute priority = (|error| + epsilon) ** alpha
        priority = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            # Buffer not full yet: append new transition
            self.buffer.append(transition)
        else:
            # Buffer full: overwrite the oldest transition
            self.buffer[self.pos] = transition
        # Update the priority at the current position
        self.priorities[self.pos] = priority
        # Move position pointer, wrap around if needed
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ########## 

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # Use full priorities if buffer is full; 
        # otherwise use priorities up to current pos
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        # Calculate sampling probabilities: P(i) = prio_i / sum(prios)
        probs = prios / prios.sum()
        # Randomly sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # Retrieve sampled transitions
        samples = [self.buffer[i] for i in indices]
        # Compute importance-sampling weights: w_i = (N * P(i))^-beta
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)
        ########## END OF YOUR CODE (for Task 3) ########## 

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, err in zip(indices, errors):
            # Compute new priority = (|error| + epsilon) ** alpha
            self.priorities[idx] = (abs(err) + 1e-5) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ########## 
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DuelingDQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DuelingDQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.step_lr_step_size * args.train_per_step,
            gamma=args.step_lr_gamma
        )

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.n_steps = args.n_steps

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # multi-step buffer
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # replay memory
        # prioritized replay memory with initial beta
        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            self.n_step_buffer.clear() # clear buffer at episode start
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)

                # Task3: store in n-step buffer --------------------
                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_steps:
                    R = sum([self.n_step_buffer[i][2] * (self.gamma**i) for i in range(self.n_steps)])
                    s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                    sn, _, _, next_sn, dn = self.n_step_buffer[-1]
                    # compute initial TD error for priority
                    with torch.no_grad():
                        s0_t = torch.from_numpy(np.array(s0)).float().unsqueeze(0).to(self.device)
                        next_sn_t = torch.from_numpy(np.array(next_sn)).float().unsqueeze(0).to(self.device)
                        q0 = self.q_net(s0_t)[0, a0]
                        next_q = self.target_net(next_sn_t).max(1)[0]
                        td_error = (R + (self.gamma**self.n_steps) * next_q * (1 - dn) - q0).abs().item()
                    self.memory.add((s0, a0, R, next_sn, dn), td_error)
                # ------------------------------------------------------

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })
                    
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Learning Rate": self.scheduler.get_last_lr()[0],
            })

            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")

                if not hasattr(self, 'milestones'):
                    self.milestones = [200000, 400000, 600000, 800000, 1000000]
                    self.segment_best = {ms: -float('inf') for ms in self.milestones}

                for ms in self.milestones:
                    if self.env_count <= ms:
                        current_ms = ms
                        break

                if eval_reward >= self.segment_best[current_ms]:
                    self.segment_best[current_ms] = eval_reward
                    seg_path = os.path.join(self.save_dir, f"Lab5_113522118_task3_pong{current_ms}.pt")
                    torch.save(self.q_net.state_dict(), seg_path)
                    print(f"Saved new best for ≤{current_ms} steps: {seg_path} (reward={eval_reward:.2f})")

                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory.buffer) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
        
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        # Double DQN target with multi-step return
        next_actions = self.q_net(next_states).argmax(1)
        next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (self.gamma**self.n_steps) * next_q * (1 - dones)
        loss = 0.5 * (weights.to(self.device) * (target.detach() - q_values).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()

        # update priorities
        td_errors = (target - q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({"Loss": loss.item(), "Q mean": q_values.mean().item(), "std": q_values.std().item(), "self.memory.beta": self.memory.beta})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-DQN-CartPole")
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--step_lr_step_size", type=int, default=100000)
    parser.add_argument("--step_lr_gamma", type=float, default=0.5)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(args.episodes)