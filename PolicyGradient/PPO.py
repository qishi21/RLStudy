# PPO-Clip

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOConfig:
    def __init__(self):
        self.env = 'CartPole-v0'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 5
        self.lr = 0.02
        self.gamma = 0.8
        self.policy_clip = 0.1
        self.train_eps = 200
        self.eval_eps = 30
        self.update_fre = 5
        self.hidden_dim = 16


class PPOAgent:
    def __init__(self, cfg, state_dim, action_dim):
        self.policy_net = MLP(state_dim, cfg.hidden_dim, action_dim)
        self.device = cfg.device
        self.action_dim = action_dim
        self.memory = PPOMemory(cfg.batch_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        probs = self.policy_net(state).detach().numpy()
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action, probs[action]

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        probs = self.policy_net(state)
        return torch.argmax(probs).item()

    def update(self):
        state_arr, action_arr, old_prob_arr, reward_arr, done_arr, batch_arr = self.memory.sample()
        for i in reversed(range(len(done_arr))):
            if not done_arr[i]:
                reward_arr[i] += self.gamma * reward_arr[i+1]
        for batch in batch_arr:
            states = torch.tensor(state_arr[batch], dtype=torch.float, device=self.device)
            actions = torch.tensor(action_arr[batch], dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards = torch.tensor(reward_arr[batch]).unsqueeze(1)
            old_probs = torch.tensor(old_prob_arr[batch]).unsqueeze(1)
            probs = self.policy_net(states)
            new_probs = probs.gather(dim=1, index=actions)
            prob_radio = new_probs / old_probs
            weighted_probs = prob_radio * rewards
            weighted_clip_probs = torch.clamp(prob_radio, 1-self.policy_clip, 1+self.policy_clip) * rewards
            objective_function = -torch.sum(torch.min(weighted_probs, weighted_clip_probs))
            self.optimizer.zero_grad()
            objective_function.backward()
            self.optimizer.step()
        self.memory.clear()


class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=0)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.dones), self.batch_size)
        indices = np.arange(len(self.dones))
        np.random.shuffle(indices)
        batch = [indices[index: index+self.batch_size] for index in batch_step]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.rewards), np.array(self.dones), batch

    def push(self, state, action, prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(cfg, state_dim, action_dim)
    return env, agent


def train(cfg, env, agent):
    for ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action, prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, prob, reward, done)
            if done:
                break
            state = next_state
        if ep % cfg.update_fre == 0:
            agent.update()
        if (ep+1) % 10 == 0:
            print(f'ep:{ep+1}/{cfg.train_eps}, reward:{ep_reward}')


def eval(cfg, env, agent):
    for ep in range(cfg.eval_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
            state = next_state
        print(f'ep:{ep+1}/{cfg.eval_eps}, reward:{ep_reward}')


if __name__ == '__main__':
    cfg = PPOConfig()
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)
    eval(cfg, env, agent)
