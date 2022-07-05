# Algorithm: PPO-Clip
# Environment: CartPole

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# 参数配置
class PPOConfig:
    def __init__(self):
        self.env = 'CartPole-v0'  # 环境
        self.train_eps = 200  # 训练轮次
        self.eval_eps = 30  # 验证轮次
        self.n_epochs = 4  # 每次参数更新时的轮次
        self.batch_size = 5  # 每一批次的样本量
        self.gamma = 1  # 折扣系数
        self.actor_lr = 0.0003  # actor网络的学习率
        self.critic_lr = 0.0003  # critic网络的学习率
        self.gae_lambda = 0.95
        self.policy_clip = 0.2  # PPO clip值
        self.hidden_dim = 128  # NN中的隐藏层的神经单元数量
        self.update_fre = 100  # 更新频率
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备


# agent
class PPOAgent:
    def __init__(self, cfg, state_dim, action_dim):
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(state_dim, cfg.hidden_dim, action_dim).to(self.device)  # actor网络
        self.critic = Critic(state_dim, cfg.hidden_dim).to(self.device)  # critic网络
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)  # actor优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)  # critic优化器
        self.memory = PPOMemory(cfg.batch_size)  # 经验池

    # 动作选择，返回动作、概率及critic生成的状态值
    def choose_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)

        dist = self.actor(state)
        action = dist.sample()
        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        value = self.critic(state)
        value = torch.squeeze(value).item()
        return action, prob, value

    # 返回概率最大的动作
    def predict(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        dist = self.actor(state)
        return torch.argmax(dist.probs).item()

    # 参数更新
    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, val_arr, reward_arr, done_arr, batches = self.memory.sample()
            # 计算优势函数
            advantage = np.zeros(len(reward_arr), dtype=np.float64)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    # \sum^{\infty}_{l=0}(\gamma\lambda)^l(r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l}))
                    a_t += discount * (
                            reward_arr[k] + self.gamma * val_arr[k + 1] * (1 - int(done_arr[k])) - val_arr[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            # 参数更新
            values = torch.tensor(val_arr).to(self.device)
            for batch in batches:
                # actor_loss
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = prob_ratio * advantage[batch]
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * \
                                         advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # critic_loss
                returns = advantage[batch] + values[batch]
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                critic_loss = torch.mean((returns - critic_value) ** 2)

                # total_loss
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear()


# 经验池
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    # 采样
    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_step]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(
            self.rewards), np.array(self.dones), batches

    # 存储数据
    def push(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    # 清除
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []



# actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


# critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


# 配置环境和agent
def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(cfg, state_dim, action_dim)
    return env, agent


# 训练
def train(cfg, env, agent):
    running_steps = 0
    for ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action, prob, val = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            running_steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if running_steps % cfg.update_fre == 0:
                agent.update()
            if done:
                break
            state = next_state
        if (ep + 1) % 10 == 0:
            print(f'ep:{ep + 1}/{cfg.train_eps}, reward:{ep_reward}')
    print('Complete training.')


# 评估
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
        print(f'ep:{ep + 1}/{cfg.eval_eps}, reward:{ep_reward}')
    print('Complete evaluation.')


if __name__ == '__main__':
    cfg = PPOConfig()
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)
    eval(cfg, env, agent)
