# Algorithm: Dueling DQN
# Environment: CartPole

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


# 参数配置
class DQNConfig:
    def __init__(self):
        self.env = 'CartPole-v0' # 环境
        self.hidden_dim = 256 # 策略网络中每层隐藏层的神经单元数量
        self.capacity = 100000 # 经验回放池的大小
        self.epsilon_end = 0 # epsilon-greedy 的终值
        self.epsilon_start = 0.95 # epsilon-greedy 的初始值
        self.epsilon_decay = 500 # epsilon-greedy 的衰减率
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备选择
        self.batch_size = 64 # 每次进行参数更新的样本量大小
        self.lr = 0.002 # 优化器的学习率
        self.gamma = 1 # 折扣系数
        self.train_eps = 200 # 训练轮次
        self.eval_eps = 30 # 评估轮次
        self.target_update = 4 # N次之后，将策略网络的参数值复制给目标网络


# 策略、目标网络
class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLP, self).__init__()
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.fc_a = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x_v = F.relu(self.fc_state(x))
        x_v = self.fc_v(x_v)

        x_a = F.relu(self.fc_state(x))
        x_a = self.fc_a(x_a)

        x = x_v.expand_as(x_a) + (x_a - x_a.mean(1).unsqueeze(1).expand_as(x_a))
        return x


# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # 存储 transition = (state, action, reward, next_state, done)
    def push(self, state, action, reward, next_state, done):
        if self.position < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position%self.capacity] = (state, action, reward, next_state, done)
        self.position += 1

    # 进行抽样
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    # 返回回放池的大小
    def __len__(self):
        return len(self.buffer)


# agent
class DQNAgent:
    def __init__(self, cfg, state_dim, action_dim):
        self.action_dim = action_dim # 动作维度

        self.policy_net = MLP(state_dim, cfg.hidden_dim, action_dim) # 策略网络
        self.target_net = MLP(state_dim, cfg.hidden_dim, action_dim) # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 将策略网络的参数值复制给目标网络
        self.memory = ReplayBuffer(cfg.capacity) # experience replay

        # 将 epsilon-greedy 中的 epsilon 不断减小
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start-cfg.epsilon_end)*math.exp(-frame_idx/cfg.epsilon_decay)

        self.device = cfg.device
        self.batch_size = cfg.batch_size # 每次参数更新权重时选择的数据量大小
        self.gamma = cfg.gamma # 折扣系数
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr) # 优化器

    # 根据epsilon-greedy选择动作
    def choose_action(self, state):
        self.frame_idx += 1
        epsilon = self.epsilon(self.frame_idx)

        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
            q_values = self.policy_net(state)
            best_action = q_values.max(1)[1].item()
            action_prob = np.ones(self.action_dim) * epsilon / self.action_dim
            action_prob[best_action] += 1 - epsilon
            action = np.random.choice(np.arange(self.action_dim), p=action_prob)

        return action

    # 根据greedy选择动作
    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    # 更新策略网络参数
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 从经验池中采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # 转换成 torch 格式
        state_batch = torch.tensor(np.array(states), dtype=torch.float, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float, device=self.device)
        done_batch = torch.tensor(np.float32(dones), dtype=torch.float, device=self.device)

        # 采用策略网络计算当前状态的Q值, Q(s_t, a_t)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # 采用目标网络计算下一状态最大的Q值, max(Q(s_t+1, a_t+1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # 期望Q值: reward + gamma * max(Q_t+1) * (1-done)
        expected_q_values = reward_batch + self.gamma*next_q_values*(1-done_batch)
        # 计算损失
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        # 参数更新
        self.optimizer.zero_grad()
        loss.backward()
        # clip 防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


# 配置环境和agent
def env_agent_config(cfg):
    env = gym.make(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(cfg, state_dim, action_dim)
    return env, agent


# 训练
def train(cfg, env, agent):
    for ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            if done:
                break
            state = next_state

        if (ep+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (ep+1) % 10 == 0:
            print(f'episode:{ep+1}/{cfg.train_eps}, reward:{ep_reward}')
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
        print(f'episode:{ep+1}/{cfg.eval_eps}, reward:{ep_reward}')
    print('Complete evaluation.')


if __name__ == '__main__':
    cfg = DQNConfig()
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)
    eval(cfg, env, agent)
