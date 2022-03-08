import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# 参数配置
class PGConfig:
    def __init__(self):
        self.train_eps = 200 # 训练批次
        self.eval_eps = 20 # 验证批次
        self.env = 'CartPole-v0' # 环境
        self.algo = 'REINFORCE' # 算法
        self.hidden_dim = 8 # 策略网络中隐藏层的神经单元数量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备选择
        self.lr = 0.02 # 优化器的学习率
        self.update_fre = 1 # 策略网络的更新频率
        self.gamma = 1 # 回报的折扣系数


# 策略网络
# input: state
# output: action
class PGMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """

        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param hidden_dim: 隐藏层的神经单元数量
        """
        super(PGMLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """策略网络

        :param x: 状态 state
        :return: 动作 action
        """
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x


# agent
class REINFORCEAgent:
    def __init__(self, cfg, state_dim, action_dim):
        self.policy_net = PGMLP(state_dim, action_dim, cfg.hidden_dim) # 策略网络
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # 优化器
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.memory = PGMemory() # 轨迹存储
        self.update_fre = cfg.update_fre

    # 根据概率抽取动作
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        prob = dist.log_prob(action)
        action = action.item()
        return action, prob

    # 选择概率值最大的动作
    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action = torch.argmax(probs).item()
        return action

    # 更新策略网络
    def update(self):
        states = self.memory.states
        probs = self.memory.probs
        returns = self.memory.rewards
        dones = self.memory.dones
        for i in reversed(range(len(returns))):
            if not dones[i]:
                returns[i] += self.gamma * returns[i + 1]
        loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for i in range(len(states)):
            loss -= returns[i] * probs[i]
        loss /= self.update_fre
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 轨迹存储
class PGMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.rewards = []
        self.dones = []

    # 存储状态，动作概率，奖励，是否完成
    def push(self, state, prob, reward, done):
        self.states.append(state)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)

    # 将之前存储的轨迹数据清除
    def clear(self):
        self.states = []
        self.probs = []
        self.rewards = []
        self.dones = []


# 配置环境和agent
def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(cfg, state_dim, action_dim)
    return env, agent


# 训练
def train(cfg, env, agent):
    for ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action, prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, prob, reward, done)
            ep_reward += reward
            if done:
                break
            state = next_state
        if ep % cfg.update_fre == 0:
            agent.update()
            agent.memory.clear() # agent更新后将存储的轨迹清除
        if (ep + 1) % 10 == 0:
            print(f'ep:{ep + 1}, reward:{ep_reward}')
    print('Complete training.')
    print('-'*20)


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
        print(f'ep:{ep + 1}, reward:{ep_reward}')
    print('Complete evaluation.')


if __name__ == '__main__':
    cfg = PGConfig()
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)
    eval(cfg, env, agent)
