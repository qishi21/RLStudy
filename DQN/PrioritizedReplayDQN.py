# Algorithm: Prioritized Replay DQN (基于DDQN)
# Environment: CartPole

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


# 参数配置
class PRDQNConfig:
    def __init__(self):
        self.env = 'CartPole-v0'  # 环境
        self.train_eps = 200  # 训练轮次
        self.eval_eps = 30  # 评估轮次
        self.capacity = 100000  # 经验回放池大小
        self.hidden_dim = 256  # 策略网络中每层隐藏层的神经单元数量
        self.start_epsilon = 0.95  # epsilon-greedy 的初始值
        self.end_epsilon = 0.  # epsilon-greedy 的终值
        self.batch_size = 12  # 每次进行参数更新的样本量大小
        self.gamma = 1  # 折扣系数
        self.lr = 0.0001  # 优化器的学习率


# 经验回放池存储
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放池的大小
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # SumTree 存放优先级的值
        self.data = np.zeros(capacity, dtype=object)  # 存放 transition = (state, action, reward, next_state, done)
        self.data_pointer = 0  # 指向 self.data 中未存放数据的第一个索引
        self.abs_err_upper = 1.  # TD Error 的上界
        self.beta = 0.4

    # 从回放池中采样， TD误差越大，越容易被采样
    def sample(self, batch_size):
        tree_idx, batch, ISWeights = np.zeros(batch_size, dtype=np.int32), np.zeros(
            (batch_size, self.data[0].shape[0])), np.zeros((batch_size, 1))
        pri_seg = self.total_p / batch_size  # 均匀区间，在每个区间内分别抽样
        self.beta = np.min((1., self.beta + 0.0001))  # self.beta 上界为1
        # 最小概率
        min_p = np.min(self.tree[-self.capacity:])
        if min_p == 0:
            min_p = 0.00001
        for i in range(batch_size):
            v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))  # 取均匀分布
            leaf_idx, transition, p = self.get_leaf(v)
            tree_idx[i] = leaf_idx
            batch[i] = transition
            ISWeights[i][0] = np.power(p / min_p, -self.beta)  # 损失函数权重，按照公式w_j = (p_j / min(p))^(-beta)进行计算

        return tree_idx, batch, ISWeights

    # 给定随机值，从SumTree中返回相应的叶子结点索引、transition、优先权重
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 若左子节点的索引大于tree的长度，则返回其父节点
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 若v值小于等于左子节点的值，则将左子节点设为父节点
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                # 若v值大于左子节点的值，则v值减去左子节点的值，再将右子节点设为父节点
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        return leaf_idx, self.data[leaf_idx - self.capacity + 1], self.tree[leaf_idx]

    # 存储 transition及优先级值
    def store_transition(self, state, action, reward, next_state, done):
        # 存储transition
        transition = np.hstack((state, action, reward, next_state, done))
        self.data[self.data_pointer] = transition

        # 存储优先级值
        tree_idx = self.data_pointer + self.capacity - 1
        max_p = np.max(self.tree[-self.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree_update(tree_idx, max_p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # 更新SumTree上的优先级值
    def tree_update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 梯度更新后，重新计算TD误差，并存储到SumTree中
    def batch_update(self, tree_idx, errors):
        errors += 0.0001
        clipped_errors = np.minimum(errors.detach().numpy(), self.abs_err_upper)
        for ti, p in zip(tree_idx, clipped_errors):
            self.tree_update(ti, p)

    # 返回SumTree根节点值，即优先级值之和
    @property
    def total_p(self):
        return self.tree[0]


# agent
class PRDQNAgent:
    def __init__(self, cfg, state_dim, action_dim):
        self.action_dim = action_dim  # 动作维度
        self.policy_net = MLP(state_dim, cfg.hidden_dim, action_dim)  # 策略网络
        self.target_net = MLP(state_dim, cfg.hidden_dim, action_dim)  # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 将策略网络的参数值复制给目标网络
        self.start_epsilon = cfg.start_epsilon  # epsilon 的初始值
        self.end_epsilon = cfg.end_epsilon  # epsilon 的终值
        self.epsilon = cfg.start_epsilon  # epsilon 值
        self.memory = SumTree(cfg.capacity)  # 经验回放池
        self.replay_total = 0  # 总共存放的次数
        self.batch_size = cfg.batch_size  # 每次参数更新时选择的样本量
        self.gamma = cfg.gamma  # 折扣系数
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器

    # 根据 epsilon-greedy 选择动作
    def choose_action(self, state):
        self.epsilon -= (self.start_epsilon - self.end_epsilon) / 20000
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.policy_net(state)
                return torch.argmax(q_values, dim=-1).item()
        else:
            return np.random.randint(self.action_dim)

    # 根据 greedy 选择动作
    def predict(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            q_values = self.policy_net(state)
            return torch.argmax(q_values, dim=-1).item()

    # agent参数更新
    def update(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
        self.replay_total += 1
        if self.replay_total > self.batch_size:
            tree_idx, batch, ISWeights = self.memory.sample(self.batch_size)  # 从经验回放池中进行采样

            # 转换成 torch 形式
            state_batch = torch.FloatTensor(batch[:, 0:4])
            action_batch = torch.tensor(batch[:, 4], dtype=torch.int64).unsqueeze(1)
            reward_batch = torch.FloatTensor(batch[:, 5]).unsqueeze(1)
            next_state_batch = torch.FloatTensor(batch[:, 6:10])
            done_batch = torch.FloatTensor(batch[:, 10]).unsqueeze(1)

            # 由策略网络生成的当前状态的Q值 Q(s_t, a_t)
            q_policy_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

            # 由策略网络生成的下一状态的最优动作 argmax Q(s_t+1, a_t+1) -> a_max_policy
            next_q_policy_action = torch.argmax(self.policy_net(next_state_batch), dim=1).unsqueeze(1)
            # 根据策略网络生成的最优动作，由目标网络计算Q值, Q(s_t+1, a_max_policy)
            next_q_target_values = self.target_net(next_state_batch).gather(dim=1, index=next_q_policy_action)

            # 期望Q值: reward + gamma * next_Q_value * (1-done)
            expected_q_values = reward_batch + self.gamma * next_q_target_values * (1 - done_batch)

            # 计算带权重的损失
            loss = torch.mean(torch.FloatTensor(ISWeights) * (q_policy_values - expected_q_values) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 参数更新后，重新计算 Q(s_t, a_t)
            new_q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
            # 计算 TD 误差
            abs_errors = torch.abs(expected_q_values - new_q_values)
            # 存放至 SumTree 中
            self.memory.batch_update(tree_idx, abs_errors)

    # 更新目标网络的参数
    def update_target_params(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 策略、目标网络
class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 配置环境和agent
def env_agent_cfg(cfg):
    env = gym.make(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PRDQNAgent(cfg, state_dim, action_dim)
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
            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        agent.update_target_params()
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
    cfg = PRDQNConfig()
    env, agent = env_agent_cfg(cfg)
    train(cfg, env, agent)
    eval(cfg, env, agent)
