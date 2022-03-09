# Algorithm: MonteCarlo ES(Exploring Starts)
# Environment: SmallGridworld

import numpy as np
from envs.SmallGridworld import SmallGridworld # 环境


# agent
class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim # 动作维度
        self.q = np.zeros((state_dim, action_dim)) # Q-table
        self.gamma = 1 # 折扣系数
        self.count = np.zeros((state_dim, action_dim)) # 计数
        self.max_step = 20 # 每个episode的最大步数

    # 生成轨迹
    def generate_trajectory(self, env):
        states, actions, returns = [], [], []

        state = env.reset() # 随机生成状态
        action = np.random.randint(self.action_dim) # 随机生成动作
        for step in range(self.max_step):
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            returns.append(reward)
            if done:
                break
            state = next_state
            action = np.argmax(self.q[state]) # 根据 greedy 采取动作

        for t in reversed(range(len(returns)-1)):
            returns[t] += self.gamma * returns[t+1]

        return states, actions, returns

    # 更新 Q_table
    def update(self, states, actions, returns):
        for state, action, G in zip(states, actions, returns):
            self.count[state][action] += 1
            self.q[state][action] += 1 / self.count[state][action] * (G - self.q[state][action])


if __name__ == '__main__':
    env = SmallGridworld()
    agent = Agent(env.state_dim, env.action_dim)

    # 训练
    for ep in range(200000):
        states, actions, returns = agent.generate_trajectory(env)
        agent.update(states, actions, returns)

    # 输出最优动作
    best_action = ['X']
    for state in range(1, 15):
        best_action.append(np.where(np.abs(agent.q[state]-np.max(agent.q[state]))<0.1)[0].tolist())
    best_action.append('X')
    print('algo: MonteCarloES')
    print('\nBest policy:\t(0:↑ 1:→ 2:↓ 3:←)')
    for i in range(0, 16, 4):
        print(best_action[i:i+4])
