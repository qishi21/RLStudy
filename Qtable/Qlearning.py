# Algorithm: Q-learning
# Environment: Maze

import numpy as np
from envs.Maze import Maze # 环境


class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
        self.epsilon = 0.95
        self.gamma = 1
        self.lr = 0.02

    # 根据epsilon-greedy选择动作
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(self.q_table[state])
        return action

    # 根据greedy选择动作
    def predict(self, state):
        return np.argmax(self.q_table[state])

    # 更新状态-动作价值函数
    def update(self, state, next_state, action, reward):
        # Q(s_t, a_t) = Q(s_t, a_t) + α * (r + γ * max(Q(s_t_1, a_t_1)) - Q(s_t, a_t))
        max_q_table_next_state = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (reward + self.gamma*max_q_table_next_state - self.q_table[state][action])


if __name__ == '__main__':
    env = Maze()
    state_dim = env.state_dim # 环境维度
    action_dim = env.action_dim # 动作维度
    train_eps = 2000 # 训练轮次
    agent = Agent(state_dim, action_dim)

    # 训练
    for ep in range(train_eps):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, next_state, action, reward)
            if done:
                break
            state = next_state

    # 输出最优策略
    print('algo: Q-learning')
    rewards = 0
    state = env.reset()
    print('\nBest action:\t(0:↑ 1:→ 2:↓ 3:←)')
    while True:
        action = agent.predict(state)
        if action == 0:
            print('↑', end='\t')
        elif action == 1:
            print('→', end='\t')
        elif action == 2:
            print('↓', end='\t')
        elif action == 3:
            print('←', end='\t')
        next_state, reward, done = env.step(action)
        rewards += reward
        if done:
            break
        state = next_state
    print('\nReturn:', rewards)