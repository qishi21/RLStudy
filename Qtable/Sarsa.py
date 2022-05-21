# Algorithm: Sarsa
# Environment: Maze

import numpy as np
from envs.Maze import Maze # 环境


class Agent:
    def __init__(self, state_dim, action_dim, eps):
        self.eps = eps
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
        self.epsilon = 0.95
        self.lr = 0.01
        self.gamma = 1

    # 根据epsilon-greedy选择动作
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state])

    # 根据greedy选择动作
    def predict(self, state):
        return np.argmax(self.q_table[state])

    # 更新状态-动作价值函数
    def update(self, state, action, reward, next_state, next_action, k):
        # 更新epsilon值
        self.epsilon = 0.95 * (1 - k/self.eps)
        # Q(s_t, a_t) = Q(s_t, a_t) + α * (r + γ * Q(s_t_1, a_t_1) - Q(s_t, a_t))
        self.q_table[state][action] += self.lr * (reward + self.gamma*self.q_table[next_state][next_action] - self.q_table[state][action])


if __name__ == '__main__':
    env = Maze()
    train_eps = 1000
    agent = Agent(env.state_dim, env.action_dim, train_eps)

    # train
    for ep in range(train_eps):
        state = env.reset()
        action = agent.choose_action(state)
        while True:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, ep)
            if done:
                break
            state = next_state
            action = next_action

    # 输出最优策略
    print('algo: Sarsa')
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