# Algorithm: Value Iteration
# 根据 Bellman Optimality Equation 迭代
# Environment: SmallGridworld
#  X    1   2   3
#  4    5   6   7
#  8    9   10  11
#  12   13  14  X
# 每走一步 reward = -1
# X 为终点

import numpy as np
from common import generate_next_state, generate_best_policy


# 策略
class Policy:
    def __init__(self):
        self.q = np.zeros((16, 4))
        self.v = np.zeros(16)
        self.pi = dict()
        self.gamma = 1

    # 价值迭代
    def iterate(self):
        while True:
            v_next = np.zeros(16)
            for state in range(1, 15):
                for action in range(4):
                    self.q[state][action] = -1 + self.gamma * self.v[generate_next_state(state, action)]
                v_next[state] = np.max(self.q[state])
            if np.all(v_next == self.v):
                break
            self.v = v_next

    # 生成最优策略
    def retrieve(self):
        for state in range(1, 15):
            self.pi[state] = np.where(self.q[state] == np.max(self.q[state]))[0].tolist()


if __name__ == '__main__':
    agent = Policy()
    # 训练
    agent.iterate()
    agent.retrieve()

    # 输出最优策略
    generate_best_policy(agent)