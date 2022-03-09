# Algorithm: Policy Iteration
# 根据 Bellman Expectation Equation 迭代
# Environment: SmallGridworld
#  X    1   2   3
#  4    5   6   7
#  8    9   10  11
#  12   13  14  X
# 每走一步 reward = -1
# 初始状态为1-14中任意一处
# X 为终点

import numpy as np
from common import generate_next_state, generate_best_policy


 # 策略
class Policy:
    def __init__(self):
        self.v = np.zeros(16) # state value function
        self.q = np.zeros((16, 4)) # state-action value function
        self.gamma = 1 # 折扣系数

        self.pi = dict() # 策略
        for i in range(1, 15):
            self.pi[i] = [0,1,2,3]

    # 策略评估
    def evaluate(self):
        # 收敛状态价值函数
        while True:
            v_next = np.zeros(16)
            for state in range(1, 15):
                prob = 1 / len(self.pi[state]) # π(a|s)
                for action in self.pi[state]:
                    v_next[state] += prob * (-1 + self.gamma*self.v[generate_next_state(state, action)])
            if np.all(v_next == self.v):
                break
            self.v = v_next.copy()

    # 策略提升
    def improve(self):
        # 根据状态价值函数计算出状态-动作价值函数
        for state in range(1, 15):
            for action in range(4):
                self.q[state][action] = -1 + self.gamma*self.v[generate_next_state(state, action)]

        # 根据状态-动作价值函数生成最优策略
        for state in range(1, 15):
            self.pi[state] = np.where(self.q[state] == np.max(self.q[state]))[0].tolist()


if __name__ == '__main__':
    agent = Policy()

    # 训练
    while True:
        pi_pre = agent.pi.copy() # 每轮迭代前的策略

        # 策略迭代
        agent.evaluate()
        agent.improve()

        # 若迭代前和迭代后的策略相同，则停止迭代
        if np.all(pi_pre == agent.pi):
            break

    # 输出最优策略
    generate_best_policy(agent)