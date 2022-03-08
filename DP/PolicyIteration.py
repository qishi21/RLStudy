# Policy Iteration
# 根据 Bellman Expectation Equation 迭代

# environment:
#  X    1   2   3
#  4    5   6   7
#  8    9   10  11
#  12   13  14  X
# 每走一步 reward = -1
# X 为终点

import numpy as np


# 根据当前的状态和动作生成下一个状态
def generate_next_state(state, action):
    if state not in np.arange(1, 15) or action not in np.arange(4):
        raise NameError('state or action 输入错误.')
    if state > 3 and action == 0:
        return state - 4
    elif state not in np.arange(3, 16, 4) and action == 1:
        return state + 1
    elif state < 12 and action == 2:
        return state + 4
    elif state not in np.arange(0, 16, 4) and action == 3:
        return state - 1
    else:
        return state


class Policy:
    def __init__(self):
        self.v = np.zeros(16)
        self.q = np.zeros((16, 4))
        self.gamma = 1

        self.pi = dict()
        for i in range(1, 15):
            self.pi[i] = [0,1,2,3]

    # 策略评估
    def evaluate(self):
        # 收敛状态价值函数
        while True:
            v_next = np.zeros(16)
            for state in range(1, 15):
                prob = 1 / len(self.pi[state])
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
    count = 0

    while True:
        pi_pre = agent.pi.copy()
        agent.evaluate()
        agent.improve()

        count += 1

        if np.all(pi_pre == agent.pi):
            break

    best_action = [np.inf] * 16
    for state in range(1, 15):
        best_action[state] = agent.pi[state]

    print('最优策略：')
    print(best_action[0:4])
    print(best_action[4:8])
    print(best_action[8:12])
    print(best_action[12:16])
    print('-'*30)
    print('状态价值：')
    agent.v = agent.v.reshape((4, 4))
    print(agent.v)
    print('-'*30)
    print('策略提升次数：', count-1)
