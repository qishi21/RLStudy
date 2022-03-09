import numpy as np


# 根据当前的状态和动作生成下一个状态
def generate_next_state(state, action):
    if state not in np.arange(1, 15) or action not in np.arange(4):
        raise NameError('state or action 输入错误.')
    # 当要走出边缘时，会停在原地
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


# 生成最优策略
def generate_best_policy(agent):
    best_policy = [np.inf] * 16
    for state in range(1, 15):
        best_policy[state] = agent.pi[state]

    print('最优策略：')
    print(best_policy[0:4])
    print(best_policy[4:8])
    print(best_policy[8:12])
    print(best_policy[12:16])
    print('-'*30)
    print('状态价值：')
    agent.v = agent.v.reshape((4, 4))
    print(agent.v)