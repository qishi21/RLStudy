import numpy as np


class Maze:
    def __init__(self):
        self.state_dim = 64
        self.action_dim = 4
        self.reset_state = False

        self.rewards = np.ones(64) * (-1)
        self.rewards[0:9] = -100
        self.rewards[15] = -100
        self.rewards[18:20] = -100
        self.rewards[21] = -100
        self.rewards[23] = -100
        self.rewards[24] = -100
        self.rewards[27:29] = -100
        self.rewards[31:34] = -100
        self.rewards[36] = -100
        self.rewards[38:41] = -100
        self.rewards[42] = -100
        self.rewards[44] = -100
        self.rewards[47:49] = -100
        self.rewards[53] = -100
        self.rewards[56:] = -100

    def reset(self):
        self.state = 16
        self.reset_state = True
        return self.state

    def step(self, action):
        if not self.reset_state:
            raise NameError('未进行初始化')

        if action not in range(4):
            raise NameError('动作错误')

        if action == 0 and self.state > 7:
            self.state -= 8
        elif action == 1 and self.state not in np.arange(7, 64, 8):
            self.state += 1
        elif action == 2 and self.state < 56:
            self.state += 8
        elif action == 3 and self.state not in np.arange(0, 64, 8):
            self.state -= 1

        if self.state == 55:
            return self.state, self.rewards[self.state], True
        else:
            return self.state, self.rewards[self.state], False
