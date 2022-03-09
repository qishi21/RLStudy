import numpy as np


class SmallGridworld:
    def __init__(self):
        self.state_dim = 16
        self.action_dim = 4
        self.reset_state = False
        self.state = None

    def reset(self):
        self.state = np.random.randint(1, 15)
        self.reset_state = True
        return self.state

    def step(self, action):
        if not self.reset_state:
            raise NameError('未进行初始化')

        if action not in range(4):
            raise NameError('动作错误')

        x = self.state // 4
        y = self.state % 4

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < 3:
            y += 1
        elif action == 2 and x < 3:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        if (x == 0 and y == 0) or (x == 3 and y == 3):
            done = True
            self.reset_state = False
        else:
            done = False

        self.state = x * 4 + y
        return self.state, -1, done
