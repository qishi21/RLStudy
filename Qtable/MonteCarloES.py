 # Monte Carlo ES(Exploring Starts)

import numpy as np
import math
from exercise.envs.SmallGridworld import SmallGridworld

class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q = np.zeros((state_dim, action_dim))
        self.gamma = 1
        self.count = np.zeros((state_dim, action_dim))
        self.k = 0

    # 生成轨迹
    def generate_trajectory(self, env):
        self.k += 1
        epsilon = 0.95 * math.exp(-self.k / 100)
        states, actions, returns = [], [], []

        state = env.reset()
        action = np.random.randint(self.action_dim)
        while True:
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            returns.append(reward)

            if done:
                break

            state = next_state

            if np.random.random() < epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = np.argmax(self.q[state])

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

    for ep in range(200000):
        states, actions, returns = agent.generate_trajectory(env)
        agent.update(states, actions, returns)

    best_action = ['X']
    for state in range(1, 15):
        best_action.append(np.where(np.abs(agent.q[state]-np.max(agent.q[state]))<0.1)[0].tolist())
    best_action.append('X')
    print('Best action:')
    for i in range(0, 16, 4):
        print(best_action[i:i+4])