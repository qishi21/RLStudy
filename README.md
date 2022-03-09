# RLStudy

## Algorithm

| Algorithm                      | Environment    | Code                                                        |
| ------------------------------ | -------------- | ----------------------------------------------------------- |
| Nature DQN                     | CartPole       | [NatureDQN.py](DQN/NatureDQN.py)                            |
| Double DQN                     | CartPole       | [DoubleDQN.py](DQN/DoubleDQN.py)                            |
| Dueling DQN                    | CartPole       | [DuelingDQN.py](DQN/DuelingDQN.py)                          |
| Prioritized Replay DQN         | CartPole       | [PrioritizedReplayDQN.py](DQN/PrioritizedReplayDQN.py)      |
| REINFORCE                      | CartPole       | [REINFORCE.py](PolicyGradient/REINFORCE.py)                 |
| PPO-Clip                       | CartPole       | [PPO.py](PolicyGradient/PPO.py)                             |
| Monte Carlo (Exploring Starts) | SmallGridworld | [MonteCarloES.py](Qtable/MonteCarloES.py)                   |
| Q-learning                     | Maze           | [Qlearning.py](Qtable/Qlearning.py)                         |
| Sarsa                          | Maze           | [Sarsa.py](Qtable/Sarsa.py)                                 |
| 策略迭代                       | SmallGridworld | [PolicyIteration.py](DynamicProgramming/PolicyIteration.py) |
| 价值迭代                       | SmallGridworld | [ValueIteration.py](DynamicProgramming/ValueIteration.py)   |

## Environment

### CartPole

<img src="https://github.com/qishi21/RLStudy/blob/main/.images/image-20220309121426010.png?raw=true" alt="image-20220309121426010" style="zoom:50%;" />

**状态（state）**：

- 小车在轨道上的位置
- 杆子与竖直方向的夹角
- 小车速度
- 角度变化率

**动作（action）**：

- 左移（0）
- 右移（1）

**奖励（reward）**:

- 左移或者右移小车的action之后，env会返回一个+1的reward，到达200个reward之后，游戏会自动结束。



### Maze

<img src="https://github.com/qishi21/RLStudy/blob/main/.images/image-20220309121917373.png?raw=true" alt="image-20220309121917373" style="zoom:50%;" />

- agent 从起点开始，然后到达终点位置。
- 每走一步，得到 -1 的奖励。
- 可以采取的动作是往上下左右走。
- 当前状态用现在 agent 所在的位置来描述。



### SmallGridworld

<img src="https://github.com/qishi21/RLStudy/blob/main/.images/image-20220309122152339.png?raw=true" alt="image-20220309122152339" style="zoom:50%;" />

- 起点位置为状态1-14中的任意一个。
- agent 从起点开始，然后到达左上角和右下角的灰色区域。
- 每走一步，得到 -1 的奖励。
- 可以采取的动作是往上下左右走。
- 当前状态用现在 agent 所在的位置来描述。
