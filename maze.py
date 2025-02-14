import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

class Maze(gym.Env):

    def __init__(self):
        self.player_position = np.array([0, 0])
        self.maze = np.array([
                    [0, 1, 0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0],
                    [0, 1, 0 ,1, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 2]
                ])
        self.current_step = 0
        self.array_player = []
        self.array_reward = []
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=len(self.maze)-1, shape=(2,))

        self.action_step = {0: [1,0], 1: [0,1], 2: [-1,0], 3: [0,-1]}

        self.reward = 0


    def step(self, action):
        actions = self.action_step[action]

        if actions == 0:
            seflf\= \
        #reward structure
        # every move -.1, 1 for reaching the goal
        obs = self._get_obs()

        self.current_step += 1

        if self.maze[self.player_position[0]][self.player_position[1]] == 2:
            reward = 1
            done = True
        else:
            reward = -.1
            done = False

        self.reward = reward
        info = {}

        self.render()

        return obs, reward, done, False, info

    def reset(self,seed=None,options=None):
        self.current_step = 0
        self.player_position = [0, 0]
        self.reward = 0
        return self._get_obs(), {}

    def render(self):
        self.array_player.append(self.player_position)
        self.array_reward.append(self.reward)

    def _get_obs(self):
        return np.array(self.player_position)

    def renderAll(self):
        return self.array_player, self.array_reward, len(self.maze[0])

env = Maze()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()

while True:
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    if done:
        break

dataOutput = env.renderAll()
print(dataOutput)
