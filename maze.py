import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

class Maze(gym.Env):

    def __init__(self):
        self.player_position = [0, 0]
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
        self.done = False

        self.reward = 0


    def step(self, action):

        if action == 0:
            self.player_position[0] += 1
            if self.player_position[0] >= len(self.maze):
                self.player_position[0] -= 1
            elif self.maze[self.player_position[0]][self.player_position[1]] == 1:
                self.player_position[0] -= 1
        elif action == 1:
            self.player_position[1] += 1
            if self.player_position[1] >= len(self.maze[0]):
                self.player_position[1] -= 1
            elif self.maze[self.player_position[0]][self.player_position[1]] == 1:
                self.player_position[1] -= 1
        elif action == 2:
            self.player_position[0] -= 1
            if self.player_position[0] < 0:
                self.player_position[0] += 1
            elif self.maze[self.player_position[0]][self.player_position[1]] == 1:
                self.player_position[0] += 1
        else:
            self.player_position[1] -= 1
            if self.player_position[1] < 0:
                self.player_position[1] += 1
            elif self.maze[self.player_position[0]][self.player_position[1]] == 1:
                self.player_position[1] += 1

        #reward structure
        # every move -.1, 1 for reaching the goal
        obs = self._get_obs()

        self.current_step += 1


        if self.maze[self.player_position[0]][self.player_position[1]] == 2:
            reward = 1
            done = True
        else:
            reward = -.3
            done = False

        self.reward = reward

        self.render()
        info = {}
        self.done = done

        return obs, reward, done, False, info

    def reset(self,seed=None,options=None):
        self.current_step = 0
        self.player_position = [0, 0]
        self.reward = 0
        self.done = False
        return self._get_obs(), {}

    def render(self):
        self.array_player.append(self.player_position)
        self.array_reward.append(self.reward)

    def _get_obs(self):
        return self.player_position

    def renderAll(self):
        return self.array_player, self.array_reward

env = Maze()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
done = False
while not done:
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)

dataOutput = env.renderAll()
print(dataOutput)
