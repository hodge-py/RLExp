import gymnasium as gym
import numpy as np


class Maze(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.player = None
        self.maze = None



    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def _get_obs(self):
        pass