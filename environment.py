import gymnasium as gym
import numpy as np
import random

import fetch_slide_2

# hardcoded for fetch_slide_2, with only friction
class RandomizedEnvironment:
    """ Randomized environment class """
    def __init__(self, env, parameter_ranges, goal_range):
        self._env = gym.make(env)
        self._parameter_ranges = parameter_ranges
        self._goal_range = goal_range
        self._params = [0]
        random.seed(123)

    def sample_env(self):
        mini = self._parameter_ranges[0]
        maxi = self._parameter_ranges[1]
        pick = mini + (maxi - mini)*random.random()

        self._params = np.array([pick])
        self._env.env.reward_type="dense"
        self._env.set_property('object0', 'geom_friction', [pick, 0.005, .0001])

    def get_env(self):
        """
            Returns a randomized environment and the vector of the parameter
            space that corresponds to this very instance
        """
        return self._env
    
    def get_params(self):
        return self._params

    def close_env(self):
        self._env.close()

    def get_goal(self):
        return

