import gymnasium as gym
import numpy as np
import random
import pendulum

# パラメータは長さと質量
class RandomizedEnvironment():
    """ Randomized environment class for custom pendulum """
    def __init__(self, env, parameter_ranges):
        self._env = gym.make(env)
        self._parameter_ranges = parameter_ranges
        self._params = np.array([0, 0])
        random.seed(123)

    def sample_env(self):
        length_range = self._parameter_ranges['length']
        # mass_range = self._parameter_ranges['mass']
        length = length_range[0] + (length_range[1] - length_range[0]) * random.random()
        # mass = mass_range[0] + (mass_range[1] - mass_range[0]) * random.random()

        self._params = np.array([length])
        self._env.reset()
        self._env.length = length
        # self._env.mass = mass
        self._env.modify_pendulum_properties(length)  # このメソッドで物理的プロパティをアップデート

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


