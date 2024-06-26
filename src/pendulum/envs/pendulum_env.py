import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import InvertedPendulumEnv
import numpy as np
import os
import random
 
class CustomInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self):
        self.render_mode='human'   
        self._parameter_ranges={
            'length':[0.1,1.5],
            'mass':[3.0,8.0],
        } 
        random.seed(123)
        super().__init__(render_mode=self.render_mode)
        pendulum_id = self.unwrapped.sim.model.body_name2id('pole')
        self._params = self.unwrapped.sim.model.geom_size[pendulum_id][1]#np.array([length])        
        self.length = None
        self.mass = None


    def get_params(self):
        return [self.length, self.mass]

    def modify_pendulum_properties(self,length, mass):
        default_length = 0.3
        pendulum_id = self.unwrapped.sim.model.body_name2id('pole')
        self.unwrapped.sim.model.geom_size[pendulum_id][1] = length  # Assuming geom_size[0] is the relevant dimension
        self.unwrapped.sim.model.body_ipos[pendulum_id][2] = length
        self.unwrapped.sim.model.body_mass[pendulum_id] = mass
        self.unwrapped.sim.forward()


    def reset(self):
        obs = super().reset()
        length_range = self._parameter_ranges['length']
        mass_range = self._parameter_ranges['mass']
        length = length_range[0] + (length_range[1] - length_range[0]) * random.random()
        mass = mass_range[0] + (mass_range[1] - mass_range[0]) * random.random()
        self.length = length
        self.mass = mass
        self.modify_pendulum_properties(length, mass)  
        return obs