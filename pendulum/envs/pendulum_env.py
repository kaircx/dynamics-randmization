import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import InvertedPendulumEnv
import numpy as np
import os

 
class CustomInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self):

        # self.render_mode='human'    
        super().__init__(render_mode=self.render_mode)

    def modify_pendulum_properties(self,length):
        pendulum_id = self.unwrapped.sim.model.body_name2id('pole')
        # self.sim.model.body_mass[pendulum_id] = self.mass
        self.unwrapped.sim.model.geom_size[pendulum_id][1] = length  # Assuming geom_size[0] is the relevant dimension
        self.unwrapped.sim.model.geom_pos[pendulum_id][2] = -length  # Assuming geom_size[0] is the relevant dimension
        self.unwrapped.sim.forward()

    def reset(self):
        """ Call reset of the super class and modify properties """
        obs = super().reset()
        # self.modify_pendulum_properties()
        return obs

    def step(self, action):
        reward=0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        angle = ob[1]  # 仮定として、ob[0]がペンジュラムの角度を示すとします。
        reward += np.abs(angle)  # 角度の絶対値の負の値を報酬とする（角度が0に近づくほど報酬が増加）
        # 終了条件を定義
        terminated = bool(not np.isfinite(ob).all())
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}