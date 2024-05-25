from gymnasium import register

register(
    id='CustomInvertedPendulumEnv-v0',
    entry_point='pendulum.envs:CustomInvertedPendulumEnv',
    max_episode_steps=500,
)
print("registered CustomInvertedPendulumEnv-v0 environment")
