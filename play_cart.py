import numpy as np
import random
import gymnasium as gym
import torch
from environment_cart import RandomizedEnvironment
from agent_t import Agent
from replay_buffer_t import Episode


experiment="CustomInvertedPendulumEnv-v0"

MODEL_NAME = "checkpoints/ckpt_episode_100"
ROLLOUT_NUMBER = 1000
BATCH_SIZE = 32
MAX_STEPS = 50


RENDER = True

parameter_ranges = {
    'length': [0.2, 0.6],  # 長さの範囲
    # 'mass': [1.0, 2.0]     # 質量の範囲
}

# initialize the environment sampler
randomized_environment = RandomizedEnvironment(experiment, parameter_ranges)

# initialize the agent, both the actor/critic (and target counterparts) networks
agent = Agent(randomized_environment.get_env(), BATCH_SIZE*MAX_STEPS)

# load the wanted model
agent.load_model(MODEL_NAME)

success_number = 0


for test_ep in range(ROLLOUT_NUMBER):
    randomized_environment.sample_env()
    env= randomized_environment.get_env()
    env_params = randomized_environment.get_params()
    print("Episode {}".format(test_ep))

    ep_rewards = 0

    current_obs_dict,_ = randomized_environment.get_env().reset()

    # read the current goal, and initialize the episode
    goal = 0.0
    episode = Episode(goal, env_params, MAX_STEPS)

    # get the first observation and first fake "old-action"
    # TODO: decide if this fake action should be zero or random
    last_action = randomized_environment.get_env().action_space.sample()
    obs, reward, truncated, done , info = randomized_environment.get_env().step(last_action)
    ep_rewards += reward
    episode.add_step(last_action, obs, reward, done)

    # rollout the whole episode
    for ep in range(ROLLOUT_NUMBER):
        if RENDER: env.render()
        action = agent.evaluate_actor(torch.from_numpy(obs).type(torch.float32),torch.tensor([goal]).type(torch.float32), episode.get_history())
        new_obs, reward, done, truncated,info = randomized_environment.get_env().step(action[0].detach().cpu().numpy())
        episode.add_step(action[0], new_obs, torch.tensor([reward],dtype=torch.float32,requires_grad=True),done)
        obs = new_obs
        ep_rewards += reward
    if info['is_success'] > 0.0:
        success_number += 1
    print(f"Episode {ep} finished with reward {ep_rewards}")
    #randomized_environment.close_env()

print("Success rate : {}".format(success_number/ROLLOUT_NUMBER))
