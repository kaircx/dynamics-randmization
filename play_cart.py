import numpy as np
import random
import gymnasium as gym
import torch
from agent_t import Agent
from replay_buffer_t import Episode


experiment="CustomInvertedPendulumEnv-v0"

MODEL_NAME = "checkpoints/cart/20240513-085644/ckpt_episode_200"
ROLLOUT_NUMBER = 1000
BATCH_SIZE = 32
MAX_STEPS = 50

RENDER = True

parameter_ranges = {
    'length': [0.2, 0.6],  # 長さの範囲
    # 'mass': [1.0, 2.0]     # 質量の範囲
}

# initialize the environment sampler


# initialize the agent, both the actor/critic (and target counterparts) networks
agent = Agent(BATCH_SIZE*MAX_STEPS)

# load the wanted model
agent.load_model(MODEL_NAME)

success_number = 0


for test_ep in range(ROLLOUT_NUMBER):
    env= gym.make(experiment)
    env.sample_env()
    env_params = env.get_params()
    print("Episode {}".format(test_ep))

    ep_rewards = 0

    current_obs_dict,_ = env.reset()

    # read the current goal, and initialize the episode
    episode = Episode(env_params, MAX_STEPS)

    # get the first observation and first fake "old-action"
    # TODO: decide if this fake action should be zero or random
    last_action = env.action_space.sample()
    obs, reward, truncated, done , info = env.step(last_action)
    ep_rewards += reward
    episode.add_step(last_action, obs, reward, done)

    # rollout the whole episode
    for ep in range(ROLLOUT_NUMBER):
        if RENDER: env.render()
        action = agent.evaluate_actor(torch.from_numpy(obs).type(torch.float32), episode.get_history())
        new_obs, reward, done, truncated,info = env.step(action[0].detach().cpu().numpy())
        episode.add_step(action[0], new_obs, torch.tensor([reward],dtype=torch.float32,requires_grad=True),done)
        obs = new_obs
        ep_rewards += reward
    if info['is_success'] > 0.0:
        success_number += 1
    print(f"Episode {ep} finished with reward {ep_rewards}")
    env.close()

print("Success rate : {}".format(success_number/ROLLOUT_NUMBER))
