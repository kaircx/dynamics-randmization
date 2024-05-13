import os
from environment import RandomizedEnvironment
from agent_t import Agent
from replay_buffer_t import Episode, ReplayBuffer
import random
from tensorboardX import SummaryWriter
import torch
from matplotlib import pyplot as plt
plt.ion() 
EPISODES = 1000000

directory = "checkpoints"
experiment = "FetchSlide2-v1"
# Program hyperparameters
TESTING_INTERVAL = 200  # number of updates between two evaluation of the policy
TESTING_ROLLOUTS = 100  # number of rollouts performed to evaluate the current policy

# Algorithm hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = 1000
HISTORY_LENGTH = 50  # WARNING: defined in multiple files...
GAMMA = 0.99
K = 0.8  # probability of replay with H.E.R.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment sampler
randomized_environment = RandomizedEnvironment(experiment, [0.0, 1.0], [])

# Initialize the agent
agent = Agent(randomized_environment.get_env(), BATCH_SIZE)




# Initialize the replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(log_dir=f"{directory}/logs")
total_rewards = []
for ep in range(EPISODES):
    ep_rewards = 0
    randomized_environment.sample_env()
    env_params = randomized_environment.get_params()
    current_obs_dict,_ = randomized_environment.get_env().reset()
    goal = current_obs_dict['desired_goal']
    episode = Episode(goal, env_params, HISTORY_LENGTH)
    obs = current_obs_dict['observation']
    achieved = current_obs_dict['achieved_goal']
    last_action = randomized_environment.get_env().action_space.sample()  # Consider initial action choice
    reward = randomized_environment.get_env().compute_reward(achieved, goal, 0)
    ep_rewards += reward
    episode.add_step(last_action, obs, reward, achieved,False)
    truncated = False
    while not truncated:
        obs = current_obs_dict['observation']
        history = episode.get_history()
        action = agent.evaluate_actor(torch.from_numpy(obs.copy()).type(torch.float32),torch.from_numpy(goal.copy()).type(torch.float32), history)
        action += agent.apply_action_noise(action)
        action=action.detach().cpu().numpy()[0]
        new_obs_dict, step_reward, done, truncated, info = randomized_environment.get_env().step(action)
        new_obs = new_obs_dict['observation']
        goal = new_obs_dict['desired_goal']
        episode.add_step(action, new_obs, step_reward, goal, done)
        ep_rewards += step_reward
        current_obs_dict = new_obs_dict

    replay_buffer.add(episode)
    print(f"Episode {ep} finished with reward {ep_rewards}")
    total_rewards.append(ep_rewards)
    writer.add_scalar("Reward/episode", ep_rewards, ep)

    # Replay the episode with HER with probability k
    if random.random() < K:
        new_goal = current_obs_dict['achieved_goal']
        replay_episode = Episode(new_goal, env_params, HISTORY_LENGTH)
        for action, state, achieved_goal, done in zip(episode.get_actions(), episode.get_states(), episode.get_achieved_goals(), episode.get_terminal()):
            achieved_goal_ = achieved_goal.clone().detach()
            step_reward = randomized_environment.get_env().compute_reward(achieved_goal_, torch.tensor(new_goal).to(device), 0)
            replay_episode.add_step(action.detach().numpy(), state.detach().numpy(), step_reward, new_goal, terminal=done)
        replay_buffer.add(replay_episode)

    # Perform a batch update of the network if a large enough batch can be sampled from the replay buffer
    if replay_buffer.size() > BATCH_SIZE:
        episodes = replay_buffer.sample_batch(BATCH_SIZE)
        agent.update_networks(episodes, GAMMA)

    if ep % TESTING_INTERVAL == 0:
        success_number, average_reward = agent.evaluate_policy(randomized_environment, TESTING_ROLLOUTS, HISTORY_LENGTH)
        print(f"Testing at episode {ep}, success rate: {success_number / TESTING_ROLLOUTS:.2f}, average reward: {average_reward:.2f}")
        agent.save_model(f"{directory}/ckpt_episode_{ep}")
        with open("csv_log.csv", "a") as csv_log:
            csv_log.write(f"{ep}; {success_number / TESTING_ROLLOUTS}; {average_reward}\n")

    randomized_environment.close_env()
