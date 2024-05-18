import os
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from agent_t import Agent
from replay_buffer_t import Episode, ReplayBuffer
import random
import gc
import pendulum
import gymnasium as gym
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
collected = gc.collect()  # ガベージコレクションを実行
print(f"Garbage collector: collected {collected} objects.")
EPISODES = 100000
directory = "checkpoints"
experiment = "CustomInvertedPendulumEnv-v0"
# Program hyperparameters
TESTING_INTERVAL = 100
TESTING_ROLLOUTS = 50
# Algorithm hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = 400
ROLLOUTS=100
HISTORY_LENGTH = 50  # 振り子のタスクに合わせたステップ数
GAMMA = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 環境サンプラーの初期化
# エージェントの初期化
agent = Agent(BATCH_SIZE)

# リプレイバッファの初期化
replay_buffer = ReplayBuffer(BUFFER_SIZE)

if not os.path.exists(directory):
    os.makedirs(directory)
log_dir = f"{directory}/cart/{current_time}/logs"
writer = SummaryWriter(log_dir=log_dir)
for ep in range(EPISODES):
    ep_rewards = 0
    env = gym.make(experiment)
    env.sample_env()
    env_params = env.get_params()
    episode = Episode([], env_params, HISTORY_LENGTH)
    obs,_ = env.reset()
    env.render()
    last_action = env.action_space.sample()
    obs, reward, truncated, done , info = env.step(last_action)
    ep_rewards += reward
    episode.add_step(last_action, obs, reward, done)
    # print(env.sim.model.geom_size[env.sim.model.body_name2id('pole')][1])
    for i in range(ROLLOUTS):
        action = agent.evaluate_actor(torch.from_numpy(obs).type(torch.float32), episode.get_history())
        if random.random() < 0.1:
            action = agent.apply_action_noise(action.detach())
        action=torch.clamp(action,max=3,min=-3)
        # print(action)
        new_obs, reward, done, truncated,info = env.step(action[0].detach().cpu().numpy())
        episode.add_step(action[0], new_obs, torch.tensor([reward],dtype=torch.float32,requires_grad=True),done)
        obs = new_obs
        ep_rewards += reward
        if done or truncated:
            # print(f"done{i}")
            break
    replay_buffer.add(episode)
    print(f"Episode {ep} finished with reward {ep_rewards} replaybuffer_size {replay_buffer.size()}")
    writer.add_scalar("Reward/episode", ep_rewards, ep)

    # 置き換えられるゴールを考慮したHERは削除または再設計が必要

    # ネットワークのバッチ更新
    if replay_buffer.size() > BATCH_SIZE:
        episodes = replay_buffer.sample_batch(BATCH_SIZE)
        agent.update_networks(episodes, GAMMA)

    #定期的なポリシー評価
    if ep % TESTING_INTERVAL == 0 and ep > 0:
        success_rate, average_reward = agent.evaluate_policy(experiment, TESTING_ROLLOUTS, HISTORY_LENGTH)
        print(f"Testing at episode {ep}, success rate: {success_rate:.2f}, average reward: {average_reward:.2f}")
        agent.save_model(f"{directory}/cart/{current_time}/ckpt_episode_{ep}")
        # with open("csv_log.csv", "a") as csv_log:
        #     csv_log.write(f"{ep}; {success_rate}; {average_reward}\n")

    # gc.set_debug(gc.DEBUG_LEAK)  # デバッグ情報を設定
    # print(gc.get_objects())  # 現在のオブジェクトのリストを取得
    # import ipdb; ipdb.set_trace()
    # obj = replay_buffer  # 調べたいオブジェクト
    # referrers = gc.get_referents(obj)
    # print("Referrers:", referrers)
    # import ipdb; ipdb.set_trace()

    env.close()