import numpy as np
from maddpg import MADDPG
from sim_env import UAVUSVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import imageio
warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency
    
    image.save(filename)

if __name__ == '__main__':
    env = UAVUSVEnv()
    n_agents = env.num_agents
    
    # 获取每个智能体的观察空间和动作空间维度
    actor_dims = []
    n_actions = []
    for agent_id in env.agents:
        obs_dim = env.observation_space[agent_id].shape[0]
        act_dim = env.action_space[agent_id].shape[0]
        # print(f"Agent {agent_id} - Observation dim: {obs_dim}, Action dim: {act_dim}")
        actor_dims.append(obs_dim)
        n_actions.append(act_dim)
    
    critic_dims = sum(actor_dims)
    # print(f"Critic input dimension: {critic_dims}")

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,
                           alpha=0.0001, beta=0.003, scenario='UAV_USV',
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=256)

    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -30

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')
    
    obs = env.reset()
    print("Initial observation shapes:")

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False]*n_agents
        episode_step = 0
        while not any(dones):
            if evaluate:
                plt.ion()  # 打开交互模式
                env.render_anime(episode_step)
                plt.pause(0.01)  # 暂停一小段时间以便观察
                plt.show()

            actions = maddpg_agents.choose_action(obs,total_steps,evaluate)
            obs_, rewards, dones = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                dones = [True]*n_agents

            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory,total_steps)

            obs = obs_
            score += sum(rewards[0:2])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score',avg_score ,'>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),'; average target score {:.1f}'.format(avg_target_score))
    
    # save data
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)

    if evaluate:
        # 创建动画
        images = []
        image_dir = 'images'
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.png'):
                file_path = os.path.join(image_dir, filename)
                images.append(imageio.imread(file_path))
        imageio.mimsave('uav_pursuit.gif', images, fps=10)