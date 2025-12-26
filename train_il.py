"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, evaluate
from utils.logger import Logger


torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env

    env = make_env(args)
    eval_env = make_env(args)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)


    if args.agent.name == 'cmil':
        wandb.init(
            entity='chr0nix',
            project='cmil',
            name=f'{args.agent.name}_{args.env.name}_{args.seed}',
            config=OmegaConf.to_container(args, resolve=True),
        )
    else:
        wandb.init(
            entity='chr0nix',
            project='OPT_AIL',
            name=f'{args.agent.name}_{args.env.name}_{args.seed}',
            config=OmegaConf.to_container(args, resolve=True),
        )



    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    if args.agent.name == "mb_ail" or args.agent.name == "mbpo" or args.agent.name == "hyper" or args.agent.name == 'cmil':
        generate_memory_replay = Memory(args.agent.generate_batch_size * args.agent.rollout_min_length * args.agent.model_retain_epochs, args.seed+1)
    else:
        generate_memory_replay = None

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, str(args.seed))
    
    # 设置 replay buffer 保存路径
    buffer_folder = getattr(args, 'buffer_folder', 'buffers/')
    buffer_dir = os.path.join(buffer_folder, args.env.name, str(args.seed))
    
    # 加载保存的 replay buffer（如果存在）
    if hasattr(args, 'load_replay_buffer') and args.load_replay_buffer:
        buffer_path = os.path.join(buffer_dir, 'online_memory_replay.pkl')
        if os.path.exists(buffer_path):
            print(f'--> Loading online replay buffer from {buffer_path}')
            try:
                online_memory_replay.load_replay_buffer(buffer_path)
                print(f'--> Loaded {online_memory_replay.size()} samples from online replay buffer')
            except Exception as e:
                print(f'--> Failed to load online replay buffer: {e}')
        
        if generate_memory_replay is not None:
            buffer_path = os.path.join(buffer_dir, 'generate_memory_replay.pkl')
            if os.path.exists(buffer_path):
                print(f'--> Loading generate replay buffer from {buffer_path}')
                try:
                    generate_memory_replay.load_replay_buffer(buffer_path)
                    print(f'--> Loaded {generate_memory_replay.size()} samples from generate replay buffer')
                except Exception as e:
                    print(f'--> Failed to load generate replay buffer: {e}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)

    steps = 0

    # track mean reward and scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False

        start_time = time.time()
        for episode_step in range(EPISODE_STEPS):

            if steps < args.num_seed_steps:
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            # evaluate
            if steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                logger.log('eval/episode_reward', returns, steps)
                logger.log('eval/episode', epoch, steps)
                logger.dump(steps, ty='eval')
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if returns > best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = returns
                    save(agent, epoch, args, output_dir='results')

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            # Start learning
            if online_memory_replay.size() > INITIAL_MEMORY:
                if begin_learn is False:
                    print('Learn begins!')
                    begin_learn = True

                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    # final_save(agent, args.env.name)
                    # 保存最终的 replay buffer
                    if hasattr(args, 'save_replay_buffer') and args.save_replay_buffer:
                        buffer_folder = getattr(args, 'buffer_folder', 'buffers/')
                        buffer_dir = os.path.join(buffer_folder, args.env.name, str(args.seed))
                        os.makedirs(buffer_dir, exist_ok=True)
                        if online_memory_replay is not None and online_memory_replay.size() > 0:
                            online_memory_replay.save(os.path.join(buffer_dir, args.env.name + '_replay.pkl'))
                        # if generate_memory_replay is not None and generate_memory_replay.size() > 0:
                        #     generate_memory_replay.save(os.path.join(buffer_dir, 'generate_memory_replay.pkl'))
                    return
                
                # update agent
                if args.agent.name == "mb_ail" or args.agent.name == "mbpo" or args.agent.name == "hyper" or args.agent.name == "cmil":
                    for _ in range(args.agent.update_per_step):
                        agent.update(online_memory_replay, generate_memory_replay, expert_memory_replay, logger, learn_steps, epoch)
                else:
                    for _ in range(args.agent.update_per_step):
                        agent.update(online_memory_replay, expert_memory_replay, logger, learn_steps)
                learn_steps += 1

                if learn_steps % 20000 == 0:
                    save_model_by_step(agent, learn_steps, args)

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        logger.log('train/episode', epoch, steps)
        logger.log('train/episode_reward', episode_reward, steps)
        logger.log('train/duration', time.time() - start_time, steps)
        logger.dump(steps, save=begin_learn)
        # save(agent, epoch, args, output_dir='results')

def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'il_{args.env.name}'
        else:
            name = f'il_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')

def save_model_by_step(agent, step, args):
    if not hasattr(args, 'model_folder') or args.model_folder is None:
        return
    
    model_folder = args.model_folder
    agent_name = args.agent.name
    env_name = args.env.name
    seed = args.seed
    
    # 创建 model/{env} 目录
    env_dir = os.path.join(model_folder, env_name)
    os.makedirs(env_dir, exist_ok=True)
    
    # 根据 agent 类型保存相应的组件
    # 文件名格式: {agent}_{component}_{step}_{seed}
    base_name = f"{agent_name}"
    
    # 保存 actor
    if hasattr(agent, 'actor'):
        actor_path = os.path.join(env_dir, f"{base_name}_actor_{step}_{seed}")
        torch.save(agent.actor.state_dict(), actor_path)
        print(f'Saved actor to {actor_path}')
    
    # 保存 critic
    if hasattr(agent, 'critic'):
        critic_path = os.path.join(env_dir, f"{base_name}_critic_{step}_{seed}")
        torch.save(agent.critic.state_dict(), critic_path)
        print(f'Saved critic to {critic_path}')
    
    # 保存 dynamics（如果存在）
    if hasattr(agent, 'dynamics'):
        dynamics_path = os.path.join(env_dir, f"{base_name}_dynamics_{step}_{seed}")
        torch.save(agent.dynamics.state_dict(), dynamics_path)
        print(f'Saved dynamics to {dynamics_path}')
    
    # 保存 discriminator（如果存在）
    if hasattr(agent, 'discriminator'):
        disc_path = os.path.join(env_dir, f"{base_name}_discriminator_{step}_{seed}")
        torch.save(agent.discriminator.state_dict(), disc_path)
        print(f'Saved discriminator to {disc_path}')

def final_save(agent, env):
    path = f'/home/ubuntu/zhangzhilong/mb_ail/pretrain/{env}'
    agent.save(path)


if __name__ == "__main__":
    main()
