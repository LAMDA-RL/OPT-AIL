import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from module.net import soft_update, hard_update
from module.critic import DoubleQCritic
from utils.utils import get_concat_samples


class OPT_AIL(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.name = "opt_ail"
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        self.discriminator = hydra.utils.instantiate(agent_cfg.disc_cfg, args=args).to(self.device)

        self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)
        self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim

        # optimizers
        self.disc_optimizer = Adam(self.discriminator.parameters(),
                                   lr=agent_cfg.disc_lr,
                                   betas=agent_cfg.disc_betas)
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    @torch.no_grad()
    def infer_r(self, state, action):
        return self.discriminator(state, action)
    
    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach().cpu().numpy()[0]

    def update(self, policy_buffer, expert_buffer, logger, step):
        policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
        expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

        losses = self.update_discriminator(policy_batch, expert_batch)
        losses.update(self.update_critic(policy_batch, expert_batch))
        losses.update(self.update_optimistic(policy_batch, expert_batch))

        if self.actor and step % self.actor_update_frequency == 0:
            obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs)

            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            if self.args.train.soft_update:
                soft_update(self.critic_net, self.critic_target_net,
                            self.critic_tau)
            else:
                hard_update(self.critic_net, self.critic_target_net)
        
        if step % 100 == 0:
            for k, v in losses.items():
                logger.log('train/' + k, v, step)

        return losses

    def update_discriminator(self, policy_batch, expert_batch):
        policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
        expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

        expert_reward = self.discriminator(expert_obs, expert_action)
        policy_reward = self.discriminator(policy_obs, policy_action)

        disc_loss = policy_reward.mean() - expert_reward.mean()
        gp_loss = self.discriminator.grad_pen(expert_obs, expert_action, policy_obs, policy_action) * self.args.method.lambda_gp
        loss = disc_loss + gp_loss

        self.disc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.disc_optimizer.step()

        loss_dict = dict()
        loss_dict['expert_reward'] = expert_reward.mean().item()
        loss_dict['policy_reward'] = policy_reward.mean().item()
        loss_dict['discriminator_loss'] = disc_loss.item()
        loss_dict['gradient_penalty'] = gp_loss.item()
        
        return loss_dict

    def update_critic(self, policy_batch, expert_batch):
        batch = get_concat_samples(policy_batch, expert_batch)
        obs, next_obs, action, reward, done = batch[:5]
        reward = self.infer_r(obs, action)

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = (reward + (1 - done) * self.gamma * target_V).clip(-100, 100)

        # get current Q estimates
        if isinstance(self.critic, DoubleQCritic):
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        else:
            current_Q = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        loss_dict = dict()
        loss_dict['critic_loss'] = critic_loss.item()
        loss_dict['target_Q'] = target_Q.mean().item()
        if isinstance(self.critic, DoubleQCritic):
            loss_dict['current_Q'] = current_Q1.mean().item()
        else:
            loss_dict['current_Q'] = current_Q.mean().item()
        return loss_dict

    def update_actor_and_alpha(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'actor_loss': actor_loss.item(),
            'actor_entropy': -log_prob.mean().item()}

        # self.actor.log(logger, step)
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return losses

    def update_optimistic(self, policy_batch, expert_batch):
        policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
        expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

        with torch.no_grad():
            mex_action = self.actor(policy_obs).mean
        obs = torch.cat([policy_obs, expert_obs], dim=0)
        action = torch.cat([mex_action, expert_action], dim=0)

        if isinstance(self.critic, DoubleQCritic):
            Q_mex1, Q_mex2 = self.critic(obs, action, both=True)
            cat_V1, cat_V2 = self.cqlV(obs, self.critic)
            optimistic_loss = self.args.method.eta * ((cat_V1 - Q_mex1).mean() + (cat_V2 - Q_mex2).mean())
        else:
            Q_mex = self.critic(obs, action)
            cat_V = self.cqlV(obs, self.critic)
            optimistic_loss = self.args.method.eta * (cat_V - Q_mex).mean()

        # optimize encoder and critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        optimistic_loss.backward()
        self.critic_optimizer.step()

        loss_dict = dict()
        loss_dict['optimistic_loss'] = optimistic_loss.item()
        return loss_dict
    
    # Save model parameters
    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"
        disc_path = f"{path}{suffix}_discriminator"

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.discriminator.state_dict(), disc_path)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
        critic_path = f'{path}/{self.args.agent.name}{suffix}_critic'
        disc_path = f'{path}/{self.args.agent.name}{suffix}_discriminator'
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, disc_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        if disc_path is not None:
            self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))

    def sample_actions(self, obs, num_actions):
        """For CQL style training."""
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        action, log_prob, _ = self.actor.sample(obs_temp)
        return action, log_prob.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        """For CQL style training."""
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        if isinstance(network, DoubleQCritic):
            preds1, preds2 = network(obs_temp, actions, both=True)
            preds1 = preds1.view(obs.shape[0], num_repeat, 1)
            preds2 = preds2.view(obs.shape[0], num_repeat, 1)
            return preds1, preds2
        else:
            preds = network(obs_temp, actions)
            preds = preds.view(obs.shape[0], num_repeat, 1)
            return preds

    def cqlV(self, obs, network, num_random=10):
        """For CQL style training."""
        # importance sampled version
        action, log_prob = self.sample_actions(obs, num_random)
        random_action = torch.FloatTensor(obs.shape[0] * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)
        random_density = np.log(0.5 ** action.shape[-1])
        alpha = self.alpha.detach()
        
        if isinstance(network, DoubleQCritic):
            current_Q1, current_Q2 = self._get_tensor_values(obs, action, network)
            rand_Q1, rand_Q2 = self._get_tensor_values(obs, random_action, network)
            cat_Q1 = torch.cat([rand_Q1 - alpha * random_density, current_Q1 - alpha * log_prob.detach()], 1)
            cat_Q2 = torch.cat([rand_Q2 - alpha * random_density, current_Q2 - alpha * log_prob.detach()], 1)
            cql_V1 = torch.logsumexp(cat_Q1 / alpha, dim=1).mean() * alpha
            cql_V2 = torch.logsumexp(cat_Q2 / alpha, dim=1).mean() * alpha
            return cql_V1, cql_V2
        else:
            current_Q = self._get_tensor_values(obs, action, network)
            rand_Q = self._get_tensor_values(obs, random_action, network)
            cat_Q = torch.cat([rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1)
            cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
            return cql_V


