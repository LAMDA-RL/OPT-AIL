import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from utils.utils import soft_update, hard_update, get_concat_samples
from agent.sac_models import DoubleQCritic


class DAC(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
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

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def update(self, policy_buffer, expert_buffer, logger, step):
        policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
        expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

        losses = self.update_discriminator(policy_batch, expert_batch)
        losses.update(self.update_critic(policy_batch, expert_batch))

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

        ones = torch.ones(policy_obs.shape[0], device=self.device)
        zeros = torch.zeros(policy_obs.shape[0], device=self.device)

        disc_output = torch.cat([expert_reward, policy_reward], dim=0)
        disc_label = torch.cat([ones, zeros], dim=0).unsqueeze(dim=1)
        disc_loss = F.binary_cross_entropy_with_logits(disc_output, disc_label, reduction='mean')
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
        batch = get_concat_samples(policy_batch, expert_batch, self.args)
        obs, next_obs, action, reward, done = batch[:5]
        reward = self.infer_r(obs, action)

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action).clip(-100, 100)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V

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
    
    @torch.no_grad()
    def infer_r(self, state, action):
        return self.discriminator(state, action)

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

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



