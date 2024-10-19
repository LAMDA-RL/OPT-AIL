import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from utils.utils import soft_update, hard_update, get_concat_samples
from agent.sac_models import DoubleQCritic


class BC(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.detach().cpu().numpy()[0]


    def update(self, policy_buffer, expert_buffer, logger, step):
        expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
        expert_obs, _, expert_action, _, _ = expert_batch[:5]
        
        losses = dict()
        loss = -self.actor.log_prob(expert_obs, expert_action).mean()
        losses['actor_loss'] = loss.item()

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        
        if step % 100 == 0:
            for k, v in losses.items():
                logger.log('train/' + k, v, step)

        return losses

    # Save model parameters
    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        torch.save(self.actor.state_dict(), actor_path)

    # Load model parameters
    def load(self, path, suffix=""):
        actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
        print('Loading models from {}'.format(actor_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
    
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


