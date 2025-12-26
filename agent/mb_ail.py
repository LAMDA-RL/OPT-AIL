import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra
import numpy as np

from module.critic import DoubleQCritic
from module.net import soft_update, hard_update
from module.scaler import StandardScaler
from utils.utils import get_concat_samples


class MB_AIL(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.name = 'mb_ail'
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.dynamics_update_frequency = agent_cfg.dynamics_update_freq
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        # Add rollout parameters
        self.rollout_min_length = agent_cfg.rollout_min_length 
        self.rollout_max_length = agent_cfg.rollout_max_length
        self.rollout_min_epoch = agent_cfg.rollout_min_epoch
        self.rollout_max_epoch = agent_cfg.rollout_max_epoch
        self.rollout_length = self.rollout_min_length
        self.include_ent_in_adv = agent_cfg.include_ent_in_adv
        self.adv_weight = agent_cfg.adv_weight

        # Initialize generate buffer
        self.generate_batch_size = agent_cfg.generate_batch_size
        self.model_retain_epochs = agent_cfg.model_retain_epochs
        self.real_ratio = agent_cfg.real_ratio
        self.generate_sample_size = np.zeros(self.model_retain_epochs)

        self.logvar_loss_coef = agent_cfg.logvar_loss_coef
        self.holdout_ratio = agent_cfg.holdout_ratio
        self.num_ensemble = agent_cfg.dynamics_cfg.num_ensemble
        self.num_elites = agent_cfg.dynamics_cfg.num_elites
        self.max_epochs_since_update = agent_cfg.max_epochs_since_update
        self.fixed_std = agent_cfg.fixed_std
        self.opt = agent_cfg.opt

        self.dynamics = hydra.utils.instantiate(agent_cfg.dynamics_cfg, args=args).to(self.device)
        self.obs_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.discriminator = hydra.utils.instantiate(agent_cfg.disc_cfg, args=args).to(self.device)

        self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)
        self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        
        self.constant_dim = []
        self.update_dim = [i for i in range(self.obs_dim)]

        # optimizers
        self.dynamics_optimizer = Adam(self.dynamics.parameters(),
                                       lr=agent_cfg.dynamics_lr,
                                       betas=agent_cfg.dynamics_betas)
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
        return action.detach().cpu().numpy()[0]
    
    @torch.no_grad()
    def log_prob(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        log_prob = self.actor.log_prob(state, action)
        return log_prob.detach().cpu().numpy()

    @torch.no_grad()
    def get_Q(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        value = self.critic(state, action)
        return value.detach().cpu().numpy()[0]

    def update(self, policy_buffer, generate_buffer, expert_buffer, logger, step, epoch):
        # Update dynamics model
        if step % self.dynamics_update_frequency == 0 or generate_buffer.size() == 0:
            self.train_dynamics(policy_buffer, logger, step)
            self.generate_samples(policy_buffer, generate_buffer, epoch)
        
        # update discriminator
        policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
        expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
        losses = self.update_discriminator(policy_batch, expert_batch)

        # update agent
        policy_batch_size = int(self.batch_size * self.real_ratio)
        generate_batch_size = self.batch_size - policy_batch_size
        generate_batch = generate_buffer.get_samples(generate_batch_size, self.device)
        if policy_batch_size > 0:
            policy_batch = get_concat_samples([p[:policy_batch_size] for p in policy_batch], generate_batch)[:5]
        else:
            policy_batch = generate_batch

        losses.update(self.update_critic(policy_batch, expert_batch))

        if self.actor and step % self.actor_update_frequency == 0:
            obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)
            # obs = policy_batch[0]

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

    def train_dynamics(self, policy_buffer, logger, step) -> None:
        data = policy_buffer.sample_all()
        obs, next_obs, action = data[0], data[1], data[2]
        inputs = np.concatenate([obs, action], axis=-1)
        
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * self.holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_obs, train_next_obs, train_action = obs[train_splits], next_obs[train_splits], action[train_splits]
        holdout_obs, holdout_next_obs, holdout_action = obs[holdout_splits], next_obs[holdout_splits], action[holdout_splits]
        train_targets = train_next_obs - train_obs
        holdout_targets = holdout_next_obs - holdout_obs

        self.obs_scaler.fit(train_obs)
        self.action_scaler.fit(train_action)
        self.target_scaler.fit(train_targets)

        train_obs = self.obs_scaler.transform(train_obs)
        train_action = self.action_scaler.transform(train_action)
        holdout_obs = self.obs_scaler.transform(holdout_obs)
        holdout_action = self.action_scaler.transform(holdout_action)
        
        train_inputs = np.concatenate([train_obs, train_action], axis=-1)
        train_targets = self.target_scaler.transform(train_targets)
        holdout_inputs = np.concatenate([holdout_obs, holdout_action], axis=-1)
        holdout_targets = self.target_scaler.transform(holdout_targets)
        
        holdout_losses = [1e10 for i in range(self.num_ensemble)]
        
        data_idxes = np.random.randint(train_size, size=[self.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            train_loss, train_adv_loss = self.learn_dynamics(train_inputs[data_idxes], train_targets[data_idxes], self.batch_size, policy_buffer)
            new_holdout_losses = self.validate_dynamics(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.num_elites]).mean()

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.001:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.dynamics.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if cnt >= self.max_epochs_since_update:
                break

        indexes = self.select_elites(holdout_losses)
        self.dynamics.set_elites(indexes)
        self.dynamics.load_save()
        self.dynamics.eval()
        
        logger.log("train/dynamics_loss", train_loss, step)
        logger.log("train/dynamics_adv_loss", train_adv_loss, step)
        logger.log("train/dynamics_holdout_loss", holdout_loss, step)

    def learn_dynamics(self, inputs, targets, batch_size, policy_buffer):
        self.dynamics.train()
        train_size = inputs.shape[1]
        adv_losses = []
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = torch.from_numpy(inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]).to(self.device)
            targets_batch = torch.from_numpy(targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]).to(self.device)
            policy_batch = policy_buffer.get_samples(self.batch_size, self.device)

            policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch

            if self.opt:
                with torch.no_grad():
                    next_actions, next_policy_log_prob, _ = self.actor.sample(policy_next_obs)
                    next_q = self.critic(policy_next_obs, next_actions)
                    if self.include_ent_in_adv:
                        next_q -= self.alpha * next_policy_log_prob

                    value = policy_reward + (1 - policy_done) * self.gamma * next_q
                    value_baseline = self.critic(policy_obs, policy_action)
                    advantage = value - value_baseline
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                if self.dynamics.deterministic:
                    mean = self.dynamics(inputs_batch)
                    loss = torch.abs(mean - targets_batch).mean(dim=(1, 2)).sum()
                    epsilon = torch.randn_like(mean)
                    next_states_sampled = mean + self.fixed_std * epsilon
                    fixed_std_tensor = torch.tensor(self.fixed_std, device=mean.device, dtype=mean.dtype)
                    log_prob = -0.5 * (
                            ((next_states_sampled - mean) / fixed_std_tensor) ** 2 +
                            2 * torch.log(fixed_std_tensor) +
                            torch.log(torch.tensor(2 * torch.pi, device=mean.device, dtype=mean.dtype))
                    )
                    log_prob = log_prob.sum(dim=-1, keepdim=True).sum(dim=0)
                else:
                    mean, logvar = self.dynamics(inputs_batch)
                    std = torch.exp(0.5 * logvar)
                    dist = torch.distributions.Normal(mean, std)
                    log_prob = dist.log_prob(targets_batch).sum(dim=-1, keepdim=True)
                    inv_var = torch.exp(-logvar)
                    mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
                    var_loss = logvar.mean(dim=(1, 2))
                    loss = mse_loss_inv.sum() + var_loss.sum()
                    loss = loss + self.dynamics.get_decay_loss()
                    loss = loss + self.logvar_loss_coef * self.dynamics.max_logvar.sum() - self.logvar_loss_coef * self.dynamics.min_logvar.sum()

                adv_loss = -(log_prob * advantage[0:log_prob.size(0), :]).mean()
                all_loss = loss + adv_loss * self.adv_weight
            else:
                with torch.no_grad():
                    next_actions, next_policy_log_prob, _ = self.actor.sample(policy_next_obs)
                    next_q = self.critic(policy_next_obs, next_actions)
                    if self.include_ent_in_adv:
                        next_q -= self.alpha * next_policy_log_prob

                if self.dynamics.deterministic:
                    mean = self.dynamics(inputs_batch)
                    loss = torch.abs(mean - targets_batch).mean(dim=(1, 2)).sum()
                else:
                    mean, logvar = self.dynamics(inputs_batch)
                    inv_var = torch.exp(-logvar)
                    mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
                    var_loss = logvar.mean(dim=(1, 2))
                    loss = mse_loss_inv.sum() + var_loss.sum()
                    loss = loss + self.dynamics.get_decay_loss()
                    loss = loss + self.logvar_loss_coef * self.dynamics.max_logvar.sum() - self.logvar_loss_coef * self.dynamics.min_logvar.sum()

                all_loss = loss

            self.dynamics_optimizer.zero_grad()
            all_loss.backward()
            self.dynamics_optimizer.step()

            if self.opt:
                adv_losses.append(adv_loss.item())
            losses.append(loss.item())

        return np.mean(losses), np.mean(adv_losses)

    @ torch.no_grad()
    def validate_dynamics(self, inputs, targets):
        self.dynamics.eval()
        inputs = torch.from_numpy(inputs).to(self.device)
        targets = torch.from_numpy(targets).to(self.device)
        if self.dynamics.deterministic:
            mean = self.dynamics(inputs)
        else:
            mean, _ = self.dynamics(inputs)
        loss = torch.abs(mean - targets).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss

    def select_elites(self, metrics):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.num_elites)]
        return elites

    def set_rollout_length(self, epoch_step):
        rollout_length = (min(max(self.rollout_min_length + (epoch_step - self.rollout_min_epoch)
                                / (self.rollout_max_epoch - self.rollout_min_epoch) * (self.rollout_max_length - self.rollout_min_length),
                                self.rollout_min_length), self.rollout_max_length))
        return int(rollout_length)
    
    def resize_generate_buffer(self, generate_buffer):
        new_size = self.rollout_length * self.generate_batch_size
        self.generate_sample_size[:-1] = self.generate_sample_size[1:]
        self.generate_sample_size[-1] = new_size
        generate_buffer.resize(new_size)

    @torch.no_grad()
    def generate_samples(self, policy_buffer, generate_buffer, epoch):
        self.dynamics.eval()
        self.actor.eval()
        rollout_length = self.set_rollout_length(epoch)
        if rollout_length != self.rollout_length:
            self.resize_generate_buffer(generate_buffer)
            self.rollout_length = rollout_length

        batch = policy_buffer.get_samples(self.generate_batch_size, device=self.device)
        obs = batch[0].detach().cpu().numpy()

        for i in range(rollout_length):
            action = self.choose_action(obs, sample=True)
            obs_ = self.obs_scaler.transform(obs)
            action_ = self.action_scaler.transform(action)
            obs_act = np.concatenate([obs_, action_], axis=-1)
            
            if self.dynamics.deterministic:
                mean = self.dynamics(torch.from_numpy(obs_act).to(self.device))
                mean = self.target_scaler.inverse_transform(mean.detach().cpu().numpy())
                ensemble_samples = obs + mean.astype(np.float32)
            else:
                mean, logvar = self.dynamics(torch.from_numpy(obs_act).to(self.device))
                mean = mean.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()
                std = np.sqrt(np.exp(logvar))
                ensemble_samples = obs + self.target_scaler.inverse_transform(mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

            # choose one model from ensemble
            num_models, batch_size, _ = ensemble_samples.shape
            model_idxs = self.dynamics.random_elite_idxs(batch_size)
            next_obs = ensemble_samples[model_idxs, np.arange(batch_size)]

            for j in range(obs.shape[0]):
                generate_buffer.add((obs[j], next_obs[j], action[j], 0, False))

            obs = next_obs

    # Save model parameters
    def save(self, path, suffix=""):
        dynamics_path = f"{path}{suffix}_dynamics"
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"
        disc_path = f"{path}{suffix}_discriminator"

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.dynamics.state_dict(), dynamics_path)
        torch.save(self.discriminator.state_dict(), disc_path)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        dynamics_path = f'{path}/{self.args.agent.name}{suffix}_dynamics'
        actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
        critic_path = f'{path}/{self.args.agent.name}{suffix}_critic'
        disc_path = f'{path}/{self.args.agent.name}{suffix}_discriminator'
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, disc_path))
        
        if dynamics_path is not None:
            self.dynamics.load_state_dict(torch.load(dynamics_path, map_location=self.device))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        if disc_path is not None:
            self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))