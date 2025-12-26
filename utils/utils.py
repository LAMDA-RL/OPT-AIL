import torch
import numpy as np
from torchvision.utils import make_grid, save_image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def evaluate(actor, env, num_episodes=10, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False

        with eval_mode(actor):
            while not done:
                action = actor.choose_action(state, sample=False)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if 'episode' in info.keys():
                    total_returns.append(info['episode']['r'])
                    total_timesteps.append(info['episode']['l'])

    return total_returns, total_timesteps

  
def get_concat_samples(policy_batch, expert_batch):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch
    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    if isinstance(policy_batch[0], np.ndarray):
        batch_state = np.concatenate([online_batch_state, expert_batch_state], axis=0)
        batch_next_state = np.concatenate(
            [online_batch_next_state, expert_batch_next_state], axis=0)
        batch_action = np.concatenate([online_batch_action, expert_batch_action], axis=0)
        batch_reward = np.concatenate([online_batch_reward, expert_batch_reward], axis=0)
        batch_done = np.concatenate([online_batch_done, expert_batch_done], axis=0)
        is_expert = np.concatenate([np.zeros_like(online_batch_reward),
                            np.ones_like(expert_batch_reward)], axis=0)
    else:
        batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
        batch_next_state = torch.cat(
            [online_batch_next_state, expert_batch_next_state], dim=0)
        batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
        batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
        batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
        is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                            torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert


def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)


def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}