import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from .scaler import StandardScaler
from .net import *


class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dims,
        dropout,
        num_ensemble,
        weight_decays,
        num_elites,
        deterministic,
        args
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites

        self.activation = Swish()

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)

        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay, dropout))
        self.backbones = nn.ModuleList(module_list)

        self.deterministic = deterministic

        if deterministic:
            self.output_layer = EnsembleLinear(
                hidden_dims[-1],
                obs_dim,
                num_ensemble,
                weight_decays[-1]
            )
        
        else:
            self.output_layer = EnsembleLinear(
                hidden_dims[-1],
                2 * obs_dim,
                num_ensemble,
                weight_decays[-1]
            )
            self.register_parameter(
                "max_logvar",
                nn.Parameter(torch.ones(obs_dim) * 0.5, requires_grad=True)
            )
            self.register_parameter(
                "min_logvar",
                nn.Parameter(torch.ones(obs_dim) * -10, requires_grad=True)
            )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )
        
    def forward(self, obs_action):
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        if self.deterministic:
            mean = self.output_layer(output)
            return mean
        else:
            mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
            logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
            return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    
    def compute_mean_std(self, obs_action: torch.Tensor, use_elites: bool = True) -> torch.Tensor:
        """
        Compute the standard deviation of ensemble model predictions: std(μθi(s,a))
        
        This function computes the epistemic uncertainty by measuring the disagreement
        among ensemble models (or elite models) on their predictions.
        
        Args:
            obs_action: Input tensor of shape [batch_size, obs_dim+action_dim] or 
                       [num_ensemble, batch_size, obs_dim+action_dim].
                       If 2D, the input will be broadcasted to all ensemble models.
            use_elites: If True, only use elite models for computing std. 
                       If False, use all ensemble models.
        
        Returns:
            std: Standard deviation of ensemble predictions, shape [batch_size, obs_dim].
                 Higher values indicate higher epistemic uncertainty.
        """
        # Get predictions from all models
        # forward() always returns shape [num_ensemble, batch_size, obs_dim]
        if self.deterministic:
            mean = self.forward(obs_action)  # shape: [num_ensemble, batch_size, obs_dim]
        else:
            mean, _ = self.forward(obs_action)  # shape: [num_ensemble, batch_size, obs_dim]
        
        # Select which models to use
        if use_elites:
            elite_idxs = self.elites.data.cpu().numpy()
            mean = mean[elite_idxs]  # shape: [num_elites, batch_size, obs_dim]
        
        # Compute standard deviation across ensemble dimension
        # mean shape: [num_ensemble/num_elites, batch_size, obs_dim]
        std = torch.std(mean, dim=0)  # shape: [batch_size, obs_dim]
        
        return std
    