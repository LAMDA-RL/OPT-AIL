import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Swish(nn.Module):
	def __init__(self) -> None:
		super(Swish, self).__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x * torch.sigmoid(x)
		return x


def soft_clamp(
	x : torch.Tensor,
	_min: Optional[torch.Tensor] = None,
	_max: Optional[torch.Tensor] = None
) -> torch.Tensor:
	# clamp tensor values while mataining the gradient
	if _max is not None:
		x = _max - F.softplus(_max - x)
	if _min is not None:
		x = _min + F.softplus(x - _min)
	return x


# Initialize Policy weights
def orthogonal_init_(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


def weighted_softmax(x, weights):
	x = x - torch.max(x, dim=0)[0]
	return weights * torch.exp(x) / torch.sum(
		weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(tau * param.data +
								(1 - tau) * target_param.data)


def hard_update(source, target):
	for param, target_param in zip(source.parameters(), target.parameters()):
		target_param.data.copy_(param.data)


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


class MLP(nn.Module):
	def __init__(self,
				 input_dim,
				 hidden_dim,
				 output_dim,
				 hidden_depth,
				 output_mod=None):
		super().__init__()
		self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
						 output_mod)
		self.apply(weight_init)

	def forward(self, x):
		return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk


class EnsembleLinear(nn.Module):
	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		num_ensemble: int,
		weight_decay: float = 0.0,
		dropout: float = 0.0,
	) -> None:
		super().__init__()

		self.num_ensemble = num_ensemble

		self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
		self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

		nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

		self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
		self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

		self.weight_decay = weight_decay
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(output_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		weight = self.weight
		bias = self.bias
		if self.training:
			x = self.dropout(x)

		if len(x.shape) == 2:
			x = torch.einsum('ij,bjk->bik', x, weight)
		else:
			x = torch.einsum('bij,bjk->bik', x, weight)

		x = x + bias

		x = self.layer_norm(x)

		return x

	def load_save(self) -> None:
		self.weight.data.copy_(self.saved_weight.data)
		self.bias.data.copy_(self.saved_bias.data)

	def update_save(self, indexes) -> None:
		self.saved_weight.data[indexes] = self.weight.data[indexes]
		self.saved_bias.data[indexes] = self.bias.data[indexes]

	def get_decay_loss(self) -> torch.Tensor:
		decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
		return decay_loss


def ensemble_mlp(input_dim, hidden_dim, output_dim, hidden_depth, ensemble_num, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, " \
			   f"out_features={self.out_features}, " \
			   f"bias={self.bias is not None}{repr_dropout}, " \
			   f"act={self.act.__class__.__name__})"

