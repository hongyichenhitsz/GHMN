from typing import Optional

from einops.layers.torch import Rearrange
from torch import nn, Tensor
import pdb
import math
from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import init

from torch import nn

_torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity'
}

def get_layer_activation(activation: Optional[str] = None):
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation in _torch_activations_dict:
        return getattr(nn, _torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")


class Dense(nn.Module):
    r"""A simple fully-connected layer implementing

    .. math::

        \mathbf{x}^{\prime} = \sigma\left(\boldsymbol{\Theta}\mathbf{x} +
        \mathbf{b}\right)

    where :math:`\mathbf{x} \in \mathbb{R}^{d_{in}}, \mathbf{x}^{\prime} \in
    \mathbb{R}^{d_{out}}` are the input and output features, respectively,
    :math:`\boldsymbol{\Theta} \in \mathbb{R}^{d_{out} \times d_{in}} \mathbf{b}
    \in \mathbb{R}^{d_{out}}` are trainable parameters, and :math:`\sigma` is
    an activation function.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        bias (bool, optional): If :obj:`True`, then the bias vector is used.
            (default: :obj:`True`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 bias: bool = True):
        super(Dense, self).__init__()
        self.affinity = nn.Linear(input_size, output_size, bias=bias)
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def reset_parameters(self) -> None:
        """"""
        self.affinity.reset_parameters()

    def forward(self, x):
        """"""
        out = self.activation(self.affinity(x))
        return self.dropout(out)
    
class MultiLinear(nn.Module):
    r"""Applies linear transformations with different weights to the different
    instances in the input data.

    .. math::

        \mathbf{X}^{\prime} = [\boldsymbol{\Theta}_i \mathbf{x}_i +
        \mathbf{b}_i]_{i=0,\ldots,N}

    Args:
        in_channels (int): Size of instance input sample.
        out_channels (int): Size of instance output sample.
        n_instances (int): The number :math:`N` of parallel linear
            operations. Each operation has different weights and biases.
        instance_dim (int or str): Dimension of the instances (must match
            :attr:`n_instances` at runtime).
            (default: :obj:`-2`)
        channel_dim (int or str): Dimension of the input channels.
            (default: :obj:`-1`)
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each instance.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)

    Examples:

        >>> m = MultiLinear(20, 32, 10, pattern='t n f', instance_dim='n')
        >>> input = torch.randn(64, 12, 10, 20)  # shape: [b t n f]
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([64, 24, 10, 32])
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_instances: int,
                 *,
                 ndim: int = None,
                 pattern: str = None,
                 instance_dim: Union[int, str] = -2,
                 channel_dim: Union[int, str] = -1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_instances = n_instances

        self.ndim = ndim
        self.instance_dim = instance_dim
        self.channel_dim = channel_dim

        # initialize by pattern, e.g.:
        #   pattern='t n f', instance_dim='n', instance_dim=-1
        #   pattern='t n f', instance_dim=-2, instance_dim=-1
        if pattern is not None:
            pattern = pattern.replace(' ', '')
            self.instance_dim = instance_dim if isinstance(instance_dim, str) \
                else pattern[instance_dim]
            self.channel_dim = channel_dim if isinstance(channel_dim, str) \
                else pattern[channel_dim]
            self.einsum_pattern = self._compute_einsum_pattern(pattern=pattern)
            self.bias_shape = self._compute_bias_shape(pattern=pattern)
            self.reshape_bias = False
        # initialize negative dim indexing (default), e.g.:
        #   instance_dim=-2, instance_dim=-1 (pattern=None, ndim=None)
        elif ndim is None and instance_dim < 0 and channel_dim < 0:
            ndim = abs(min(instance_dim, channel_dim))
            self.einsum_pattern = self._compute_einsum_pattern(ndim)
            self.bias_shape = self._compute_bias_shape(ndim)
            self.reshape_bias = False
        # initialize with ndim and dim (positive/negative) indexing, e.g.:
        #   ndim=3, instance_dim=1, instance_dim=-1
        elif ndim is not None:
            # initialize with lazy ndim calculation, e.g.:
            #   ndim=-1, instance_dim=1, instance_dim=-1
            if ndim < 0:
                # lazy initialize einsum pattern
                self.einsum_pattern = None
                self.bias_shape = (n_instances, out_channels)
                self.reshape_bias = True
                self._hook = self.register_forward_pre_hook(
                    self.initialize_module)
            else:
                self.einsum_pattern = self._compute_einsum_pattern(ndim)
                self.bias_shape = self._compute_bias_shape(ndim)
                self.reshape_bias = False
        # cannot initialize if all:
        #   1. pattern is None
        #   2. ndim is None and instance_dim >= 0 or channel_dim >= 0
        else:
            raise ValueError("One of 'pattern' or 'ndim' must be given if one "
                             "of 'instance_dim' or 'channel_dim' is positive.")

        self.weight: nn.Parameter = nn.Parameter(
            torch.empty((n_instances, in_channels, out_channels),
                        **factory_kwargs))

        if bias:
            self.bias: nn.Parameter = nn.Parameter(
                torch.empty(*self.bias_shape, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        """"""
        return 'in_channels={}, out_channels={}, n_instances={}'.format(
            self.in_channels, self.out_channels, self.n_instances)

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_channels)
        init.uniform_(self.weight.data, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias.data, -bound, bound)

    def _compute_bias_shape(self, ndim: int = None, pattern: str = None):
        if ndim is not None:
            bias_shape = [1] * ndim
            bias_shape[self.instance_dim] = self.n_instances
            bias_shape[self.channel_dim] = self.out_channels
        elif pattern is not None:
            pattern = pattern.replace(' ', '')
            bias_shape = []
            for token in pattern:
                if token == self.channel_dim:
                    bias_shape.append(self.out_channels)
                elif token == self.instance_dim:
                    bias_shape.append(self.n_instances)
                else:
                    bias_shape.append(1)
        else:
            raise ValueError("One of 'pattern' or 'ndim' must be given.")
        return tuple(bias_shape)

    def _compute_einsum_pattern(self, ndim: int = None, pattern: str = None):
        if ndim is not None:
            pattern = [chr(s + 97) for s in range(ndim)]  # 'a', 'b', ...
            pattern[self.instance_dim] = 'x'
            pattern[self.channel_dim] = 'y'
            input_pattern = ''.join(pattern)
            pattern[self.channel_dim] = 'z'
            output_pattern = ''.join(pattern)
            weight_pattern = 'xyz'
        elif pattern is not None:
            input_pattern = pattern.replace(' ', '')
            output_pattern = input_pattern.replace(self.channel_dim, 'z')
            weight_pattern = f'{self.instance_dim}{self.channel_dim}z'
        else:
            raise ValueError("One of 'pattern' or 'ndim' must be given.")
        return f"...{input_pattern},{weight_pattern}->...{output_pattern}"

    @torch.no_grad()
    def initialize_module(self, module, input):
        self.ndim = input[0].ndim
        self.einsum_pattern = self._compute_einsum_pattern(self.ndim)
        self.bias_shape = self._compute_bias_shape(self.ndim)
        self._hook.remove()
        delattr(self, '_hook')

    def forward(self, input: Tensor) -> Tensor:
        r"""Compute :math:`\mathbf{X}^{\prime} =
        [\boldsymbol{\Theta}_i \mathbf{x}_i + \mathbf{b}_i]_{i=0,\ldots,N}`"""
        out = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            if self.reshape_bias:
                out = out + self.bias.view(*self.bias_shape).contiguous()
            else:
                out = out + self.bias
        return out
    
class Readout(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int = None,
                 horizon: int = None,
                 multi_readout: bool = False,
                 n_hidden_layers: int = 0,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon

        readout_input_size = input_size if n_hidden_layers == 0 else hidden_size

        # Add layers in reverse order, starting from the readout
        layers = []

        # Last linear layer
        if not multi_readout:
            # after transformation, reshape to have "... time nodes features"
            if horizon is not None:
                layers.append(Rearrange('... n (h f) -> ... h n f', h=horizon))
            else:
                horizon = 1
            layers.append(nn.Linear(readout_input_size, output_size * horizon))
        else:
            assert horizon is not None
            layers.append(MultiLinear(readout_input_size, output_size,
                                      n_instances=horizon,
                                      instance_dim=-3))

        # Optionally add hidden layers
        for i in range(n_hidden_layers):
            layers.append(
                Dense(input_size if i == (n_hidden_layers - 1) else hidden_size,
                      output_size=hidden_size,
                      activation=activation,
                      dropout=dropout)
            )

        self.mlp = nn.Sequential(*reversed(layers))

    def forward(self, x: Tensor):
        # x: [*, nodes, features] or x: [*, horizon, nodes, features]
        x = self.mlp(x)
        return x


class AttentionReadout(nn.Module):

    def __init__(self,
                 input_size: int,
                 dim_size: int,  # num elements on which to apply attention
                 horizon: int = None,
                 dim: int = -2,  # dimension along which to apply attention
                 output_size: int = None,
                 hidden_size: int = None,
                 mask_size: int = None,
                 fully_connected: bool = False,
                 multi_step_scores: bool = True,
                 ff_layers: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size
        self.horizon = horizon
        self.fully_connected = fully_connected
        self.multi_step_scores = multi_step_scores

        horizon = horizon or 1
        out_f = horizon if multi_step_scores else 1

        if fully_connected:
            # b n l f -> b n (l h) f
            self.lin_scores_state = MultiLinear(dim_size, dim_size * out_f,
                                                n_instances=input_size,
                                                instance_dim=-1,
                                                channel_dim=-2)
        else:
            # b n l f -> b n l h
            self.lin_scores_state = nn.Linear(input_size, out_f)

        if mask_size is not None:
            # mask: [batch nodes features]
            if fully_connected:
                self.lin_scores_mask = nn.Sequential(
                    nn.Linear(mask_size, dim_size * out_f * input_size),
                    Rearrange('b n (l h f) -> b n (l h) f',
                              l=dim_size, f=out_f),
                )
            else:
                self.lin_scores_mask = nn.Sequential(
                    nn.Linear(mask_size, dim_size * out_f),
                    Rearrange('b n (l h) -> b n l h', l=dim_size, f=out_f),
                )
        else:
            self.register_parameter('lin_scores_mask', None)

        # Rearrange scores to have the same shape as the input
        self.rearrange = nn.Identity()
        if multi_step_scores and fully_connected:
            self.rearrange = Rearrange('b n (l h) f -> b h n l f', l=dim_size)
        elif multi_step_scores:
            self.rearrange = Rearrange('b n l h -> b h n l 1', l=dim_size)

        if output_size is not None:
            self.readout = Readout(input_size=input_size,
                                   hidden_size=hidden_size,
                                   output_size=output_size,
                                   horizon=self.horizon,
                                   multi_readout=self.multi_step_scores,
                                   n_hidden_layers=ff_layers - 1,
                                   activation=activation,
                                   dropout=dropout)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # mask: [batch, nodes, features]

        if self.dim != -2:
            x = x.movedim(self.dim, -2)
        # x: [batch, nodes, layers, features]

        # Compute scores from features with a linear reduction
        scores = self.lin_scores_state(x)  # -> [batch, nodes, layers, out_f]

        # Optionally add mask information inside the score
        if self.lin_scores_mask is not None:
            scores = scores + self.lin_scores_mask(mask)

        # Normalize scores with softmax
        scores = self.rearrange(scores)#torch.Size([1, 20531, 2, 1])
        alpha = scores.softmax(-2)  # -> [batch, *, nodes, layers, features]

        # Aggregate along layers dimension (self.dim) according to the scores
        if self.multi_step_scores:
            x = x.unsqueeze(-4)  # apply different score at each (layer, step)
        x = (x * alpha).sum(-2)  # -> [batch, *, nodes, features]

        if self.dim != -2:
            alpha = alpha.movedim(-2, self.dim)
        alpha = alpha.mean(-1)  # ... l 1 -> ... l ...

        if self.readout is None:
            return x, alpha

        return self.readout(x), x, alpha
