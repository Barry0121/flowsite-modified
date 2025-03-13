# this file uses classes from https://github.com/aqlaboratory/openfold
import importlib
import math
from typing import Callable, Optional

import numpy as np
import torch
from scipy.stats import truncnorm
from torch import nn

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

class Encoder(torch.nn.Module):
    """
    Embeds categorical features into a continuous vector space.

    This module takes one-hot or integer encoded categorical features and embeds
    them into a continuous vector space. Multiple categorical features are embedded
    separately and then summed to form the final embedding.

    Args:
        emb_dim (int): Embedding dimension for each categorical feature
        feature_dims (list): List of integers representing the number of possible
                            values for each categorical feature
    """
    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims is a list with the length of each categorical feature
        super(Encoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding

def _prod(nums):
    """
    Calculates the product of all elements in a sequence.

    Args:
        nums (sequence): Sequence of numbers

    Returns:
        Product of all numbers in the sequence
    """
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    """
    Calculates the fan-in, fan-out, or fan-average for weight initialization.

    Args:
        linear_weight_shape (tuple): Shape of the weights tensor (fan_out, fan_in)
        fan (str): Type of fan to calculate - "fan_in", "fan_out", or "fan_avg"

    Returns:
        int: The calculated fan value
    """
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    """
    Initializes weights using a truncated normal distribution.

    Samples from a truncated normal distribution and scales the values
    based on the fan value.

    Args:
        weights (torch.Tensor): Tensor to initialize
        scale (float): Scaling factor. Default: 1.0
        fan (str): Type of fan to use for scaling. Default: "fan_in"
    """
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))

def lecun_normal_init_(weights):
    """
    Applies LeCun normal initialization to weights.

    LeCun initialization scales the truncated normal distribution
    with a factor of 1.0.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    """
    Applies He normal initialization to weights.

    He initialization scales the truncated normal distribution
    with a factor of 2.0, optimized for ReLU activations.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    """
    Applies Glorot/Xavier uniform initialization to weights.

    Initializes weights using a uniform distribution scaled
    by the average of fan-in and fan-out.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    """
    Initializes all weights to zero.

    Typically used for the final layer of a network to zero
    out initial outputs.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    """
    Initializes weights for gating mechanisms to zero.

    Used for gating layers where the initial behavior should
    be to pass through inputs unchanged.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    """
    Applies Kaiming normal initialization with linear scaling.

    Initializes weights using the Kaiming normal distribution
    with the "linear" nonlinearity setting.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    """
    Initializes weights for invariant point attention.

    Sets all weights to softplus_inverse_1 (0.541324854612918),
    a specific value used in the IPA architecture.

    Args:
        weights (torch.Tensor): Tensor to initialize
    """
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

class LayerNorm(nn.Module):
    """
    Custom Layer Normalization with bfloat16 handling.

    Implements layer normalization with special handling for bfloat16 precision,
    particularly useful when working with mixed precision training or DeepSpeed.

    Args:
        c_in (int): Number of input features
        eps (float): Small constant for numerical stability. Default: 1e-5
    """
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = (
            deepspeed_is_installed and deepspeed.utils.is_initialized()
        )
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out