"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d

from retrieval.utils import pytorch_util as ptu
from retrieval.utils.pytorch_util import PyTorchModule


def identity(x):
    return x


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
        batch_norm=False,
        batch_norm_before_output_activation=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_before_output_activation = batch_norm_before_output_activation
        self.fcs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_size = input_size

        for next_size in hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)

            if self.batch_norm:
                bn = BatchNorm1d(next_size)
                self.batch_norms.append(bn)

        if self.batch_norm_before_output_activation:
            bn = BatchNorm1d(output_size)
            self.batch_norms.append(bn)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    @torch.jit.ignore
    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        if self.batch_norm_before_output_activation:
            preactivation = self.batch_norms[-1](preactivation)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    @torch.jit.export
    def jit_forward(self, input):
        assert self.layer_norm is False
        assert self.batch_norm is False
        assert self.batch_norm_before_output_activation is False
        h = input
        for fc in self.fcs:
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    @torch.jit.ignore
    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

    @torch.jit.export
    def jit_forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().jit_forward(flat_inputs, **kwargs)
