import numpy as np

from chainer import cuda
from chainer import function
from chainer import configuration
import chainer.functions as F


class WSA(function.Function):

    def __init__(self, wr=0.0, normalize=True):
        self.winner_ratio = wr
        self.normalize = normalize

    def forward(self, inputs):
        x = inputs[0]
        xp = cuda.get_array_module(*inputs)
        n = x.shape[1]  # number of filters

        if self.winner_ratio < 1.0:
            n_prop = int(n * self.winner_ratio)
            
            if n_prop <= 1:  # Winner Takes All
                x_max = x.max(axis=1, keepdims=True)  # (batch, ch, x, y)
                mask = (x == x_max)
                if self.normalize:
                    norm = x_max
                    ret = mask.astype(x.dtype)
                else:
                    norm = 1.0
                    ret = x * mask
            else:  # Winners Share All
                order = xp.argsort(x, axis=1)
                ranks = xp.argsort(order, axis=1)
                ranks -= n - n_prop - 1
                mask = ranks > 0.0
                if self.normalize:
                    ranks[xp.invert(mask)] = 0.0
                    norm = np.sqrt(np.sum((np.arange(n_prop) + 1)**2))
                    ret = ranks.astype(x.dtype) / norm.astype(x.dtype)
                else:
                    norm = 1.0
                    ret = x * mask
        else:  # pass through
            if self.normalize:
                norm = xp.mean(xp.sqrt(xp.sum(x**2, axis=1, keepdims=True)))
                ret = x / norm
            else:
                norm = 1.0
                ret = x

        # keep mask and scales only when learning is enabled
        if configuration.config.train:
            if self.winner_ratio < 1.0:
                self.mask = mask
            self.scale = norm
        
        return ret,

    def backward(self, inputs, grad_outputs):
        if self.winner_ratio < 1.0:
            return grad_outputs[0] * self.mask, # * self.scale,
        else:
            return grad_outputs[0], # * self.scale,

def wsa(x, wr=0.0, normalize=True):
    """winners share all"""
    return WSA(wr, normalize)(x)

#EOF
