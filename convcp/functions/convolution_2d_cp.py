import numpy as np

from chainer import cuda
from chainer import function
from chainer import configuration
from chainer.utils import argument
from chainer.utils import conv
from chainer.functions.connection import convolution_2d


class Convolution2DCPFunction(convolution_2d.Convolution2DFunction):

    def __init__(self, stride=1, pad=0, cover_all=False, **kwargs):
        self.with_frad = kwargs.pop('with_frad', True)
        self.with_bp = kwargs.pop('with_bp', False)
        self.winner_ratio = kwargs.pop('wr', 0.0)
        self.conscience_factor = kwargs.pop('cf', 5.0)
        super(Convolution2DCPFunction, self).__init__(stride, pad, cover_all,
                                                       **kwargs)

    def forward(self, inputs):
        y, = super(Convolution2DCPFunction, self).forward(inputs)

        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs
        
        xp = cuda.get_array_module(*inputs)
        n = y.shape[1]  # number of filters

        if self.winner_ratio < 1.0:
            n_learn = int(n * self.winner_ratio)
            if n_learn <= 1:  # Winner Takes All
                mask = (y == y.max(axis=1, keepdims=True))
            else:  # Winners Share All
                th = xp.sort(y, axis=1)[:,-n_learn,:]
                mask = (y >= xp.expand_dims(th, axis=1))
        else:
            mask = xp.ones_like(y, dtype=bool)

        # keep mask and scales only when learning is enabled
        if configuration.config.train:
            self.mask = mask
        
        # extract conscience factors from return values of inherited func
        return y - b[xp.newaxis, :, xp.newaxis, xp.newaxis],

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]

        if not self.with_bp:
            xp = cuda.get_array_module(gy)
            gy = xp.zeros_like(gy)

        return super(Convolution2DCPFunction, self).backward(inputs, (gy,))

    def forward_grad(self, inputs, rho):
        x, W = inputs[:2]
        xp = cuda.get_array_module(*inputs)
        rho = xp.array(rho, dtype=W.dtype)

        # regenerate input matrix
        _, _, kh, kw = W.shape
        if isinstance(x, cuda.ndarray):
            col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        else:
            col = conv.im2col_cpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)

        if rho.ndim > 0:  # if rho is vector, prep for broadcasting
            rho = rho[:, xp.newaxis, xp.newaxis, xp.newaxis]
        
        gW = xp.tensordot(self.mask * rho, -1.0 * col, 
                          ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)

        # calc conscience factors [DeSieno, 1988] as bias of conv layer
        n = self.mask.shape[1]
        v = self.mask.sum(axis=(0, 2, 3), dtype=W.dtype)
        gb = -1.0 * self.conscience_factor * (1.0 / n - v / v.sum())

        return gW, gb


def convolution_2d_cp(x, W, b=None, stride=1, pad=0, cover_all=False, 
                       **kwargs):
    """Two-dimensional convolution function with competitive learning.

    Returns:
        ~chainer.Variable: Output variable.
    
    .. seealso:: :class:`~chainer.links.Convolution2DCP`

    """
    fnode = Convolution2DCPFunction(stride, pad, cover_all, **kwargs)

    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y

#EOF
