from chainer import cuda
from chainer import function
from chainer import configuration


class WSA(function.Function):

    def __init__(self, wr=0.0):
        self.winner_ratio = wr

    def forward(self, inputs):
        x = inputs[0]
        xp = cuda.get_array_module(*inputs)
        n = x.shape[1]  # number of filters
        
        x_max = x.max(axis=1, keepdims=True)  # (batch, ch, x, y)
        
        if self.winner_ratio < 1.0:
            n_prop = int(n * self.winner_ratio)
            
            if n_prop <= 1:  # Winner Takes All
                mask = (x == x_max)
                norm = x_max
                ret = mask.astype(x.dtype)
            else:  # Winners Share All
                th = xp.sort(x, axis=1)[:,-n_prop,:]
                mask = (x >= xp.expand_dims(th, axis=1))
                x *= mask
                norm = xp.sqrt(xp.sum(x**2, axis=1, keepdims=True))
                ret = x / norm
        else:
            norm = xp.sqrt(xp.sum(x**2, axis=1, keepdims=True))
            ret = x / norm

        # keep mask and scales only when learning is enabled
        if configuration.config.train:
            if self.winner_ratio < 1.0:
                self.mask = mask
            self.scale = norm
        
        return ret,

    def backward(self, inputs, grad_outputs):
        if self.winner_ratio < 1.0:
            return grad_outputs[0] * self.mask * self.scale,
        else:
            return grad_outputs[0] * self.scale,

def wsa(x, wr=0.0):
    """winners share all"""
    return WSA(wr)(x)

#EOF
