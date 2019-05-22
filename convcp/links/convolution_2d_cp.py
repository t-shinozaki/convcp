from chainer import cuda
#from chainer.functions.connection import convolution_2d_cp
import convcp.functions.convolution_2d_cp as convolution_2d_cp
#from chainer import link
from chainer.links.connection import convolution_2d


class Convolution2DCP(convolution_2d.Convolution2D):

    """Two-dimensional convolutional layer with Winner Takes All dynamics.

    This link wraps the :func:`~chainer.functions.convolution_2d_cp` 
    function and holds the filter weight and bias vector as parameters.

    .. seealso::
       See :func:`chainer.functions.convolution_2d_cp` for the definition of
       two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        with_frad = kwargs.pop('with_frad', True)
        with_bp = kwargs.pop('with_bp', True)
        winner_ratio = kwargs.pop('wr', 0.0)
        conscience_factor = kwargs.pop('cf', 5.0)
        super(Convolution2DCP, self).__init__(in_channels, out_channels, 
                                               ksize, stride, pad, nobias, 
                                               initialW, initial_bias, 
                                               **kwargs)
        #super(Convolution2DCP, self).__init__(*args, **kwargs)
        self.with_frad = with_frad
        self.with_bp = with_bp
        self.winner_ratio = winner_ratio
        self.conscience_factor = conscience_factor
        self.norm()  # normalize initialized weights

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        return convolution_2d_cp.convolution_2d_cp(
            x, self.W, self.b, self.stride, self.pad, 
            with_frad=self.with_frad, with_bp=self.with_bp, 
            wr=self.winner_ratio, cf=self.conscience_factor)

    def norm(self):
        # normalize weights
        xp = self.xp
        w = self.W.data
        norm = xp.sqrt(xp.sum(w**2, axis=(1, 2, 3), keepdims=True))
        self.W.data = w / norm
        # if self.b is not None:
        #     self.b.data /= norm.squeeze()
        return

#EOF
