#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
piyo
"""

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

#import frad.functions.wsa as FX
#import frad.links.convolution_2d_cp as LX


class CP(chainer.Chain):
    def __init__(self, with_bp=True):
        super(CP, self).__init__(
        )

    def __call__(self, x, wr=0.00):
        return x

    def norm(self):
        pass


class BP(chainer.Chain):
    def __init__(self, n=100):
        super(BP, self).__init__(
            conv1 = L.Convolution2D(1, 25, 5),
            conv2 = L.Convolution2D(25, 50, 5),
            fc3 = L.Linear(None, n),
            fc4 = L.Linear(n, 10),
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.fc3(h)))
        h = self.fc4(h)
        return h

#EOF
