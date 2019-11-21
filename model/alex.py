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


class Conv(chainer.Chain):
    def __init__(self, with_bp=True):
        super(Conv, self).__init__(
            pass
        )
        self.insize = 209  # 227

    def __call__(self, x, wr=0.00):
        return x

    def norm(self):
        pass


class FC(chainer.Chain):
    def __init__(self, n=4096):
        super(FC, self).__init__(
            conv1 = L.Convolution2D(3, 96, 11, stride=4,
                                    initialW=Norm_01, bias=0.0),
            conv2 = L.Convolution2D(96, 256, 5, pad=2,
                                    initialW=Norm_01, bias=0.1),
            conv3 = L.Convolution2D(256, 384, 3, pad=1,
                                    initialW=Norm_01, bias=0.0),
            conv4 = L.Convolution2D(384, 384, 3, pad=1,
                                    initialW=Norm_01, bias=0.1),
            conv5 = L.Convolution2D(384, 256, 3, pad=1,
                                    initialW=Norm_01, bias=0.1),
            fc6 = L.Linear(None, n, initialW=Norm_005, bias=0.1),
            fc7 = L.Linear(n, n, initialW=Norm_005, bias=0.1),
            fc8 = L.Linear(n, 1000, initialW=Norm_01, bias=0.0),
        )

    def __call__(self, x):
        h = F.local_response_normalization(F.relu(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(F.relu(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h

#EOF
