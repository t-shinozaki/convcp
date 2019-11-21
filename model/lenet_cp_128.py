#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
piyo
"""

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import convcp.functions.wsa as FX
import convcp.links.convolution_2d_cp as LX


class CP(chainer.Chain):
    def __init__(self, with_bp=True):
        super(CP, self).__init__(
            #conv1  = LX.Convolution2DCP(  3, 128, 5, with_bp=with_bp),
            conv1s = LX.Convolution2DCP(  3, 128, 3, with_bp=with_bp),
            #conv2l = LX.Convolution2DCP(128, 128, 7, with_bp=with_bp, pad=2),
            conv2  = LX.Convolution2DCP(128, 128, 5, with_bp=with_bp, pad=1),
            conv2s = LX.Convolution2DCP(128, 128, 3, with_bp=with_bp),
        )

    def __call__(self, x, wr=0.0):
        #h_m = FX.wsa(self.conv1(x), wr=wr)
        #h_s  = x[:, :, 1:-1, 1:-1]
        h = FX.wsa(self.conv1s(x), wr=wr)
        #h = F.hstack((h_m, h_s))
        h = F.max_pooling_2d(h, 4, stride=2)
        
        #h_l = FX.wsa(self.conv2l(h), wr=wr)
        h_m = FX.wsa(self.conv2(h), wr=wr)
        h_s = FX.wsa(self.conv2s(h), wr=wr)
        #h = F.hstack((h_l, h_m, h_s))
        h = F.hstack((h_m, h_s))
        h = F.max_pooling_2d(h, 4, stride=2)
        
        return h

    def norm(self):
        #self.conv1.norm()
        self.conv1s.norm()
        #self.conv2l.norm()
        self.conv2.norm()
        self.conv2s.norm()


class BP(chainer.Chain):
    def __init__(self, n=4096):
        super(BP, self).__init__(
            fc3 = L.Linear(None, 10),
        )

    def __call__(self, x):
        #h = self.fc3(F.dropout(x))
        h = self.fc3(x)
        return h

#EOF
