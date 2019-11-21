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
            conv1xl = LX.Convolution2DCP(  3, 256,11, stride=4, pad=1, 
                                          with_bp=with_bp),
            conv1   = LX.Convolution2DCP(  3, 256, 9, stride=4, 
                                          with_bp=with_bp),
            conv1m  = LX.Convolution2DCP(  3, 256, 7, stride=4, 
                                          with_bp=with_bp),
            conv1s  = LX.Convolution2DCP(  3, 256, 5, stride=4, 
                                          with_bp=with_bp),
            conv1xs = LX.Convolution2DCP(  3, 256, 3, stride=4, 
                                          with_bp=with_bp),
            conv2  = LX.Convolution2DCP(1280,1024, 5, pad=1, with_bp=with_bp),
            conv2s = LX.Convolution2DCP(1280,1024, 3, with_bp=with_bp),
            conv3  = LX.Convolution2DCP(2048,1024, 5, pad=1, with_bp=with_bp),
            conv3s = LX.Convolution2DCP(2048,1024, 3, with_bp=with_bp),
        )
        self.insize = 227

    def __call__(self, x, wr=0.0):
        h_xl  = FX.wsa(self.conv1xl(x), wr=wr)
        h_l  = FX.wsa(self.conv1(x), wr=wr)
        h_m  = x[:, :, 1:-1, 1:-1]
        h_m  = FX.wsa(self.conv1m(h_m), wr=wr)
        h_s  = x[:, :, 2:-2, 2:-2]
        h_s  = FX.wsa(self.conv1s(h_s), wr=wr)
        h_xs = x[:, :, 3:-3, 3:-3]
        h_xs = FX.wsa(self.conv1xs(h_xs), wr=wr)
        h = F.hstack((h_xl, h_l, h_m, h_s, h_xs))
        h = F.max_pooling_2d(h, 3, stride=2)
        
        h_m = FX.wsa(self.conv2(h), wr=wr)
        h_s = FX.wsa(self.conv2s(h), wr=wr)
        h = F.hstack((h_m, h_s))
        h = F.max_pooling_2d(h, 3, stride=2)
        
        h_m = self.conv3(h)
        h_s = self.conv3s(h)
        h = F.hstack((h_m, h_s))
        h = F.max_pooling_2d(h, 3, stride=2)
        return h

    def norm(self):
        self.conv1xl.norm()
        self.conv1.norm()
        self.conv1m.norm()
        self.conv1s.norm()
        self.conv1xs.norm()
        self.conv2.norm()
        self.conv2s.norm()
        self.conv3.norm()
        self.conv3s.norm()


class BP(chainer.Chain):
    def __init__(self, n=4096):
        super(BP, self).__init__(
            #fc4 = L.Linear(None, n),
            #fc5 = L.Linear(n, n),
            #fc6 = L.Linear(n, 1000),
            fc4 = L.Linear(None, 1000),
        )

    def __call__(self, x):
        #h = F.dropout(F.relu(self.fc4(x)))
        #h = F.dropout(F.relu(self.fc5(h)))
        #h = self.fc6(h)
        h = self.fc4(x)
        return h

#EOF
