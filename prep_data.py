#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import chainer


# MNIST
outfile = 'mnist.npz'
(train, test) = chainer.datasets.get_mnist()

X_train = np.array([d[0] for d in train])
X_train = (X_train * 255).astype(np.int).reshape(-1, 1, 28, 28)
T_train = np.array([d[1] for d in train])

X_test = np.array([d[0] for d in test])
X_test = (X_test * 255).astype(np.int).reshape(-1, 1, 28, 28)
T_test = np.array([d[1] for d in test])

# save data
print('save to', outfile)
X_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
np.savez(outfile,
         X_train=X_train, T_train=T_train,
         X_test=X_test, T_test=T_test,
         X_mean=X_mean)


# CIFAR-10
outfile = 'cifar10.npz'
(train, test) = chainer.datasets.get_cifar10()

X_train = np.array([d[0] for d in train])
X_train = (X_train * 255).astype(np.int)
T_train = np.array([d[1] for d in train])

X_test = np.array([d[0] for d in test])
X_test = (X_test * 255).astype(np.int)
T_test = np.array([d[1] for d in test])

print('save to', outfile)
X_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
np.savez(outfile, 
         X_train=X_train, T_train=T_train, 
         X_test=X_test, T_test=T_test, 
         X_mean=X_mean)

#EOF
