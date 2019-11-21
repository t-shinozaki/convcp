#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: <2019-11-19 13:51:08 tshino>
#
"""
Train CNN with various structures

Copyright (C) 2016-19 Takashi Shinozaki
"""

from __future__ import print_function
import argparse
import os, sys, six
import importlib
from datetime import datetime
from multiprocessing import Process, Pool

from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#import myloss as FX
import convcp.add_forward_grad


parser = argparse.ArgumentParser(description='CNN for image discrimination')
parser.add_argument('--unsupervised', '-u', action='store_true', 
                    help='Unsupervised learning mode')
parser.add_argument('--kill', '-k', action='store_true', 
                    help='kill when the std of outputs fall below threshold')
parser.add_argument('--processes', '-j', default=4, type=int,
                    help='# of processs for loading images')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--batchsize', '-B', default=100, type=int,
                    help='# of samples in a mini-batch')
parser.add_argument('--epoch', '-E', default=1, type=int,
                    help='# of epochs')
parser.add_argument('--startepoch', '-S', default=1, type=int,
                    help='Initial epoch #')
parser.add_argument('--iteration', '-i', default=500, type=int,
                    help='# of iteration for each epoch')
parser.add_argument('--checkinterval', '-c', default=100, type=int,
                    help='# of iteration for each validation')
parser.add_argument('--plotepoch', '-C', default=25, type=int,
                    help='# of epochs for plot the kernels')
parser.add_argument('--plotkernel', '-K', default='',
                    help='Kernel name for plotting')
parser.add_argument('--saveepoch', '-s', default=200, type=int,
                    help='# of epochs for saving the state')
parser.add_argument('--decay', '-d', nargs='?', const=0.0001, type=float,
                    help='Weight decay for optimizer (chainer def: 0.0001)')
parser.add_argument('--adam', '-a', nargs='?', const=0.001, type=float,
                    help='Learning Rate for Adam (chainer def: 0.001)')
parser.add_argument('--adadelta', action='store_true', 
                     help='Use AdaDelta for optimization')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for SGD (chainer def: 0.9)')
parser.add_argument('--lr', '-l', default=0.01, type=float,
                    help='Learning Rate for SGD (chainer def: 0.01)')
parser.add_argument('--rho', '-r', default=1e-2, type=float,
                    help='Learning Rate for frad (def: 1e-3?)')
parser.add_argument('--outputdir', '-O', default='./out_',
                    help='Directory name for output files')
parser.add_argument('--netmodeldir', '-N', default='./model',
                    help='Directory name for network model files')
parser.add_argument('--model', '-M', default='lenet5',
                    help='Model module name without .py')
parser.add_argument('--weight', '-W', default='',
                    help='Initialize weights from given file')
parser.add_argument('--fcweight', '-F', default='',
                    help='Initialize FC weights from given file')
parser.add_argument('--datafile', '-D', default='cifar10.npz',
                    help='Data filename for training as npz file')
parser.add_argument('--labellist', '-L', 
                    help='List filename for enabled label')
parser.add_argument('--restrict', '-R', default=0, type=int,
                    help='Ristricted # of training dataset')
parser.add_argument('--bias', '-b', default=0.0, type=float,
                    help='Ratio of error signals for incorrect answers')
#parser.add_argument('--ace', action='store_true', 
#                    help='Use Averaged Cross Entropy instead of MSE')
parser.add_argument('--mse', action='store_true', 
                    help='Use MSE instead of Averaged Cross Entropy')
parser.add_argument('--wr', '-w', default=0.0, type=float)
parser.add_argument('--no-bp', '-n', action='store_true')
parser.add_argument('--fcunits', '-f', default=4096, type=int,
                    help='# of units for the 1st fully connected layer')
args = parser.parse_args()


# parameters
start_epoch = args.startepoch
n_epoch = args.epoch
n_iter = args.iteration
batchsize = args.batchsize
t_bias = args.bias

n_test = 10000  # 1000
n_val = 2000
out_n = 10 # old: 50
chk_trial_interval = args.checkinterval
plot_epoch_interval = args.plotepoch
save_epoch_interval = args.saveepoch
th_dom_answer = 0.5

out_dir = args.outputdir


# util functions
def plot_rf(w, fname):  # 100.0 or 50.0 or 20.0 or 5.0
    n, n_ch = w.shape[:2]
    n_sqrt = np.ceil(np.sqrt(n)).astype(np.int32)
    fig_size = 1.6 * n_sqrt # inch
    plt.figure(figsize=(fig_size, fig_size))
    
    # calc the intesity scale factor
    scale = 1 / (w[:, 0::3, :, :].mean(axis=1).std() * 4.0)
     
    for i in six.moves.range(n):
        plt.subplot(n_sqrt, n_sqrt, i+1)
        if n_ch == 1:
            img = w[i, 0, :, :]
            plt.imshow(img, cmap='gray', interpolation='bicubic')  # 'catrom'
        else:
            r = w[i, 0::3, :, :].mean(axis=0, keepdims=True)
            g = w[i, 1::3, :, :].mean(axis=0, keepdims=True)
            if n_ch > 2:
                b = w[i, 2::3, :, :].mean(axis=0, keepdims=True)
            else:
                b = g
            img = np.concatenate((r, g, b)).transpose(1, 2, 0) * scale
            # img = img[::-1]  # convert BGR to RGB
            img = (img + 1.0) / 2.0  # transform [-1.0, 1.0] to [0, 1.0]
            img = img.clip(0.0, 1.0)  # clipping pixel values
            plt.imshow(img, interpolation='bicubic')  # 'catrom'
        plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', dpi=50)
    return

def label2array(m, n, label):
    if p_lose == 0.0:
        t = mp.eye(m, n, dtype=np.float32)[label]  # 1-hot vec
    else:
        t = mp.ones((m, n), dtype=np.float32) * p_lose
        t[range(batchsize), label] = p_win
    return t


# load dataset
print('loading dataset:', args.datafile)
sys.stdout.flush()
npzfile = np.load(args.datafile)
X_train = npzfile['X_train'].astype(np.float32) / 255.0
X_test  = npzfile['X_test'].astype(np.float32) / 255.0
T_train = npzfile['T_train'].astype(np.int32)
T_test  = npzfile['T_test'].astype(np.int32)
X_mean  = npzfile['X_mean'].astype(np.float32) / 255.0
X_train -= X_mean
X_test -= X_mean

# use a part of labels
if args.labellist:
    llst = []
    with open(args.labellist, 'rt') as f:
        for line in f:
            llst.append(line.strip())
    mask_train = np.isin(T_train, llst)
    X_train = X_train[mask_train]
    T_train = T_train[mask_train]
    mask_test = np.isin(T_test, llst)
    X_test = X_test[mask_test]
    T_test = T_test[mask_test]

# use a part of training samples
if args.restrict:
    ix = np.random.permutation(len(T_train))[:args.restrict]
    X_train = X_train[ix]
    T_train = T_train[ix]

n_data_train = len(T_train)
n_data_test = len(T_test)
n_train = n_iter * batchsize
n_train_all = (n_train // n_data_train + 1) * n_data_train
print('loaded:', end='')
print(' N_train = ', n_data_train, ', N_test = ', n_data_test, sep='')
sys.stdout.flush()


# model definition
print('loading model module:', args.model)
sys.stdout.flush()
sys.path.append(args.netmodeldir)
model_module = importlib.import_module(args.model)
model = model_module.CP(with_bp=not args.no_bp)
fc = model_module.BP(args.fcunits)

# GPU check
if args.gpu >= 0:
    print('GPU mode, ID =', args.gpu)
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    fc.to_gpu()
    mp = cuda.cupy
else:
    print('CPU mode')
    mp = np

# mode check
if args.unsupervised:
    print('Unsupervised learning mode', end='')
else:
    print('Supervised learning mode', end='')
if args.mse:
    print(' with MSE loss function')
else:
    print(' with ACE loss function')
print('Iteration =', n_iter, end='')
print(', mini-batch size =', batchsize)
print('Total:', n_iter * batchsize, 'samples/epoch')

# Setup optimizer
if args.adam and args.adam > 0.0:
    print('Adam optimizer is selected: alpha =', args.adam)
    opt = optimizers.Adam(alpha=args.adam)  # def: alpha=0.001
    opt_fc = optimizers.Adam(alpha=args.adam)
elif args.adadelta:
    print('AdaDelta optimizer is selected.')
    opt = optimizers.AdaDelta()  # def: rho=0.95, eps=1e-06
    opt_fc = optimizers.AdaDelta()
elif args.momentum > 0.0:
    print('Momentum SGD optimizer is selected:', end='')
    print(' lr = {}, momentum = {}'.format(args.lr, args.momentum))
    opt = optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    opt_fc = optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
else:
    print('Standard SGD optimizer is selected: lr =', args.lr)
    opt = optimizers.SGD(lr=args.lr)  # def: lr=0.01
    opt_fc = optimizers.SGD(lr=args.lr)
if not args.no_bp:
    opt.setup(model)
opt_fc.setup(fc)
if args.decay:
    print('Weight decay:', args.decay)
    if not args.no_bp:
        opt.add_hook(chainer.optimizer.WeightDecay(args.decay))  # def: 0.0001
    opt_fc.add_hook(chainer.optimizer.WeightDecay(args.decay))
print('BP in competitive learning layers:', (not args.no_bp))
#print('Competitive learning: rho =', args.rho)

# load weight data from hdf5 file
if args.weight:
    print('Load model weight:', args.weight)
    fname = os.path.join(args.netmodeldir, args.weight)
    serializers.load_hdf5(fname, model)
if args.fcweight:
    print('Load FC weight:', args.fcweight)
    serializers.load_hdf5(args.fcweight, fc)

# calc unsupervised biases
t_sum = 1.0 + (out_n - 1) * t_bias
p_win = 1.0 / t_sum
p_lose = t_bias / t_sum
print('p_win, p_lose (t_bias): {}, {} ({})'.format(p_win, p_lose, t_bias))

# prepare output directory
try:
    os.mkdir(out_dir)
except:
    pass


# plot initial weight
if start_epoch == 1 and args.plotkernel:
    w = cuda.to_cpu(eval('{}.W.data'.format(args.plotkernel)))
    fname = os.path.join(out_dir, 'k_{}_0.png'.format(args.plotkernel))
    #plot_rf(w, fname)
    th_plot = Process(target=plot_rf, args=(w, fname)) 
    th_plot.start()


# main loop
for epoch in range(start_epoch, n_epoch + 1):
    print('epoch:', epoch)
    sys.stdout.flush()

    # generate a randomized list of training samples
    perm = np.mod(np.random.permutation(n_train_all), n_data_train)[:n_train]

    for loop in six.moves.range(n_iter):
        ix_batch = perm[loop * batchsize:(loop + 1) * batchsize]

        # prep input data
        x = mp.asarray(X_train[ix_batch])
        t = mp.asarray(T_train[ix_batch])

        # feedforward & retrieve
        if args.no_bp:
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    h = model(x, wr=args.wr)
        else:
            h = model(x, wr=args.wr)
        y = fc(h)
        y_label = y.data.argmax(axis=1)
        t_learn = t if not args.unsupervised else y_label

        # calc loss values
        if args.mse:  # MSE
            t_array = label2array(batchsize, out_n, t_learn)  # biased p-dist
            loss = F.mean_squared_error(y, t_array)
        else:  # ACE
            #loss = myloss.my_cross_entropy(y, t_array)
            loss = F.softmax_cross_entropy(y, t)

        model.cleargrads()
        fc.cleargrads()
        loss.backward()
        #loss.forward_grad(rho=args.rho, decay=0.8)  # 1e-2, 0.5
        # 1e-2 for SGD, 1e-6 AdaDelta, 1e-11 for Adam

        if not args.no_bp:
            opt.update()
            model.norm()
        opt_fc.update()
        

        # evaluation
        if (loop + 1) % chk_trial_interval == 0:
            perm_test = np.random.permutation(n_data_test)
            sum_accuracy = 0
            for i in range(0, n_test, batchsize):
                ix_batch = perm_test[i:i + batchsize]
                x_test = mp.asarray(X_test[ix_batch])
                t_test = mp.asarray(T_test[ix_batch])

                # feedforward & retrieve for test
                with chainer.using_config('train', False):
                    with chainer.no_backprop_mode():
                        y_test = fc(model(x_test, wr=args.wr))
                
                # calc loss values
                if args.mse:  # MSE
                    t_array_test = label2array(batchsize, out_n, t_test)
                    loss = F.mean_squared_error(y_test, t_array_test)
                else:  # ACE
                    # loss = FX.my_cross_entropy(y_test, t_array_test)
                    loss = F.softmax_cross_entropy(y_test, t_test)
                
                acc = F.accuracy(y_test, mp.asarray(t_test))
                sum_accuracy += float(acc.data) * len(t_test)

            print('{0:7} -> loss & acc: '.format(loop + 1), end='')
            print(' {0:.4f}'.format(float(loss.data)), end='')
            print(' {0:.4f}'.format(sum_accuracy / n_test))
            sys.stdout.flush()


    # message for each epoch finished
    t = cuda.to_cpu(t)
    ix = np.argsort(t)
    y_label = cuda.to_cpu(y_label)
    u, cnts = np.unique(y_label, return_counts=True)
    label_dom = u[cnts.argmax()]
    dom_answer = (y_label == label_dom).sum().astype(np.float32) / len(y_label)
    print('t  =   [{:3d}'.format(t[ix[0]]), end='')
    for i in ix[1:]:
        print(',{:3d}'.format(t[i]), end='')
    print(']')
    print('y  =   [{:3d}'.format(y_label[ix[0]]), end='')
    for i in ix[1:]:
        print(',{:3d}'.format(y_label[i]), end='')
    print(']')
    print('winner is {} [{:.2f}]'.format(label_dom, dom_answer))
    sys.stdout.flush()

    if args.kill and dom_answer > th_dom_answer:
        break

    # plot current kernels
    if epoch % plot_epoch_interval == 0 and args.plotkernel:
        # retrieve kernels in the 1st layer
        w = cuda.to_cpu(eval('{}.W.data'.format(args.plotkernel)))
        fname = os.path.join(out_dir, 
                             'k_{}_{}.png'.format(args.plotkernel, epoch))
        #plot_rf(w, fname)
        th_plot = Process(target=plot_rf, args=(w, fname)) 
        th_plot.start()
    
    # save current state
    if epoch % save_epoch_interval == 0:
        fname = os.path.join(out_dir, 'model_{}_bp_ft.h5'.format(epoch))
        serializers.save_hdf5(fname, fc)

# save model
if not args.no_bp:
    fname = os.path.join(out_dir, 'model_fin_cp_ft.h5')
    serializers.save_hdf5(fname, model)
fname = os.path.join(out_dir, 'model_fin_bp_ft.h5')
serializers.save_hdf5(fname, fc)


# log output
fname = os.path.join(out_dir, 'log.txt')
with open(fname, 'a') as f:
    f.write(datetime.now().ctime())
    f.write('  model={}, '.format(args.model))
    if args.mse:
        f.write('MSE, ')
    else:
        f.write('ACE, ')
    if args.adam and args.adam > 0.0:
        f.write('Adam(alpha={}), '.format(args.adam))
    elif args.adadelta:
        f.write('AdaDelta, ')
    elif args.momentum > 0.0:
        f.write('Momentum SGD(lr={},'.format(args.lr))
        f.write(' momentum={}), '.format(args.momentum))
    else:
        f.write('SGD(lr={}), '.format(args.lr))
    f.write('\n  ')
    f.write('BPinCLs={}, '.format(not args.no_bp))
    #f.write('rho={}, '.format(args.rho))
    f.write('epoch={}/{}, '.format(epoch, n_epoch))
    f.write('iter={}, '.format(n_iter))
    f.write('bsize={}, '.format(batchsize))
    if args.decay:
        f.write('Wdecay={}, '.format(args.decay))
    f.write('bias={:.2f}'.format(t_bias))
    f.write('\n')


# plot learnt weight
if args.plotkernel:
    w = cuda.to_cpu(eval('{}.W.data'.format(args.plotkernel)))
    fname = os.path.join(out_dir, 'k_{}_fin.png'.format(args.plotkernel))
    plot_rf(w, fname)

# term        
print(epoch, '/', n_epoch, 'epoch finished.')

#EOF
