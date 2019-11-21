#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: <2019-05-22 14:27:05 tshino>
#
"""
Train CNN with various structures

Copyright (C) 2016-19 Takashi Shinozaki
"""

from __future__ import print_function
import argparse
import os, sys
import importlib
from datetime import datetime
from multiprocessing import Process, Pool
import signal

from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from tqdm import trange

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
parser.add_argument('--batchsize', '-B', default=16, type=int,
                    help='# of samples in a mini-batch')
parser.add_argument('--netmodeldir', '-N', default='./model',
                    help='Directory name for network model files')
parser.add_argument('--model', '-M', default='alex_cp',
                    help='Model module name without .py')
parser.add_argument('--weight', '-W', default='',
                    help='Initialize weights from given file')
parser.add_argument('--fcweight', '-F', default='',
                    help='Initialize FC weights from given file')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy', 
                    help='Mean subtraction by given npy file')
parser.add_argument('--root', '-R', default='ImageNet', 
                    help='Root directory of image files')
parser.add_argument('--wr', '-w', default=0.0, type=float)
parser.add_argument('--fcunits', '-f', default=4096, type=int,
                    help='# of units for the 1st fully connected layer')
args = parser.parse_args()


# parameters
batchsize = args.batchsize

base_size = 256
#n_test = 1000 # batchsize * 8  # 1000
n_val = 2000
out_n = 1000  # for ImageNet dataset


# some filenames
lst_train_image = 'all_train.lst'
lst_train_label = 'all_train_label.lst'
lst_val_image = 'all_val.lst'
lst_val_label = 'all_val_label.lst'


# util functions
def read_image(fname):
    # open image file
    img = Image.open(os.path.join(args.root, fname))

    # crop centeral square region & resize
    w, h = img.size
    if w > h:  
        img = img.crop(((w - h)/2, 0, (w + h)/2, h))
    else:
        img = img.crop((0, (h - w)/2, w, (h + w)/2))
    #img = img.resize((base_size, base_size))
    img = img.resize((insize, insize))

    # # random 5 types crop
    # d = base_size - insize
    # c = np.random.randint(5)
    # if c == 0:
    #     box = (0, 0, insize, insize)
    # elif c == 1:
    #     box = (d, 0, d + insize, insize)
    # elif c == 2:
    #     box = (0, d, insize, d + insize)
    # elif c == 3:
    #     box = (d, d, d + insize, d + insize)
    # else:
    #     dh = d/2
    #     box = (dh, dh, dh + insize, dh + insize)
    # img = img.crop(box)

    # # random horizontal flip
    # if np.random.randint(2):
    #     img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if img.mode is not 'RGB':  # check color mode
        img = img.convert('RGB')  # convert from L or CMYK to RGB

    # convert to numpy array with appropriate shape
    img_mx = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    img_mx = img_mx[::-1]  # convert RGB to BGR
    img_mx -= X_mean  # subtract mean value
    return img_mx


# load ImageNet dataset
print('loading file lists of dataset:')

print(lst_train_image, end=' ')
sys.stdout.flush()
with open(os.path.join(args.root, lst_train_image), 'r') as f:
    X_train = [line.strip() for line in f.readlines()]

print(lst_train_label, end=' ')
sys.stdout.flush()
with open(os.path.join(args.root, lst_train_label), 'r') as f:
    T_train = [line.strip() for line in f.readlines()]
T_train = np.array(T_train, dtype=np.int32)

print(lst_val_image, end=' ')
sys.stdout.flush()
with open(os.path.join(args.root, lst_val_image), 'r') as f:
    X_test = [line.strip() for line in f.readlines()]

print(lst_val_label)
sys.stdout.flush()
with open(os.path.join(args.root, lst_val_label), 'r') as f:
    T_test = [line.strip() for line in f.readlines()]
T_test = np.array(T_test, dtype=np.int32)

# # use a part of labels
# if args.labellist:
#     llst = []
#     with open(args.labellist, 'rt') as f:
#         for line in f:
#             llst.append(line.strip())
#     mask_train = np.isin(T_train, llst)
#     X_train = X_train[mask_train]
#     T_train = T_train[mask_train]
#     mask_test = np.isin(T_train, llst)
#     X_test = X_test[mask_test]
#     T_test = T_test[mask_test]

n_data_train = len(T_train)
n_data_test = len(T_test)


# model definition
print('loading model module:', args.model)
sys.stdout.flush()
sys.path.append(args.netmodeldir)
model_module = importlib.import_module(args.model)
model = model_module.CP(with_bp=False)
insize = model.insize
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
print(', mini-batch size =', batchsize)


# load weight data from hdf5 file
if args.weight:
    print('Load model weight:', args.weight)
    fname = os.path.join(args.netmodeldir, args.weight)
    serializers.load_hdf5(fname, model)
if args.fcweight:
    print('Load FC weight:', args.fcweight)
    serializers.load_hdf5(args.fcweight, fc)

# load mean file
if args.mean:
    fname = os.path.join(args.root, args.mean)
    print('loading mean data:', fname)
    X_mean = np.load(fname).astype(np.float32)
    offset_x = (X_mean.shape[1] - insize) / 2
    offset_y = (X_mean.shape[2] - insize) / 2
    X_mean = X_mean[:, offset_x:offset_x+insize, offset_x:offset_x+insize]


# prepare multiprocess loading of input images
if args.processes > 0:
    print('use multi-thread data loading')
    pool = Pool(processes=args.processes)
else:
    print('use single-thread data loading')

def handler(signum, frame):
    pool.terminate()
    sys.exit(1)

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


# main loop
(sum_top_1, sum_top_5, sum_top_10) = (0.0, 0.0, 0.0)
for i in trange(0, n_data_test, batchsize):
    ix_batch = np.arange(i, min(i + batchsize, n_data_test))
    n = len(ix_batch)

    # prep input data for test
    lst_fname = [X_test[j] for j in ix_batch]
    if args.processes > 0:
        x_test = mp.asarray(pool.map(read_image, lst_fname))
    else:
        x_test = mp.asarray(map(read_image, lst_fname))
    t_test = np.asarray(T_test[ix_batch])
    
    # feedforward & retrieve for test
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            y_test = fc(model(x_test, wr=args.wr))

    # gest top-5 result
    y_test = mp.asnumpy(y_test.data)
    top10 = np.argsort(y_test, axis=1)[:, :-11:-1]  # [:, :-6:-1] for top 5
    top5 = top10[:, :5]
    top1 = top10[:, 0]
    sum_top_10 += np.sum(np.any(top10 == t_test[:, np.newaxis], axis=1))
    sum_top_5 += np.sum(np.any(top5 == t_test[:, np.newaxis], axis=1))
    sum_top_1 += np.sum(top1 == t_test)
    #print(t_test)
    #print(top10.T)

print('Top-1:  {0:.4f}'.format(sum_top_1 / n_data_test))
print('Top-5:  {0:.4f}'.format(sum_top_5 / n_data_test))
print('Top-10:  {0:.4f}'.format(sum_top_10 / n_data_test))

# term        
print('done.')

#EOF
