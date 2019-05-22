#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: <2018-05-18 16:21:38 tshino>
#
"""
Add feedforward gradient method to chainer.Variable

Copyright (C) 2016-18 Takashi Shinozaki
"""

import heapq
import numpy

import chainer
from chainer import cuda
from chainer import Variable
from chainer.variable import _check_grad_type

from types import MethodType



def forward_grad(self, rho=1e-3, decay=0.50, loss_scale=None):
    """test
    """
    self._node._check_old_style_gradient()
    if self.creator_node is None:
        return
    initial_device = None
    if cuda.available and isinstance(self.data, cuda.ndarray):
        try:
            initial_device = cuda.Device()
        except cuda.cupy.cuda.runtime.CUDARuntimeError as e:
            if e.status != 38:  # cudaErrorNoDevice
                raise

    is_debug = chainer.is_debug()

    cand_funcs = []
    seen_set = set()

    def add_cand(cand):
        if cand not in seen_set:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
            seen_set.add(cand)

    add_cand(self.creator_node)

    cur_decay = 1.0
    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_input_indexes = [
            i for i, x in enumerate(inputs) if x.requires_grad
        ]
        if not target_input_indexes:
            continue

        in_data = tuple([x.data for x in inputs])
        cuda.get_device_from_array(*in_data).use()
        if hasattr(func, 'with_frad') and func.with_frad:
            gW, gb = func.forward_grad(in_data, rho)
            gxs = [None, Variable(gW * cur_decay), Variable(gb * cur_decay)]
            cur_decay *= decay
        else:
            gxs = [None] * len(inputs)

        if is_debug:
            for gx in gxs:
                if gx is None:
                    continue
                gx_data = gx.data
                if gx_data.dtype.kind == 'f':
                    cuda.get_device_from_array(gx_data).use()
                    if cuda.get_array_module(gx_data).isnan(gx_data).any():
                        raise RuntimeError(
                            'NaN is detected on forward-grad computation of '
                            '{}'.format(func.label))

        for i, gx in enumerate(gxs):
            x = inputs[i]
            if x.creator_node is not None:
                add_cand(x.creator_node)

            if gx is None:
                continue

            if not x.requires_grad:
                continue

            _check_grad_type(func, x, gx.data)

            x_var = x.get_variable_or_none()
            if x_var is not None:
                x_var._grad_var = gx
                x_var._loss_scale = loss_scale

        del gxs  # to reduce memory usage
        if initial_device is not None:
            initial_device.use()

# Add the method to 'Variable' object
Variable.forward_grad = MethodType(forward_grad, None, Variable)

#EOF
