#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import pycuda.autoinit
import numpy as np 
from pycuda import gpuarray
from time import time


host_data = np.float32(np.random.random(500000000))

t1 = time()
host_data_x2 = host_data*np.float32(2)
t2 = time()

print('total time to compute on CPU:{:0>5.2f}'.format(t2-t1))

t1 = time()
device_data = gpuarray.to_gpu(host_data)
t2 = time()
device_data_x2 = np.float(2) * device_data
t3 = time()
host_data_x2 = device_data_x2.get()
t4 = time()

print('Copy from host to GPU:{:0>5.2f}'.format(t2-t1))
print('total time to compute on GPU:{:0>5.2f}'.format(t3-t2))
print('Copy from GPU to host:{:0>5.2f}'.format(t4-t3))
print('Total time:{:0>5.2f}'.format(t4-t1))

# Data transfer between host and gpu cost a lot time.

