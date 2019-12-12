#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import pycuda.driver as drv 

drv.init()
print ('Detected {} CUDA Capable device(s)'.format(drv.Device.count()))
for i in range(drv.Device.count()):
    gpu_device = drv.Device(i)
    print ('Device {}: {}'.format( i, gpu_device.name() ))
    compute_capability = float( '%d.%d' % gpu_device.compute_capability())
    print ('\t Compute Capability: {}'.format(compute_capability))
    print ('\t Total Memory: {} megabytes'.format(gpu_device.total_memory()//(1024**2)))

device_attributes_tuples = gpu_device.get_attributes().items()
device_attributes = {}
for k, v in device_attributes_tuples:
    device_attributes[str(k)] = v

num_mp = device_attributes['MULTIPROCESSOR_COUNT']

cuda_cores_per_mp = { 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 :128, 6.2 : 128}[compute_capability]

print ('\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))

device_attributes.pop('MULTIPROCESSOR_COUNT')
for k in device_attributes.keys():
    print ('\t {}: {}'.format(k, device_attributes[k]))