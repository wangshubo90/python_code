#! \home\spl\ml\sitk\bin\python

# -*- coding: utf-8 -*-

from pycuda import autoinit
import pycuda.driver as drive
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

ker = SourceModule("""
__global__ void scalar_multiply_kernel(
    float *outvec, float scalar, float *invec){
        int i = threadIdx.x;
        outvec[i] = invec[i]*scalar;
}
""")
# __global__ distinguish the function as a kenel to the compiler.
# no return, so we delar the function as void. Output is always obtained by gpuarray.get()


scalar_multiply = ker.get_function("scalar_multiply_kernel")
testvec = np.random.randn(1024).astype(np.float32)
testvec_gpu = gpuarray.to_gpu(testvec)
outvec_gpu = gpuarray.empty_like(testvec_gpu)

scalar_multiply(outvec_gpu,np.float32(2),testvec_gpu, block = (1024,1,1), grid=(1,1,1))

outvec = outvec_gpu.get()

print("Does our kernel work properly?:{}".format(np.allclose(outvec,testvec*2)))


