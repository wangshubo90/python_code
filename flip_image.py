#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from shubow_tools import imreadseq_multithread,imsaveseq
import os
import re
import shutil

wkdir = r"/media/spl/D/MicroCT_data/Machine learning/Treadmill running 35n tibia"
dst = r"/media/spl/D/MicroCT_data/Machine learning/Flipped"
os.chdir(wkdir)

with open("failed_retry.txt", "r") as f :
    retry_file = f.readlines()

retry_list = [i[:-1] for i in retry_file]

for fd in retry_list:
    print(fd)
    output = os.path.join(dst,fd)
    if os.path.exists(output):
        shutil.rmtree(output)
        
    os.mkdir(output)

    image = imreadseq_multithread(os.path.join(wkdir, fd))
    if 'right' in fd:
        image = image[::-1,:,:]
        imsaveseq(image, os.path.join(dst,fd), fd)
    else:
        pass
    print('Next!')