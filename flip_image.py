#! /home/blue/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from shubow_tools import imreadseq_multithread,imsaveseq
import os
import re
import shutil

wkdir = r"/home/blue/SITK_registered_image_14um/3rd batch tibia"
dst = r"/home/blue/SITK_registered_image_14um/Flipped"
os.chdir(wkdir)

with open("failed.txt", "r") as f :
    retry_file = f.readlines()

retry_list = [i[:-1] for i in retry_file]

for fd in retry_list:

    if not 'right' in fd:
        continue
    print(fd)
    output = os.path.join(dst,fd)
    if os.path.exists(output):
        shutil.rmtree(output)
        
    os.mkdir(output)

    image = imreadseq_multithread(os.path.join(wkdir, fd))
    image = image[::-1,:,:]
    imsaveseq(image, os.path.join(dst,fd), fd)
    print('Next!')
