#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
import sys
from pathlib import Path
import re
from shubow_tools import imreadseq_multithread
import glob

src = r'/media/spl/D/MicroCT_data/4th batch bone mets loading study'
dst = r'/run/user/1000/gvfs/afp-volume:host=Lywanglab.local,user=shubow,volume=Micro_CT_Data/Deep learning project/7_um_data'
os.chdir(src)

for wk in glob.glob('Registration week*'):
    os.chdir(wk)

    for fd in glob.glob('*registered'):
        print('reading: ' + fd)
        s = re.search(r'(\d{3}) week (\d) (left|right)', fd)
        name = '{}{}T_w{}'.format(s.group(1),s.group(3)[0].upper(), s.group(2))
        img = imreadseq_multithread(fd, z_range=[-300,None])
        sitk.WriteImage(img, os.path.join(dst,name+'.nii.gz'))
        print(name + '.nii.gz saved')

    os.chdir('..')