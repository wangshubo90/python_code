#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
import sys
from pathlib import Path
import re
from shubow_tools import imreadseq_multithread
import glob

src = r'/media/spl/D/MicroCT_data/Machine learning/SITK_reg_7um'
dst = r'/run/user/1000/gvfs/smb-share:server=lywanglab.local,share=micro_ct_data/Deep learning project/7_um_data'
os.chdir(src)

for fd in glob.glob('*registered'):
    print('reading: ' + fd)
    s = re.search(r'(\d{3}) week (\d) (left|right)', fd)
    name = '{}{}T_w{}'.format(s.group(1),s.group(3)[0].upper(), s.group(2))
    img = imreadseq_multithread(fd, z_range=[-300,None])
    sitk.WriteImage(img, os.path.join(dst,name+'.nii.gz'))
    print(name + '.nii.gz saved')
