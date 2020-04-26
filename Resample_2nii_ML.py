#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
import re
import logging
from shubow_tools import *
import shutil
import numpy as np

ref = '/home/blue/machine learning/418LT_w0.nii.gz'
ref = sitk.ReadImage(ref)

tardir = r'/home/blue/machine learning/3rd batch tibia registered'
outdir = r'/home/blue/machine learning/100x100x48 niis'
os.chdir(tardir)



for fd in os.listdir(tardir):
        s = re.search(r'(\d{3}) week (\d) (left|right) tibia', fd)
        name = '{}{}T_w{}'.format(s.group(1),s.group(3)[0].upper(), s.group(2))
        tar = imreadseq_multithread(fd, z_range=[-104,None])
        
        if tar.GetSize() == (276, 275, 104):
                initial_transform = sitk.ReadTransform(r'/home/blue/machine learning/x276y275.tfm')
        elif tar.GetSize() == (207, 303, 104):
                initial_transform = sitk.ReadTransform(r'/home/blue/machine learning/x207y303.tfm')
        elif tar.GetSize() == (304, 278, 104):
                initial_transform = sitk.ReadTransform(r'/home/blue/machine learning/x304y278.tfm')

        tar = down_scale(tar,2.15)
        tar.SetSpacing(ref.GetSpacing())
        tar.SetOrigin(ref.GetOrigin())

        tar_reg = sitk.Resample(sitk.Cast(tar, sitk.sitkFloat32), sitk.Cast(ref, sitk.sitkFloat32), initial_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

        sitk.WriteImage(tar_reg, os.path.join(outdir,name+'.nii.gz'))