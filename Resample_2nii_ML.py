#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
import re
import logging
from shubow_tools import *
import shutil
import numpy as np

ref = '/home/spl/Machine Learning/Data100x100x48/418LT_w0.nii.gz'
ref = sitk.ReadImage(ref)

tardir = r'/media/spl/D/MicroCT_data/Machine learning/rest'
outdir = r'/media/spl/D/MicroCT_data/Machine learning/rest'
os.chdir(tardir)

#mask = imreadseq_multithread(r'/media/spl/D/MicroCT_data/Machine learning/mask/ROI', sitkimg= False)
#mask = mask == 255

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                datefmt="%H:%M:%S")

for fd in glob.glob('*registered'):
        if fd[:3] in ['360','361','362','363','364','365','366']:
                logging.info('Resampling {}'.format(fd))
                s = re.search(r'(\d{3}) week (\d) (left|right) tibia', fd)
                name = '{}{}T_w{}'.format(s.group(1),s.group(3)[0].upper(), s.group(2))
                tar = imreadseq_multithread(fd, z_range=[-104,None])

                if tar.GetSize() == (276, 275, 104):
                        initial_transform = sitk.ReadTransform(r'/media/spl/D/MicroCT_data/Machine learning/x276y275.tfm')
                elif tar.GetSize() == (270, 303, 104):
                        initial_transform = sitk.ReadTransform(r'/media/spl/D/MicroCT_data/Machine learning/x270y303.tfm')
                elif tar.GetSize() == (304, 278, 104):
                        initial_transform = sitk.ReadTransform(r'/media/spl/D/MicroCT_data/Machine learning/x304y278.tfm')
                elif tar.GetSize() == (304, 282, 104):
                        initial_transform = sitk.ReadTransform(r'/media/spl/D/MicroCT_data/Machine learning/x304y282.tfm')

                tar = down_scale(tar,2.15)
                tar.SetSpacing(ref.GetSpacing())
                tar.SetOrigin(ref.GetOrigin())

                tar_reg = sitk.Resample(sitk.Cast(tar, sitk.sitkFloat32), sitk.Cast(ref, sitk.sitkFloat32), initial_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                #tar_reg = sitk.GetImageFromArray(sitk.GetArrayFromImage(tar_reg) * mask)
                sitk.WriteImage(tar_reg, os.path.join(outdir,name+'.nii.gz'))