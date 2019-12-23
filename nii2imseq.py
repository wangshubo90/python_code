#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import imsaveseq
import os

niidir = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg/Tibia week 0'

for file in os.listdir(niidir):
    if file.endswith('.nii'):
        imgtitle = file[:-4]
        outsub = os.path.join(niidir, imgtitle)
        if not os.path.exists(outsub): 
            os.mkdir(outsub)
        img = sitk.ReadImage(os.path.join(niidir,file))
        imsaveseq(img, imgtitle,outsub)
