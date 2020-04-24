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
tar = '/media/spl/D/MicroCT_data/Machine learning/Treadmill running 35n tibia and registration/Treadmill running 35n tibia registered/349 week 3 right tibia registered'
tar = imreadseq_multithread(tar, z_range=[-110,-6])
tar = down_scale(tar,2.15)
tar.SetSpacing(ref.GetSpacing())
tar.SetOrigin(ref.GetOrigin())
tar = sitk.Cast(tar,sitk.sitkFloat32)

initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer( 
        sitk.Cast(ref, sitk.sitkFloat32), sitk.Cast(tar, sitk.sitkFloat32),
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.MOMENTS)
        )

trans = initial_transform.GetTranslation()
trans = [trans[0], trans[1], 0.0]
initial_transform.SetTranslation(trans)  

tar_reg = sitk.Resample(sitk.Cast(tar, sitk.sitkFloat32), sitk.Cast(ref, sitk.sitkFloat32), initial_transform, sitk.sitkLinear, 0.0, tar.GetPixelID())
sitk.WriteTransform(initial_transform, r'/media/spl/D/MicroCT_data/Machine learning/x207y303.tfm')
sitk.WriteImage(tar_reg, r'/media/spl/D/MicroCT_data/Machine learning/x207y303test.nii')