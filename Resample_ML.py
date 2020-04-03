#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
import re
import logging
from shubow_tools import imreadseq_multithread,imsaveseq, auto_crop, down_scale, init_transform_best_angle
import shutil
import numpy as np

wkdir = r"/media/spl/D/MicroCT_data/Shubo/treadmill running trail 4.8.2019"
os.chdir(wkdir)
masterdir = r"Treadmill trial 6 male mice LT RT LF RF"
masteroutput = r"Treadmill trial tibia registration" 

format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO, 
                    datefmt="%H:%M:%S")
logging.info('Loading reference image...')

refdir = r"/media/spl/D/MicroCT_data/Shubo/treadmill running trail 4.8.2019/Treadmill trial tibia registration/M93 week 4 left tibia registered"
ref_img = imreadseq_multithread(refdir,thread = 2, sitkimg=True)

failed_list = []

with open("failed.txt", "r") as f :
    retry = f.readlines()

retry = [i[:-1] for i in retry]

for file in sorted(os.listdir(masterdir))[:]:
    if re.search(r"M\d{2} (week \d) (left|right) tibia", file) and file in retry[2:]:
        imgtitle = file
        logging.info('Loading image {} ...'.format(imgtitle))
        tar_img = imreadseq_multithread(os.path.join(masterdir,file,"Registration"), thread=2,
                                sitkimg = True, rmbckgrd=75, z_range=(-752, None))
        tar_img = down_scale(tar_img, down_scale_factor=2.0)
        tar_img.SetSpacing((1.0,1.0,1.0))

        suboutput = os.path.join(masteroutput,imgtitle+" registered")
        logging.info('Resampling of {} is in process...'.format(imgtitle))

        if os.path.exists(suboutput): 
            shutil.rmtree(suboutput)
        else:
            os.mkdir(suboutput)
        
        initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer( sitk.Cast(ref_img, sitk.sitkFloat32), tar_img,
                                                                                sitk.Euler3DTransform(), 
                                                                                sitk.CenteredTransformInitializerFilter.MOMENTS))
        
        trans = initial_transform.GetTranslation()
        trans = [trans[0], trans[1], 0.0]
        initial_transform.SetTranslation(trans)        
        
        tar_reg = sitk.Resample(tar_img, ref_img, initial_transform, sitk.sitkLinear, 0.0, tar_img.GetPixelID())
        imsaveseq(tar_reg, suboutput, imgtitle+'_Reg')
        