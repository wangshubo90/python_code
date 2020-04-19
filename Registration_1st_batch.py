#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import reg_transform, cent_transform
import os
import re
import logging
from shubow_tools import imreadseq_multithread,imsaveseq, auto_crop, down_scale, init_transform_best_angle
import shutil
import numpy as np

wkdir = r"/media/spl/D/MicroCT_data/Machine learning/1st batch tibia registration"
os.chdir(wkdir)
masterdir = r"/media/spl/D/MicroCT_data/Machine learning/Jul 2018 1st batch L & R tibia/L & R tibia 7.9"
masteroutput = r"/media/spl/D/MicroCT_data/Machine learning/1st batch tibia registration" 

refdir = r"/media/spl/D/MicroCT_data/MicroCT registration ref/6w F WT tibia"

format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO, 
                    datefmt="%H:%M:%S")
logging.info('Loading reference image...')

ref_img = imreadseq_multithread(refdir,thread = 2, sitkimg=True, rmbckgrd=75, z_range=[-756,None])
ref_img = down_scale(ref_img, down_scale_factor=2.0)

failed_list = []
'''
with open("failed2.txt", "r") as f :
    retry = f.readlines()

retry = [i[:-1] for i in retry]
'''
for file in sorted(os.listdir(masterdir))[:]:
    if re.search(r"\d{3} (week \d) (left|right) tibia", file): #and file in retry:
        imgtitle = file
        logging.info('Loading image {} ...'.format(imgtitle))
        
        if 'right' in file:
            tar_img = imreadseq_multithread(os.path.join(masterdir,file), thread=2,
                                sitkimg = False, rmbckgrd=75, z_range=(-1100, -150))
            tar_img = sitk.GetImageFromArray(np.flip(tar_img, axis = 2))
        else:
            tar_img = imreadseq_multithread(os.path.join(masterdir,file), thread=2,
                                sitkimg = True, rmbckgrd=75, z_range=(-1100, -150))
        
        tar_img = down_scale(tar_img, down_scale_factor=2.0)

        logging.info('Initial Transforming ...')
        ini_transform = init_transform_best_angle(tar_img,ref_img, angles=[np.pi*i/8 for i in range(-5,2)])
        #ini_transform = sitk.ReadTransform("/media/spl/D/MicroCT_data/Machine learning/Heart inj Aug-2019 tibia registration/381 week 0 left tibia registered/381 week 0 left tibiareg_transform.tfm")
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masteroutput,imgtitle+" registered")
        logging.info('Registration of {} is in process...'.format(imgtitle))

        if os.path.exists(suboutput): 
            shutil.rmtree(suboutput)
        else:
            os.mkdir(suboutput)

        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform,imgtitle,suboutput)
            logging.info("Saving images...")
            imsaveseq(tar_reg, suboutput, imgtitle+'_Reg')
            sitk.WriteTransform(tar_reg_transform,os.path.join(suboutput,imgtitle+'reg_transform.tfm'))
            logging.info('Registration of {} is in completed...'.format(imgtitle))
        except RuntimeError as ex:
            logging.info('Registration of {} failed...'.format(imgtitle))
            failed_list.append(file)
            print(ex)
            pass

print(failed_list)

with open("failed.txt", "w") as f:
    for i in failed_list:
        f.write(i+"\n")