#! /home/blue/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import reg_transform, cent_transform
import os
import re
import logging
from shubow_tools import imreadseq_multithread,imsaveseq, auto_crop, down_scale, init_transform_best_angle
import shutil
import numpy as np

wkdir = r"E:\Yoda1-tumor-loading 2.26.2021\Tibia & femur week 3"
os.chdir(wkdir)
masterdir = r"E:\Yoda1-tumor-loading 2.26.2021\Tibia & femur week 3"
masteroutput = r"E:\Yoda1-tumor-loading 2.26.2021\Registration week 3" 
refdir = r"E:\Yoda1-tumor-loading 2.26.2021\Registration week 0"

BASELINE_REGISTRATION = True

if not os.path.exists(masteroutput):
    os.mkdir(masteroutput)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO,
                    datefmt="%H:%M:%S")

if not BASELINE_REGISTRATION:
    logging.info('Loading reference image...')
    ref_img = imreadseq_multithread(refdir,thread = 2, sitkimg=True, z_range=[-355,-5])
#ref_img = down_scale(ref_img, down_scale_factor=1.0)

failed_list = []

'''with open("failed.txt", "r") as f :
    retry_file = f.readlines()

retry_list = [i[:-3] for i in retry_file]
read_range_list = [i[-2] for i in retry_file]'''

for file in sorted(os.listdir(masterdir)):
    if re.search(r"\d{3}.(week.\d) (left|right) tibia", file):
        imgtitle = file
        
        if BASELINE_REGISTRATION:
            logging.info('Loading reference image : {}'.format(re.sub(r"week [1-5]", "week 0",file)+' registered'))
            ref_img = imreadseq_multithread(os.path.join(refdir, re.sub(r"week [1-5]", "week 0",file)+' registered')\
                ,thread = 2, sitkimg=True, rmbckgrd=75)

        logging.info('Loading image {} ...'.format(imgtitle))
        
        '''read_range = read_range_list[retry_list.index(file)] # if need adjustment to read_range
        if read_range == 'u':
            lower = -500
            upper = -1
        elif read_range == 'd' :
            lower = -600
            upper = -120'''
        
        lower = -560 
        upper = -1

        if '__right__' in file:
            tar_img = imreadseq_multithread(os.path.join(masterdir,file), thread=2,
                                sitkimg = False, z_range=(lower, upper))
            tar_img = sitk.GetImageFromArray(np.flip(tar_img, axis = 2))
        else:
            tar_img = imreadseq_multithread(os.path.join(masterdir,file), thread=2,
                                    sitkimg = True, z_range=(lower, upper))
        
        #tar_img = down_scale(tar_img, down_scale_factor=1.0)

        logging.info('Initial Transforming ...')
        ini_transform = init_transform_best_angle(sitk.Cast(tar_img, sitk.sitkFloat32),sitk.Cast(ref_img, sitk.sitkFloat32),
                angles=[np.pi*i/8 for i in range(-3,1)])
        #ini_transform = sitk.ReadTransform("/media/spl/D/MicroCT_data/Machine learning/Heart inj Aug-2019 tibia registration/381 week 0 left tibia registered/381 week 0 left tibiareg_transform.tfm")
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masteroutput,imgtitle+" registered")
        logging.info('Registration of {} is in process...'.format(imgtitle))

        if os.path.exists(suboutput):
            shutil.rmtree(suboutput)
        
        os.mkdir(suboutput)

        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img[100:,:,:],ini_transform,imgtitle,suboutput)
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

with open("failed_retry.txt", "w") as f:
    for i in failed_list:
        f.write(i+"\n")