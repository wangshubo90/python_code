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

wkdir = r"/home/blue/SITK_registered_image_14um/2nd batch tibia"
os.chdir(wkdir)
masterdir = r"/home/blue/SITK_registered_image_14um/2nd batch tibia"
masteroutput = r"/home/blue/SITK_registered_image_7um" 

#refdir = r"/home/blue/SITK_registered_image_14um/418 week 1left registered/M91 week 2 left tibia registered"

format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO,
                    datefmt="%H:%M:%S")
#logging.info('Loading reference image...')

#ref_img = imreadseq_multithread(refdir,thread = 2, sitkimg=True, rmbckgrd=75, z_range=[-756,None])
#ref_img = down_scale(ref_img, down_scale_factor=1.0)

failed_list = []

with open("failed.txt", "r") as f :
    retry_file = f.readlines()

retry_list = [i[:-3] for i in retry_file]
read_range_list = [i[-2] for i in retry_file]

for file in sorted(os.listdir(masterdir))[:]:
    if re.search(r"\d{3} (week \d) (left|right) tibia", file) and file in retry_list:
        imgtitle = file
        
        ref_img = os.path.join(masteroutput, re.sub(r"week [2-5]", "week 1", file)+' registered')
        ref_img = imreadseq_multithread(ref_img, sitkimg=True, rmbckgrd=75, z_range=[-350, None])

        
        read_range = read_range_list[retry_list.index(file)] # if need adjustment to read_range
        if read_range == 'u':
            lower = -560
            upper = -40
        elif read_range == 'd' :
            lower = -630
            upper = -800
        
        flip = False if read_range == "f" else True

        lower = -630
        upper = -80

        if 'right' in file and flip:
            tar_img = imreadseq_multithread(os.path.join(masterdir,file), thread=2,
                                sitkimg = False, rmbckgrd=75, z_range=(lower, upper))
            tar_img = sitk.GetImageFromArray(np.flip(tar_img, axis = 2))
        else:
            tar_img = imreadseq_multithread(os.path.join(masterdir,file), thread=2,
                                    sitkimg = True, rmbckgrd=75, z_range=(lower, upper))
        
        #tar_img = down_scale(tar_img, down_scale_factor=1.0)

        logging.info('Initial Transforming ...')
        #ini_transform = init_transform_best_angle(sitk.Cast(tar_img, sitk.sitkFloat32),sitk.Cast(ref_img, sitk.sitkFloat32), angles=[np.pi*i/8 for i in range(-2,1)])
        ini_transform = sitk.ReadTransform(os.path.join(masterdir, imgtitle, imgtitle+".tfm"))
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masteroutput,imgtitle+" registered")
        logging.info('Registration of {} is in process...'.format(imgtitle))

        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img[100:,:,:],ini_transform,imgtitle,suboutput)
            logging.info("Saving images...")
            
            if os.path.exists(suboutput):
                shutil.rmtree(suboutput)
            os.mkdir(suboutput)
            
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