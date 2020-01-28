#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import *
import os
import re
import logging
from shubow_tools import imreadseq_multithread,auto_crop, Rotate_by_Euler_angles

#ref_img = imreadseq_multithread('/media/spl/D/MicroCT data/MicroCT registration ref/dist_femur_ref/VOI',rmbckgrd=60)
wkdir = r'/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg'
masterdir = os.path.join(wkdir,'week 3 femur')
masteroutput = os.path.join(wkdir,'VOI_Registered femur week 3')
refdir = os.path.join(wkdir,'Registered femur week 0')

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

for file in sorted(os.listdir(masterdir)):
    if re.search(r'\d{3} week \d (left|right) femur',file):
    # if re.search(r'\d{3} week \d (left|right) femur',file):
        imgtitle = file
        logging.info('Loading image of {} ...'.format(imgtitle))
        reftitle = file.replace('week 3','week 0')+' registered'
        ref_img = imreadseq_multithread(os.path.join(refdir,reftitle),z_range=[450,590])
        tar_img = imreadseq_multithread(os.path.join(masterdir,file),sitkimg=False,z_range=[420,620], rmbckgrd=60)
        tar_img = sitk.GetImageFromArray(auto_crop(Rotate_by_Euler_angles(tar_img))) # femur
        
        ini_transform = cent_transform(ref_img,tar_img)
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masteroutput,imgtitle+" registered")

        logging.info('Registration of {} is in process...'.format(imgtitle))
        try:
            if not os.path.exists(suboutput): os.mkdir(suboutput)
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform,imgtitle,suboutput)
            logging.info("Saving images...")
            imsaveseq(tar_reg,imgtitle+'_Reg',suboutput)
            sitk.WriteTransform(tar_reg_transform,os.path.join(suboutput,imgtitle+'reg_transform.tfm'))
            logging.info('Registration of {} is in completed...'.format(imgtitle))
        except RuntimeError as ex:
            logging.info('Registration of {} failed...'.format(imgtitle))
            print(ex)
            pass
