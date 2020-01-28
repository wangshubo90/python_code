#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import reg_transform, cent_transform
import os
import re
import logging
from shubow_tools import imreadseq_multithread,imsaveseq, Rotate_by_Euler_angles,auto_crop

#ref_img = imreadseq_multithread('/media/spl/D/MicroCT data/MicroCT registration ref/whole_tibia_ref',rmbckgrd=60,z_range=[40,840])
wkdir = r'D:/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg'
masterdir = os.path.join(wkdir,'week 3 tibia')
masteroutput = os.path.join(wkdir,'Registered tibia week 3')
refdir = os.path.join(wkdir,'Registered tibia week 0')

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

for file in sorted(os.listdir(masterdir)):
    if re.search(r'(410.*left)',file):
    #if re.search(r'\d{3} week \d (left|right) tibia',file):
        imgtitle = file

        logging.info('Loading image of {} ...'.format(imgtitle))
        reftitle = file.replace('week 3','week 0')+' registered'
        ref_img = imreadseq_multithread(os.path.join(refdir,reftitle))
        tar_img = imreadseq_multithread(os.path.join(masterdir,file),sitkimg=False,rmbckgrd=60,z_range=[0,840])
        tar_img = sitk.GetImageFromArray(auto_crop(Rotate_by_Euler_angles(tar_img)))
        
        ini_transform = cent_transform(ref_img,tar_img)
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masteroutput,imgtitle+" registered")
        if not os.path.exists(suboutput): 
            os.mkdir(suboutput)
        logging.info('Registration of {} is in process...'.format(imgtitle))
        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform,imgtitle,suboutput)

            imsaveseq(tar_reg,suboutput,imgtitle+'_Reg')
            logging.info('Registration of {} is finished...'.format(imgtitle))
        except RuntimeError as ex:
            logging.info('Registration of {} failed...'.format(imgtitle))
            print(ex)
            pass

logging.info("Done !")
