#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import *
import os
import re
import logging

#ref_img = imreadseq('/media/spl/D/MicroCT data/MicroCT registration ref/whole_tibia_ref',rmbckgrd=60)
#ref_img = imreadseq('/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg/Registered tibia week 0/416 week 0 left tibia registered')
#ref_img = ref_img[:,:,200:1000] # for tibia


masterdir = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg/week 3 tibia'
refdir = os.path.join(masterdir,'..','Registered tibia week 0')

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

for file in sorted(os.listdir(masterdir))[5:6]:
    if re.search(r'\d{3} week \d (left|right) tibia',file):
        imgtitle = file

        logging.info('Loading image of {} ...'.format(imgtitle))
        reftitle = file.replace('week 3','week 0')+' registered'
        ref_img = imreadseq(os.path.join(refdir,reftitle))
        tar_img = imreadseq(os.path.join(masterdir,file))
        tar_img = tar_img[150:-200,100:-150,125:1100]
        
        ini_transform = cent_transform(ref_img,tar_img)
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masterdir,'..','Registered tibia week 3',imgtitle+" registered")

        logging.info('Registration of {} is in process...'.format(imgtitle))
        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform,imgtitle)
            if not os.path.exists(suboutput): os.mkdir(suboutput)
            imsaveseq(tar_reg,imgtitle+'_Reg',suboutput)
            logging.info('Registration of {} is finished...'.format(imgtitle))
        except RuntimeError as ex:
            logging.info('Registration of {} failed...'.format(imgtitle))
            print(ex)
            pass

logging.info("Done !")
