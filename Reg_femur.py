#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import *
import os
import datetime
import re

ref_img = imreadseq('/media/spl/D/MicroCT data/dist_femur_ref',rmbckgrd=60)
ref_img = ref_img[:,:,::-1]
masterdir = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg/Femur week 0'
refdir = os.path.join(masterdir,'..','Registered week 0')

for file in sorted(os.listdir(masterdir)[0:1]):
    if re.search(r'\d{3} week \d (left|right) femur',file):
        imgtitle = file

        '''reftitle = file.replace('week 3','week 0')+' registered'
        ref_img = imreadseq(os.path.join(refdir,reftitle))
        ref_img = ref_img[125:]'''

        tar_img = imreadseq(os.path.join(masterdir,file))
        tar_img = tar_img[:,:,200:1000]

        ini_transform = cent_transform(ref_img,tar_img)
        
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masterdir,imgtitle+" registered")
        if not os.path.exists(suboutput): os.mkdir(suboutput)

        print('Registration of {} is in process...'.format(imgtitle))
        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform,imgtitle)
            imsaveseq(tar_reg,imgtitle+'_Reg',suboutput)
            print('Registration of {} is in completed...'.format(imgtitle))
        except RuntimeError as ex:
            print('Registration of {} failed...'.format(imgtitle))
            print(ex.message)
            pass
        print(datetime.datetime.now().time())