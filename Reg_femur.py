#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
from Across_limb_registration import *
import os
import datetime
import re

print(datetime.datetime.now().time())

'''
ref_img = imreadseq('/media/spl/D/MicroCT data/whole_tibia_ref/VOI',rmbckgrd=60)
ref_img = ref_img[:,:,200:1000]
'''

masterdir = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg/Femur week 3'
refdir = os.path.join(masterdir,'..','Registered week 0')

for file in sorted(os.listdir(masterdir))[4:6]:
    if re.search(r'\d{3} week \d (left|right) femur',file):
        imgtitle = file
        
        reftitle = file.replace('week 3','week 0')+' registered'
        ref_img = imreadseq(os.path.join(refdir,reftitle))
        
        tar_img = imreadseq(os.path.join(masterdir,file))
        tar_img = tar_img[25:-200,0:-150,:] # femur
        #tar_img = tar_img[150:-200,50:-150,150:1050] # tibia
        
        ini_transform = cent_transform(ref_img,tar_img)
        
        metric_values = []
        multires_iterations = []

        suboutput = os.path.join(masterdir,imgtitle+" registered")

        print('Registration of {} is in process...'.format(imgtitle))
        try:
            tar_reg,tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform,imgtitle)
            if not os.path.exists(suboutput): os.mkdir(suboutput)
            imsaveseq(tar_reg,imgtitle+'_Reg',suboutput)
            print('Registration of {} is in completed...'.format(imgtitle))
        except RuntimeError as ex:
            print('Registration of {} failed...'.format(imgtitle))
            print(ex)
            pass
        print(datetime.datetime.now().time())