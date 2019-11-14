#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from skimage import io

recon_path = 'D:\\MicroCT data\\4th batch bone mets loading study\\Reconstruction week 2'
#recon_path = '/media/spl/D/MicroCT data/4th batch bone mets loading study/Reconstruction week 2'
LRmaster_path = os.path.join(recon_path,'..', 'L & R week 2')
if not os.path.exists(LRmaster_path):
    os.mkdir(LRmaster_path)

def splitrecon(folder):
    global recon_path, LRmaster_path
    image_list = os.listdir(os.path.join(recon_path,folder))
   
    for image in sorted(image_list)[950:-1061]:
        if '000' in image:
            Image = cv2.imread(os.path.join(recon_path,folder,image),0)
            half = int(Image.shape[1]/2)
            lhalf = half - 150
            rhalf = half + 150
            left = Image[125:-225, 200:lhalf]
            right = Image[125:-225,-200:rhalf:-1]
            
            left_fpath = os.path.join(LRmaster_path,folder[:-4]+' left')
            right_fpath = os.path.join(LRmaster_path,folder[:-4]+' right')
            if not os.path.exists(left_fpath):
                os.mkdir(left_fpath)
            
            if not os.path.exists(right_fpath):
                os.mkdir(right_fpath)
            io.imsave(os.path.join(left_fpath,image[:10]+' left '+image[11:-4]+'.tif'),left)
            io.imsave(os.path.join(right_fpath,image[:10]+' right '+image[11:-4]+'.tif'),right)
            
    print('Recon_split for '+folder+' is completed')


for folder in sorted(os.listdir(recon_path)):
    #if folder.endswith('Rec'):
    if folder[0:3] in ['446']:
        splitrecon(folder)

print('done!')