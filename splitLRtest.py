#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from skimage import io
from joblib import Parallel,delayed
import multiprocessing
import glob

def splitrecon(folder):
    global recon_path, LRmaster_path
   
    for image in sorted(os.listdir(folder))[1100:-911]:
        if '000' in image:  # the '000' makes sure that only images in the sequence are read
            Image = cv2.imread(os.path.join(folder,image),0)
            half = int(Image.shape[1]/2) # split the whole image into left and right
            lhalf = half - 150           # reduce the overall framesize by a certain amount of pixels
            rhalf = half #+ 150
            left = Image[125:-225, 200:lhalf]
            right = Image[125:-225,-350:rhalf:-1]
            
            left_fpath = os.path.join(LRmaster_path,os.path.basename(folder)[:-4]+' left')
            right_fpath = os.path.join(LRmaster_path,os.path.basename(folder)[:-4]+' right')
            if not os.path.exists(left_fpath):
                os.mkdir(left_fpath)
            
            if not os.path.exists(right_fpath):
                os.mkdir(right_fpath)
            io.imsave(os.path.join(left_fpath,image[:10]+' left '+image[11:-4]+'.tif'),left)
            io.imsave(os.path.join(right_fpath,image[:10]+' right '+image[11:-4]+'.tif'),right)
    return
            
    print('Recon_split for '+folder+' is completed')

if __name__ == "__main__":
    
    #recon_path = 'D:\\MicroCT data\\4th batch bone mets loading study\\Reconstruction week 2'
    recon_path = '/media/spl/D/MicroCT data/4th batch bone mets loading study/Reconstruction week 4'
    LRmaster_path = os.path.join(recon_path,'..', 'L & R week 4')
    if not os.path.exists(LRmaster_path):
        os.mkdir(LRmaster_path)

    #for folder in sorted(glob.glob(os.path.join(recon_path,"*Rec"))):
    #    if True:
    #    #if folder[0:3] in ['446']:
    #        splitrecon(folder)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(splitrecon)(i) 
                        for i in sorted(glob.glob(os.path.join(recon_path,"*Rec"))))        
    print('done!')