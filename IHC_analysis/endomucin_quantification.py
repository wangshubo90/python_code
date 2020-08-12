#! /home/spl/ml/ihc/bin/python

# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import cv2
import numpy as np
import skimage.color
import skimage.io
import skimage.measure
from skimage.color import rgb2grey

import histomicstk as htk

#_______global________

fdpath = r'/media/spl/UBUNTU 18_0/ki67 measure/test'    # target folder
#imgref_path = r'/home/spl/Desktop/Measurement-Endomucin/ref.tif'
#imgref = skimage.io.imread(imgref_path)

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map # construct deconvolution matrix
stains = ['hematoxylin', 'dab','null']
conv_matrix = np.array([stain_color_map[st] for st in stains]).T
#mean_ref, stdev_ref = htk.preprocessing.color_conversion.lab_mean_std(imgref) # image normalization


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # kmeans parameters
k=3
attempts=15
color = np.uint8([[160,80,0],[127,171,236],[220,220,220]])

dtype = [('Img_ID',(np.str_,40)), ('DAP_pixels', np.int32),
         ('Hematoxylin_pixels', np.int32),('Background_pixels', np.int32), ('DAB ratio', np.float32)]

#________batch_processing________
def ihc_quant(filename,fdpath):
        results = []
      #__image_preprocessing-filtering-normalization__
        imginput_name = filename[:-4]
        imginput_path = os.path.join(fdpath,filename)
        imginput = skimage.io.imread(imginput_path)[:,:,:3]

        imgoutput_name = imginput_name + '_segmented.tif' 
        imgoutput_path = os.path.join(fdpath,imgoutput_name)
    
        #imginput = htk.preprocessing.color_normalization.reinhard(imginput, mean_ref, stdev_ref)
          #	imginput = skimage.filters.gaussian(imginput, sigma = 2.0, truncate = 1/5)
        imginput = cv2.GaussianBlur(imginput, (5,5), 0)
        imginput = cv2.medianBlur(imginput, 5)
          #__Color_deconvolution__
        img_deconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(imginput, conv_matrix)
          #__Kmeans_segmentation__
        vectorized = np.float32(img_deconvolved.Stains.reshape((-1,3)))
        ret, label, center = cv2.kmeans (vectorized, k,None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
          #__reconstruct_image__
        center = np.uint8(rgb2grey(center))
        #print('center of '+imginput_name+' is',center)
        DAB_index = np.argsort(center[:,1])[0]
        He_index = np.argsort(center[:,0])[0]
        if He_index == DAB_index:
            He_index = np.argsort(center[:,0])[1]
        
        bk_index = np.delete(np.array([0,1,2]),[DAB_index,He_index])[0]
        center[DAB_index]=color[0]
        center[He_index]=color[1]
        center[bk_index]=color[2]
        #print(DAB_index,He_index,bk_index)
        rec_img = center[label.flatten()]
        rec_img = rec_img.reshape((imginput.shape))
        
          #__measure_stained_area__
        
        DABpixel = np.count_nonzero(label == DAB_index)
        Hepixel = np.count_nonzero(label == He_index)
        bkgrdpixel = np.count_nonzero(label == bk_index)
        DAB_ratio = float(DABpixel / (DABpixel + Hepixel))
        
        imgnormoutput_name = imginput_name + '_normalized2ref.tif'
        imgnormoutput_path = os.path.join(fdpath,imgnormoutput_name)
        #skimage.io.imsave(imgnormoutput_path, imginput)
        skimage.io.imsave(imgoutput_path, rec_img)         # output images
        result = np.array([(imginput_name, DABpixel, Hepixel, bkgrdpixel, DAB_ratio)], dtype=dtype)     # save measurements
        
        return result

results = []

for filename in sorted(os.listdir(fdpath)):  
    if filename.endswith(".tif"): 
        result = ihc_quant(filename, fdpath)
        results.append(result[0])
        continue
    else:
        continue

np.savetxt(os.path.join(fdpath,'Endomucin_Results.csv'),results, delimiter=',', 
           fmt = ['%s', '%d','%d','%d','%f'], header = 'ID,DAB,Hemotaxylin,Background,DAB_ratio')

import pandas as pd 
ID, DAB, H, B, ratio = zip(*results)
df = pd.DataFrame({'ID':ID, 'DAB':DAB, 'Hemotaxylin':H,'Background':B, 'DAB_ratio':ratio})

