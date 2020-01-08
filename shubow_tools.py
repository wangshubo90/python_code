#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-
import os
import numpy as np
import skimage
from skimage import io
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import datetime
import re
import concurrent.futures
import glob

def imreadseq(fdpath,sitkimg=True,rmbckgrd = None) :
    images = []
    
    for image in sorted(os.listdir(fdpath)):
        if re.search(r"(00\d{4,6})",image):
            simage = cv2.imread(os.path.join(fdpath,image),0)
            if not rmbckgrd is None:
                mask = simage > rmbckgrd
                simage = simage * mask
            images.append(simage)
    images = np.asarray(images)

    if sitkimg == True:
        images = sitk.GetImageFromArray(images)
    return images

def imsaveseq(images,fdpath,imgtitle, sitkimages=True):
    if sitkimages ==True:
        images = sitk.GetArrayFromImage(images)
    len = images.shape[0]
    for i in range(len):
        newimage = images[i,:,:].astype('uint8')
        skimage.io.imsave(os.path.join(fdpath,imgtitle+'%7.6d.tif' %(i+1)),newimage,check_contrast=False)
    #   skimage.io.imsave(os.path.join(outputsubdir,'{} {:0>6}.tif'.format(folder, (i+1))),newimage)

def imreadseq_multithread(fdpath,sitkimg = True, rmbckgrd = None):
    images = []
    imglist = [p for p in glob.glob(os.path.join(fdpath,"*")) if re.search(r"(00\d{4,6})",p)]
    imglist = sorted(imglist)

    def imreadgrey(imagepath):
        image_at_z=cv2.imread(imagepath,0)
        return image_at_z

    with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
        for idx,image in enumerate(executor.map(imreadgrey,imglist)):
            if not rmbckgrd is None:
                image = image * (image > rmbckgrd)
            
            images.append(image)

    images = np.array(images)
    if sitkimg == True:
        images = sitk.GetImageFromArray(images)

    return images
