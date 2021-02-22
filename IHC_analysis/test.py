#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

from __future__ import print_function
import histomicstk as htk
import numpy as np
import skimage.io
import skimage.measure
import skimage.color
from skimage.color import rgb2grey
import cv2
from matplotlib import pyplot as plt
import os
import re

def __define_conv_matrix(stains = ['hematoxylin', 'dab','null']):
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    conv_matrix = conv_matrix = np.array([stain_color_map[st] for st in stains]).T
    return conv_matrix

def color_deconvolution(image, conv_matrix, output_dir="."):
    image = cv2.GaussianBlur(image, (3,3), 0)
    img_deconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(image, conv_matrix)

    return img_deconvolved.Stains

def vectorized_roi(image):
    grey_img = np.uint8(rgb2grey(image)*255)
    roi = grey_img <155
    roi = roi.flatten()
    roi_indices = np.argwhere(roi==True)
    return roi, roi_indices

def kmeans_segmentation(image, k, attempts = 15, color = np.uint8([[160,80,0],[127,171,236],[220,220,220]])):

    shape=image.shape
    roi, roi_indices = vectorized_roi(image)
    vectorized = np.float32(image.reshape(-1,3))[roi]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans (vectorized, k,None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(rgb2grey(center))
    DAB_index = np.argsort(center[:,1])[0]
    He_index = np.argsort(center[:,0])[0]  

    if He_index == DAB_index:
        He_index = np.argsort(center[:,0])[1]  

    bk_index = np.delete(np.array([0,1,2]),[DAB_index,He_index])[0]
    center[DAB_index]=color[0]
    center[He_index]=color[1]
    #center[bk_index]=color[2]   

    color_label = center[label.flatten()]

    
    rec_img = np.full(image.shape,255, dtype=np.uint8).reshape(-1,3)
    for index, color in zip(roi_indices, color_label):
        rec_img[index] = color  

    rec_img=rec_img.reshape(shape)

    DABpixel = np.count_nonzero(label == DAB_index)
    Hepixel = np.count_nonzero(label == He_index)
    bkgrdpixel = np.count_nonzero(label == bk_index)
    DAB_ratio = float(DABpixel / (DABpixel + Hepixel))

    #rec_img = np.where(rec_img[:] ==(0,0,0),(255,255,255),  rec_img[:])
    return rec_img, DAB_ratio

if __name__ == "__main__":
    conv_matrix = __define_conv_matrix()
    image = skimage.io.imread("/home/shubow/Desktop/test.tif")
    deconv_img = color_deconvolution(image, conv_matrix)
    clustered_image, DAB_ratio = kmeans_segmentation(image,2)
    skimage.io.imsave("/home/shubow/Desktop/test_kmeans.tif", clustered_image)
    plt.imshow(clustered_image)
    
    
