import os
from shubow_tools import imreadseq_multithread as imread
import numpy as np 
import matplotlib.pyplot as plt
import SimpleITK as sitk

path = 'C:\\Users\\wangs\\Documents\\MicroCT data\\309 week 1 left tibia ref'
image = imread(path,sitkimg=False,rmbckgrd=60)
plt.set_cmap('inferno')
def auto_crop(image):
    image = np.array(image.max(axis=0) > 120, dtype=np.int)
    ylen, xlen = image.shape

    xbin = image.max(axis = 0)
    ybin = image.max(axis = 1)
    
    np.where(xbin==1)


    '''
    fig , ax =plt.subplots(1,2)
    
    ax[0].plot(range(xlen),xbin)
    ax[0].set_title("xbin")
    ax[1].plot(ybin,range(ylen))
    ax[1].set_title("ybin")

    fig2,hist = plt.subplots(2,1)
    hist[0].hist(xbin,bins = 20)
    hist[0].set_title("xbin")
    hist[1].hist(ybin,bins = 20)
    hist[1].set_title("ybin")
    '''
    return image[:,xl:-xr,yl:-yr]

find_edge(image)
plt.figure()
plt.imshow(image.max(axis=0))


