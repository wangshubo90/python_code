import os
from shubow_tools import imreadseq_multithread as imread
import shubow_tools
from shubow_tools import auto_crop, z_axis_alignment,Rotate_by_Euler_angles
import numpy as np 
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage.measurements import center_of_mass
import math

if __name__ == "__main__":
    path = r'E:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg\week 0 femur\411 week 0 right femur'
    image = imread(path,sitkimg=False,rmbckgrd=60)
    image_original = image
    image = Rotate_by_Euler_angles(image)
    fig, ax = plt.subplots(2,1,figsize = (10,20))
    plt.set_cmap('inferno')
    ax[0].imshow(auto_crop(image).max(axis=0))
    ax[0].set_title('z-axis aligned')
    #ax[0].plot(*zip(*map(lambda x:center_of_mass(x),image)),'b-.-')
    ax[1].imshow(auto_crop(image_original).max(axis=0))
    ax[1].set_title('Original')
    plt.show()


