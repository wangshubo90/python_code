#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-
import os
import numpy as np
import skimage
from skimage import io
import SimpleITK as sitk
from cv2 import imread #pylint: disable=no-name-in-module
import matplotlib.pyplot as plt
import datetime
import re
import concurrent.futures
import glob
from scipy.ndimage.measurements import center_of_mass
import math

def imreadseq(fdpath,sitkimg=True,rmbckgrd = None, z_range = None) :
    images = []

    imglist = [image for image in sorted(glob.glob(os.path.join(fdpath,'*'))) 
                if re.search(r"(00\d{4,6}).(tif|bmp|png)$",image)]
    if z_range is None:
        z_down, z_up = [0,len(imglist)]
    else:
        z_down,z_up = z_range
    imglist=imglist[z_down:z_up]

    for image in imglist:
        simage = imread(os.path.join(fdpath,image),0)

    if z_range is None:
        z_down, z_up = [0,len(imglist)]
    else:
        z_down, z_up = z_range
    imglist=imglist[z_down:z_up]

    for image in imglist:
        simage = imread(image,0)
        if not rmbckgrd is None:
            mask = simage > rmbckgrd
            simage = simage * mask
        images.append(simage)
    images = np.asarray(images)

    if sitkimg == True:
        images = sitk.GetImageFromArray(images)
    return images

def imsaveseq(images,fdpath,imgtitle, sitkimages=True, idx_start=None):
    if sitkimages ==True:
        images = sitk.GetArrayFromImage(images)
    len = images.shape[0]

    if idx_start is None:
        idx_start = 1
    else:
        pass

    for i in range(len):
        newimage = images[i,:,:].astype('uint8')
        skimage.io.imsave(os.path.join(fdpath,imgtitle+'%7.6d.tif' %(i+idx_start)),newimage,check_contrast=False)
    #   skimage.io.imsave(os.path.join(outputsubdir,'{} {:0>6}.tif'.format(folder, (i+1))),newimage)

def imreadgrey(imagepath):
    image_at_z=imread(imagepath,0)
    return image_at_z

def imreadseq_multithread(fdpath,thread = 4,sitkimg = True, rmbckgrd = None, z_range=None):
    images = []
    imglist = [p for p in sorted(glob.glob(os.path.join(fdpath,"*"))) if re.search(r"(00\d{4,6}).*(tif|tiff|png|jmp)",p)]
    if z_range is None:
        z_down, z_up = [0,len(imglist)]
    else:
        z_down, z_up = z_range

    imglist=imglist[z_down:z_up]

    with concurrent.futures.ThreadPoolExecutor(max_workers = thread) as executor:
        for _, image in enumerate(executor.map(imreadgrey,imglist)):
            if not rmbckgrd is None:
                image = image * (image > rmbckgrd)
            
            images.append(image)

    images = np.array(images)
    if sitkimg == True:
        images = sitk.GetImageFromArray(images)

    return images

def auto_crop(image,background=120):
    '''
    Description: this function shrint the frame in x-y plane of a 3D image. 
                        Z-axis is not changed.
    Parameters: image: 3D, np.array
                background: int, default value 120, to be used to remove noise
    Returns:    image: ndarray
    '''
    # make a z-project as in ImageJ
    zstack = np.array(image.max(axis=0) > background, dtype = 'int')

    ylen, xlen = zstack.shape #pylint:disable=unpacking-non-sequence

    xbin = zstack.sum(axis = 0)
    ybin = zstack.sum(axis = 1)

    xl,*_, xr = np.where(xbin > int(0.02*ylen))[0]  # note : np.where() returns a tuple not a ndarray
    yl,*_, yr = np.where(ybin > int(0.02*xlen))[0]

    # if close to edges already, set as edges
    xl = max(0,xl-20)
    xr = min(xr+20,xlen)
    yl = max(0,yl-20)
    yr = min(yr+20,ylen)

    return image[:,yl:yr,xl:xr]

def z_axis_alignment(image):
    '''
    Description: adjust the orientation of the object by the following steps:
                    1. find the center of mass of the image 
                        in the middle of z-axis
                    2. find the center of mass of the bottom image
                    3. calculate Euler angles to rotate the object
                    4. determine a translation that takes the object to the center of resampling grid
    Parameter:  image: 3D np.array
    Returns:    cent_rotation : [x, y, z] 1D np.array, center of rotation
                [alpha,beta,theta]: [alpha, beta, gamma] 1D np.array, angles to rotate by x, y, z axis.
                translation = [x, y ,z]] 1D np.array, translation vector that takes the object to the center

    Note: as image is in the form of np.ndarray, indexing of image.shape is in the order of z,y,x
            however, the actual rotation and resampling will be done using simpleITK in which indexing of image.GetSize()
            is in the order of x,y,z. Thus outputs are all in the order of x, y, z.
    '''
    # input image should be a 3D ndarray
    z_o = int(image.shape[0]*0.5)   # center of rotation somewhere in the middle, like z*0.5
    y_o, x_o = center_of_mass(image[z_o])
    cent_rotation = np.array([x_o,y_o,z_o])
 
    # moving point is the center of mass of the bottom
    y_m, x_m = center_of_mass(image[0])
    moving_point = np.array([x_m, y_m, 0])
    #fixed vector is z-axis
    #fixed_vector = [0,0,-1] 
    # moving vector which will be rotated to align with fixed vector
    x, y, z = moving_point-cent_rotation 
    # three euler angle of rotation respectively about the X, Y and Z axis
    alpha = -y/math.fabs(y)*(math.acos(z/math.sqrt(y**2+z**2))-math.pi)
    beta = -x/math.fabs(x)*math.asin(x/math.sqrt(x**2+y**2+z**2))
    theta = 0
    
    # figure a translation to move the object to the center of a resampling grid
    mv_vector_norm = math.sqrt(x**2+y**2+z**2) # this is the length of the moving vector
    translation = cent_rotation-[image.shape[2]/2,image.shape[1]/2, mv_vector_norm]

    return cent_rotation, [alpha,beta,theta],translation

def Rotate_by_Euler_angles(image):
    '''
    Description: rotate a 3d image using simpleITK transformation to align
                    the object with z-axis. The original orientation is defined
                    by a vector from center of mass (COM) of the image(z=z_max/2)
                    to COM of the image(z=0) 
    parameter(s): image, ndarray
    return(s)   : image, ndarray
    '''
    center,angles,translation = z_axis_alignment(image)
    rigid_euler = sitk.Euler3DTransform()
    rigid_euler.SetCenter(center)
    rigid_euler.SetRotation(*angles)
    rigid_euler.SetTranslation(translation)
    image=sitk.Cast(sitk.GetImageFromArray(image),sitk.sitkFloat32)
    # determine resampling grid size
    resample_size = [image.GetSize()[0],image.GetSize()[1],image.GetSize()[2]+int(abs(translation[2])*2)]
    resample_origin = image.GetOrigin()
    resample_spacing = image.GetSpacing()
    resample_direction = image.GetDirection()
    image=sitk.Resample(image,resample_size,rigid_euler,sitk.sitkLinear,
                        resample_origin, resample_spacing, resample_direction,sitk.sitkUInt8)
    image = sitk.GetArrayFromImage(image)
    return image


