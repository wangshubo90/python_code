#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os
import numpy as np
import skimage
from skimage import io
import SimpleITK as sitk
from cv2 import imread
import matplotlib.pyplot as plt
import datetime
import re

#_______imreader for 2d image sequence_________________________________________
def imreadseq(fdpath,sitkimg=True,rmbckgrd = None) :
    images = []
    
    for image in sorted(os.listdir(fdpath)):
        if re.search(r"(00\d{4,6})",image):
            simage = imread(os.path.join(fdpath,image),0)
            if rmbckgrd:
                mask = simage > rmbckgrd
                simage = simage * mask
            images.append(simage)
    images = np.asarray(images)

    if sitkimg == True:
        images = sitk.GetImageFromArray(images)
    return images

#_______imsave for 2d .tif image sequence which can be read by CTan____________
def imsaveseq(images,imgtitle, suboutput):
    global masteroutput
    images = sitk.GetArrayFromImage(images)
    len = images.shape[0]
    for i in range(len):
        newimage = images[i,:,:].astype('uint8')
        skimage.io.imsave(os.path.join(suboutput,imgtitle+'%7.6d.tif' %(i+1)),newimage,check_contrast=False)
    #   skimage.io.imsave(os.path.join(suboutput,'{} {:0>6}.tif'.format(imgtitle, (i+1))),newimage)
#_______Define a couple of functions for sitk registration_____________________

def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

def end_plot(title,path ):
    global metric_values, multires_iterations,masteroutput, suboutput
    
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.title(title)
    plt.savefig(os.path.join(path,title+' regplot.png'))

    #del metric_values
    #del multires_iterations
    plt.close()

def update_metric_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                        

def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

# initial transformation to overlap the center of tar_img to ref_img
def cent_transform (ref_img,tar_img):
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(ref_img,sitk.sitkFloat32), 
                                                      sitk.Cast(tar_img,sitk.sitkFloat32), 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    #tar_resampled = sitk.Resample(tar_img, ref_img, initial_transform, sitk.sitkLinear, 0.0, tar_img.GetPixelID())
    return initial_transform

# registration transform
def reg_transform(ref_img,tar_img, ini_transform, imgtitle,suboutput, resample_ref = None):
      # get sitk image from folderID
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.40)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.3,
                                                                numberOfIterations=200,
                                                                convergenceMinimumValue=1e-5,
                                                               convergenceWindowSize=5)
    '''
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.3,
                                                            numberOfIterations=500,
                                                            convergenceMinimumValue=1e-6,
                                                            convergenceWindowSize=10)'''
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # or can do initial transform in place
    registration_method.SetInitialTransform(ini_transform, inPlace=False)

    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, lambda: end_plot(imgtitle,suboutput))
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: update_metric_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(ref_img, sitk.sitkFloat32), sitk.Cast(tar_img, sitk.sitkFloat32))
    
    if resample_ref is None:
        resample_ref = ref_img
        
    tar_resampled = sitk.Resample(tar_img, resample_ref, final_transform, sitk.sitkLinear, 0.0, tar_img.GetPixelID())
    return tar_resampled, final_transform

if __name__ == "__main__":
    

    print(datetime.datetime.now())

    # masterinput, the dir that contains all the subfolders of scans
    masterdirpath = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/L & R tibia week 0' 
    masteroutput = os.path.join(masterdirpath,'..','Registration tibia week 0')
    if not os.path.exists(masteroutput):
        os.mkdir(masteroutput)

    # load reference VOI
    Reference_img = imreadseq("/media/spl/D/MicroCT data/Yoda1 11.13.2019/Ref_adult_male_tibia", rmbckgrd=60)
    #Reference_img = sitk.ReadImage('/media/spl/D/CT image analysis test/339 L week 1 VOI350.nii')

    for folder in sorted(os.listdir(masterdirpath)):
        if folder in ["413 day 1 right tibia"]:
            #folder = folder[:-11] 

            metric_values = []
            multires_iterations = []

            suboutput = os.path.join(masteroutput, folder+' registered')

            if not os.path.exists(suboutput):
                os.mkdir(suboutput)
            print('Registration of {} is in process...'.format(folder))
            tar_img = imreadseq(os.path.join(masterdirpath,folder),rmbckgrd=60)
            mask_tar = tar_img>70
            mask_tar = sitk.Cast(mask_tar, sitk.sitkFloat32)
            ini_transform = cent_transform(Reference_img, mask_tar)
            del mask_tar
            #ini_transform = sitk.ReadTransform(os.path.join(suboutput,folder+'reg_transform.tfm'))
        
            try:
                tar_reg, tar_reg_transform = reg_transform(Reference_img,tar_img,ini_transform,folder,suboutput)
                print('Registration of {} is completed. Saving...'.format(folder))
                imsaveseq(tar_reg, folder, suboutput)
                sitk.WriteTransform(tar_reg_transform,os.path.join(suboutput,folder+'reg_transform.tfm'))
                del tar_img, tar_reg, tar_reg_transform, metric_values, multires_iterations
            except RuntimeError:
                print('Registration of {} failed...'.format(folder))
                pass
            print(datetime.datetime.now().time())

    print('Done!')