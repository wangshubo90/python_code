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


def imreadseq(fdpath,sitkimg=True,rmbckgrd = None) :
    images = []
    
    for image in sorted(os.listdir(fdpath)):
        if '000' in image:
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
        skimage.io.imsave(os.path.join(fdpath,imgtitle+'%7.6d.tif' %(i+1)),newimage)
    #   skimage.io.imsave(os.path.join(outputsubdir,'{} {:0>6}.tif'.format(folder, (i+1))),newimage)

def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

def end_plot( ):
    global metric_values, multires_iterations,imgtitle,outputsubdir
    
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.title(imgtitle)
    plt.savefig(os.path.join(outputsubdir,imgtitle+' regplot.png'))

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
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)

    #tar_resampled = sitk.Resample(tar_img, ref_img, initial_transform, sitk.sitkLinear, 0.0, tar_img.GetPixelID())
    return initial_transform

# registration transform
def reg_transform(ref_img,tar_img, ini_transform):
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.2,
                                                                numberOfIterations=500,
                                                                convergenceMinimumValue=1e-5,
                                                               convergenceWindowSize=5)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # or can do initial transform in place
    registration_method.SetInitialTransform(ini_transform, inPlace=False)

    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: update_metric_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(ref_img, sitk.sitkFloat32), sitk.Cast(tar_img, sitk.sitkFloat32))
    tar_resampled = sitk.Resample(tar_img, ref_img, final_transform, sitk.sitkLinear, 0.0, tar_img.GetPixelID())
    return tar_resampled, final_transform

if __name__ == "__main__":

    print(datetime.datetime.now())

    # masterinput, the dir that contains all the subfolders of scans
    masterdirpath = '/media/spl/D/MicroCT data/4th batch bone mets loading study/L & R week 3' 
    #masterdirpath = 'D:\\MicroCT data\\4th batch bone mets loading study\\L & R week 2' 

    masteroutput = os.path.join(masterdirpath,'..','Registration week 3')
    if not os.path.exists(masteroutput):
        os.mkdir(masteroutput)

    # load reference VOI
    refmasterdir = '/media/spl/D/MicroCT data/4th batch bone mets loading study/Registration week 0'
    #refmasterdir = 'D:\\MicroCT data\\4th batch bone mets loading study\\Registration week 0'

    failedreg = []
    for folder in sorted(os.listdir(masterdirpath)):
        if folder in ['443 week 3 right']: 
            #imgtitle = folder[:-11] 
            imgtitle = folder
            reftitle = folder[:9]+'0'+folder[10:]+' registered'
            metric_values = []
            multires_iterations = []
            outputsubdir = os.path.join(masteroutput, imgtitle+' registered')
            if not os.path.exists(outputsubdir):
                os.mkdir(outputsubdir)
            print('Registration of {} is in process...'.format(imgtitle))

            ref_img = imreadseq(os.path.join(refmasterdir, reftitle),rmbckgrd=60)
            tar_img = imreadseq(os.path.join(masterdirpath,imgtitle),rmbckgrd=60)
            

            ini_transform = cent_transform(ref_img, tar_img)
            #ini_transform = sitk.ReadTransform(os.path.join(outputsubdir,imgtitle+'reg_transform.tfm'))
        
            try:
                tar_reg, tar_reg_transform = reg_transform(ref_img,tar_img,ini_transform)
                print('Registration of {} is completed. Saving...'.format(imgtitle))
                imsaveseq(tar_reg,outputsubdir, imgtitle)
                sitk.WriteTransform(tar_reg_transform,os.path.join(outputsubdir,imgtitle+'reg_transform.tfm'))
                del ref_img,tar_img, tar_reg, tar_reg_transform, metric_values, multires_iterations
            except RuntimeError:
                failedreg.append(folder)
                print('Registration of {} failed...'.format(imgtitle))
                pass

            print(datetime.datetime.now().time())

    print('Done!')



    
