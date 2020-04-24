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
from ipywidgets import interact, fixed #, IntSlider, interactive_output

def imreadseq(fdpath,sitkimg=True,rmbckgrd = None, z_range = None,seq_pattern=None) :
    '''
    Description: Read 2d image seqeunces as sitk image or ndarray
    Parameters: 
                fdpath: string, dir path
                sitkimg: binary, whether convert to sitk image object.
                rmbckgrd: int[0-255], remove pixels below a threshold
                z_range: list or ndarray with 2 elements indecating the lower and upper bound
    Returns: 3d ndarray / sitk image object
    '''
    assert os.path.exists(fdpath), "The folder doesn't exist!"
    images = []

    if seq_pattern is None:
        seq_pattern = re.compile(r"(00\d{4,6}).(tif|bmp|png)$")
    else:
        pass

    imglist = [image for image in sorted(glob.glob(os.path.join(fdpath,'*'))) 
                if seq_pattern.search(image)]

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

def imreadseq_multithread(fdpath,thread = 4,sitkimg = True, rmbckgrd = None, z_range=None,seq_pattern=None):
    '''
    
    '''
    images = []

    if seq_pattern is None:
        seq_pattern = re.compile(r"(00\d{4,6}).(tif|bmp|png)$")
    else:
        pass

    imglist = [p for p in sorted(glob.glob(os.path.join(fdpath,"*"))) if seq_pattern.search(p)] #

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
    if image.ndim == 3:
        # make a z-project as in ImageJ
        image2D = np.array(image.max(axis=0) > background, dtype = 'int')
    else:
        image2D = image

    ylen, xlen = image2D.shape #pylint:disable=unpacking-non-sequence

    xbin = image2D.sum(axis = 0)
    ybin = image2D.sum(axis = 1)

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
                        at the middle of z-axis
                    2. find the center of mass of the bottom
                    3. calculate Euler angles to rotate the object
                    4. determine a translation that takes the object to the center of resampling grid
    Args:  image: 3D np.array
    Returns:    cent_rotation : [x, y, z] 1D np.array, center of rotation
                [alpha,beta,theta]: [alpha, beta, gamma] 1D np.array, angles to rotate by x, y, z axis.
                translation = [x, y ,z]] 1D np.array, translation vector that takes the object to the center

    Note: as image is in the form of np.ndarray, indexing of image.shape is in the order of z,y,x
            however, the actual rotation and resampling will be done using simpleITK in which indexing of image.GetSize()
            is in the order of x,y,z. Thus outputs are all in the order of x, y, z.
    '''
    # input image should be a 3D ndarray
    z_o = int(image.shape[0]*0.75)   # center of rotation somewhere in the middle, like z*0.5
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

def rotate_by_euler_angles(image):
    '''
    Description: rotate a 3d image using simpleITK transformation to align
                    the object with z-axis. The original orientation is defined
                    by a vector from center of mass (COM) of the image(z=z_max/2)
                    to COM of the image(z=0) 
    Args: image, ndarray
    return(s)   : image, ndarray
    '''
    center,angles,translation = z_axis_alignment(image)
    rigid_euler = sitk.Euler3DTransform()
    rigid_euler.SetCenter(center)
    rigid_euler.SetRotation(*angles)
    rigid_euler.SetTranslation(translation)
    image=sitk.Cast(sitk.GetImageFromArray(image),sitk.sitkFloat32)
    # determine resampling grid size
    resample_size = [image.GetSize()[0],image.GetSize()[1],image.GetSize()[2]+int(abs(translation[2]))]
    resample_origin = image.GetOrigin()
    resample_spacing = image.GetSpacing()
    resample_direction = image.GetDirection()
    image=sitk.Resample(image,resample_size,rigid_euler,sitk.sitkLinear,
                        resample_origin, resample_spacing, resample_direction,sitk.sitkUInt8)
    image = sitk.GetArrayFromImage(image)
    return image

def down_scale(tar_img,down_scale_factor=1.0,new_dtype=sitk.sitkFloat32):
    '''
    Description:
        Use sitk.Resample method to extract an image with lower resolution
    Args:
        tar_img: sitk.Image / numpy.ndarray
        down_scale_factor:  float/double, 
    Returns:
        sitk.Image
    '''
    if type(tar_img) == np.ndarray:
        tar_img = sitk.GetImageFromArray(tar_img)

    dimension = sitk.Image.GetDimension(tar_img)
    idt_transform = sitk.Transform(dimension,sitk.sitkIdentity)
    resample_size = [int(i/down_scale_factor) for i in sitk.Image.GetSize(tar_img)]
    resample_spacing = [i*down_scale_factor for i in sitk.Image.GetSpacing(tar_img)]
    resample_origin = sitk.Image.GetOrigin(tar_img)
    resample_direction = sitk.Image.GetDirection(tar_img)
    new_img = sitk.Resample(sitk.Cast(tar_img,sitk.sitkFloat32),resample_size, idt_transform, sitk.sitkLinear,
                     resample_origin,resample_spacing,resample_direction,new_dtype)
    new_img = sitk.Cast(new_img,new_dtype)

    return new_img

def show_images(*args,**kwds):
    '''
    Description: 
        show multiple images simutaneously
    Args:
        *args: ndaray, multiple images in form of np.ndarray
        *kwds: additional keywords are passed to plt.figure() call.
    Return:
        fig : plt.Figure
        ax : plt.axes.Axes object or array of Axes objects. 
    '''

    n = len(args)
    rows = int((n-1)/4)+1

    if n < 4:
        columns = n
    else:
        columns = 4

    fig, ax = plt.subplots(rows,columns,**kwds)

    for i in range(n):
        evals, evecs, center = PCA(args[i])
        x1=np.linspace(0,evecs[0,0]*200,200)+center[0]
        y1=np.linspace(0,evecs[1,0]*200,200)+center[1]
        x2=np.linspace(0,evecs[0,1]*200*evals[1]/evals[0],200)+center[0]
        y2=np.linspace(0,evecs[1,1]*200*evals[1]/evals[0],200)+center[1]
        ax[i].imshow(args[i],cmap=plt.cm.Greys_r)
        ax[i].plot(x1, y1, 'red')
        ax[i].plot(x2, y2, 'blue')
        ax[i].axis("off")
        ax[i].title("Image {}".format(i))

    return fig, ax

def interact_display(*args):
    '''
    Description:
        Display multiple 3d images interactively
    Args:
        *args: ndarrays with np.dmin=3
    '''
    n = len(args)

    kwds_z = {"img_z_"+str(i+1): (0,args[i].shape[0]-1) for i in range(n)} 
    # the value is a tuple (start, stop) for ipywidgets.interact call
    kwds_npa = {"img_"+str(i+1): fixed(args[i]) for i in range(n)} 
    # the value is ipywidgets.fixed() for ipywidgets.interact call
    kwds_input = dict(kwds_z, **kwds_npa)
    # combine the two keys to be passed to display()

    def display(**kwds):
        
        m=int(len(kwds)/2) # this is the number of images to be displayed.
        key = list(kwds.keys()) # have to call list after Python3.7
        rows = int((m-1)/3)+1 # 3 images per row.

        if m < 3:
            columns = m
        else:
            columns = 3

        fig, ax = plt.subplots(rows,columns,figsize=(4*columns,3*rows))

        if m ==1:
            ax.imshow(kwds[key[0+m]][kwds[key[0]],:,:],cmap=plt.cm.Greys_r)
            ax.axis("off")
            ax.set_title("Image 1")
        else:
            for i in range(m):
                z_interact = kwds[key[i]] # retrieve z_index from the dictionary
                img = kwds[key[i+m]]    # retrieve img/np.array from the dictionary
                ax[i].imshow(img[z_interact,:,:],cmap=plt.cm.Greys_r)
                ax[i].set_title("Image {}".format(i+1))
                ax[i].axis("off")
        
        fig.show()
    
    interact(display, **kwds_input)

def PCA(image,threshold = 90):
    '''
    Desription: find the eigen vectors of a 2D image by PCA. 
    Args: 
            Image: 2d np.ndarray / sitk.Image()
            threshold: int, a grey value threshold to create a binary image.
    Returns: 
            evals: ndarray, each element is a eigein value with descending order
            evecs: ndarray, each column is a eigein vector
            center: ndarray, the center [x, y, z] of the image after thresholding
            note: corresponding eigein_values in descending order
    '''
    if type(image) == sitk.Image:
        image = sitk.GetArrayFromImage(image)
    elif type(image) == np.ndarray:
        pass

    coords = np.flip(np.vstack(np.nonzero(image>threshold)),axis = 0) # get coordinates
    center = coords.mean(axis=1,dtype=np.float64)   # get center
    centered_coords = np.subtract(coords,center.reshape(-1,1))  # get centered coordinates
    cov = np.cov(centered_coords)   # get covariance matrix
    evals, evecs = np.linalg.eig(cov)   
    sort_indices = np.argsort(evals)[::-1]

    return evals[sort_indices], evecs[:, sort_indices], center

def rotation_matrix(mv_coord, ref_coord):
    '''
    Description:
        Given two coordinates, find a rotation matrix that transform mv_coord to ref_coord
    Args:
        mv_coord: ndarray, dimension = 3
        ref_coord: ndarray, dimension = 3
    Return(s):
        rotation_matrix: ndarray, a rotation matrix that transforms a 3d vector
    '''

    def direction_cosine(vect1,vect2):

        return np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2))
    
    assert (mv_coord.shape == ref_coord.shape and mv_coord.shape[0] == mv_coord.shape[1]), "mv_coord and ref_coord need to square matrices with the same dimension!"
    
    dim = mv_coord.shape[0]
    _matrix = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(dim):
            _matrix[i,j] =direction_cosine(mv_coord[j],ref_coord[i]) 
    
    return _matrix.transpose()

def rotate_2D(center, angle):
    '''
    Description:
        rotate a 2D image by an angle clockwisely
    Args:
        image: ndarray or sitk.Image
        angle: float32, an angle in radian; for example, np.pi
    Returns:
        transfrom: sitk.transform
    '''
    transform = sitk.Euler2DTransform()
    transform.SetCenter(center)
    transform.SetAngle(angle)

    return transform

def resample_insitu(image,transform,interpolator = sitk.sitkLinear, sitkdtype=sitk.sitkFloat32):
    
    if type(image) == np.ndarray:
        image = sitk.GetImageFromArray(image)
    elif type(image) == sitk.Image:
        pass

    return sitk.Resample(image,image,transform,interpolator,sitkdtype)

def init_transform_PCA_new(tar_img, ref_img):
    '''
    Description:
        This function use PCA to find a rotation matrix that transform the tar_img to ref_img.
        sitk.Euler2DTransform or sitk.Euler3DTransform will be used.
    Args:
        tar_img: sitk.Image
        ref_img: sitk.Image
    Returns: 
        sitk.Transfrom(): an sitk.Transform object that can be used in registration or resampling
    '''

    eval_tar, evec_tar, center_tar = PCA(tar_img)
    _, evec_ref, center_ref = PCA(ref_img)

    if np.dot(evec_tar[:,0], evec_ref[:,0]) < 0:
        evec_tar[:,0] = evec_tar[:,0]*-1
    
    evec2 = np.copy(evec_tar)
    evec2[:,1] = evec2[:,1]*-1
    evec3 = np.copy(evec_tar)
    evec3[:,2] = evec2[:,2]*-1
    evec4 = np.copy(evec_tar)
    evec4[:,(1,2)] = evec2[:,(1,2)]*-1

    # setup a registration method to evaluate similarity matrics
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)    

    # setup a transformation method for registration
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_tar) # this is the rotation center   
    transform.SetTranslation(center_tar-center_ref)

    similarity_values = np.array([])
    rot_matrics = []

    for evec in [evec_tar,evec2,evec3, evec4]:

        matrix = rotation_matrix(evec, evec_ref)        # find the rotation matrix
        transform.SetMatrix(matrix.transpose().flatten())   # SetMatrix() take input as tuple or 1d-array
        registration.SetInitialTransform(transform,inPlace=False)
        rot_matrics.append(matrix)
        np.append(similarity_values, registration.MetricEvaluate(sitk.Cast(tar_img, sitk.sitkFloat32), sitk.Cast(ref_img, sitk.sitkFloat32)))

    od = np.argsort(similarity_values) # find the smallest similarity value and its corresponding rotation matrix
    # parse to transform
    transform.SetMatrix(rot_matrics[od[0]].transpose().flatten())

    return transform

def init_transform_PCA(tar_img, ref_img):
    '''
    Description:
        This function use PCA to find a rotation matrix that transform the tar_img to ref_img.
        sitk.Euler2DTransform or sitk.Euler3DTransform will be used.
    Args:
        tar_img: np.ndarray or sitk.Image
        ref_img: np.ndarray or sitk.Image
    Returns: 
        sitk.Transfrom(): an sitk.Transform object that can be used in registration or resampling
    '''

    eval_tar, evec_tar, center_tar = PCA(tar_img)
    _, evec_ref, center_ref = PCA(ref_img)

    if len(eval_tar) == 2:
        transform = sitk.Euler2DTransform()
    elif len(eval_tar) == 3:
        transform = sitk.Euler3DTransform()
    
    # the indexing oder is [x,y,z] in sitk and [z,y,x] in numpy. So we need to change it.
    matrix = rotation_matrix(evec_tar, evec_ref)

    transform.SetCenter(center_tar) # this is the rotation center
    transform.SetMatrix(matrix.flatten()) # SetMatrix() take input as tuple or 1d-array
    transform.SetTranslation(center_tar-center_ref) 

    return transform

def init_transform_best_angle(tar_img, ref_img, angles = None, z_translation = True):
    '''
    Description: 
        Given a list of angles, find the best initial transfromation with the smallest similarity value
    Args:
        tar_img, ref_img: sitk.Image() type, target and reference images
        angles: a list a radiant angles, by default 
    '''

    if angles is None:
        angles = np.arange(-6,2)*np.pi/6

    # Registration framework setup.
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Evaluate the similarity metric using the rotation parameter space sampling, translation remains the same for all.
    initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer( ref_img, tar_img,
                                                                                sitk.Euler3DTransform(), 
                                                                                sitk.CenteredTransformInitializerFilter.MOMENTS))

    # If trasition along z-axis is not prefered:
    if z_translation:
        pass
    else:
        trans = initial_transform.GetTranslation()
        trans = [trans[0], trans[1], 0.0]
        initial_transform.SetTranslation(trans)
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    similarity = np.array([])

    # Iterate over all other rotation parameter settings. 
    for i, angle in enumerate(angles):

        initial_transform.SetRotation(0.0,0.0,angle)
        registration_method.SetInitialTransform(initial_transform)
        similarity = np.append(similarity, registration_method.MetricEvaluate(ref_img, tar_img))
    
    od = np.argsort(similarity)
    initial_transform.SetRotation(0.0,0.0,angles[od[0]])

    return initial_transform