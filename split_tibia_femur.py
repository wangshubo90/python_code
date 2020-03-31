#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os
import SimpleITK as sitk
from shubow_tools import *
import re
import time
import concurrent.futures
import logging
import cv2
import shutil
import glob

def knee_join_z_index(limb):
    '''
    Desription:
        Given a 3D image of one limb, find the z_index to split tibia and femur
    Args:
        limb: ndarray, dimension = 3
    return: 
        z_index: integer
    '''
    # np.nonzero finds all (pixels>threshold)' index as a tuple
    # np.vstack cancatenate the tuple elements and returns a ndarray
    # np.std calculate standar deviation of x and y in each plane
    index = [np.std(np.vstack(np.nonzero(i>100)), axis = 1) for i in limb]
    # the sums up x^2 and y^2; this is the second order momentum / total numer of value
    index = np.array(list(map(lambda x:x[0]**2+x[1]**2,index)))
    z_index = np.argsort(index[1000:1700])[0]+1000

    return z_index

def LR_mid_x(image):
    '''
    Desription:
        Given a 3D reconstruction, find the x_index to split left and right
    Args:
        limb: ndarray, dimension = 3
    return: 
        center: ndarray, 
    '''       

    z_project = image.mean(axis = 0)
    xy_dex = np.vstack(np.nonzero(z_project))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # kmeans parameters
    k=2
    attempts=15
    ret, label, center = cv2.kmeans(np.float32(xy_dex.transpose()),k,
                            None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    mid_x_index = int(center[:, 1].sum()/2)

    return mid_x_index


def splitLRTF(folder,imgtitle,outfd = None):
    logging.info("Reading {}".format(imgtitle))
    img = imreadseq_multithread(folder, sitkimg= False, rmbckgrd=60, thread=2)
    logging.info("Processing...split LT & LF")
    width = img.shape[2]
    mid_idx = LR_mid_x(img)

    if not outfd is None:
        pass
    else:
        outfd = folder

    pathlist = []
    pathlist.append(os.path.join(outfd,imgtitle+' left tibia' ))
    pathlist.append(os.path.join(outfd,imgtitle+' left femur' ))
    pathlist.append(os.path.join(outfd,imgtitle+' right tibia'))
    pathlist.append(os.path.join(outfd,imgtitle+' right femur'))

    for fd in pathlist:
        if os.path.exists(fd):
            shutil.rmtree(fd)
            os.mkdir(fd)
        else:
            os.mkdir(fd)

    titlelist = [imgtitle+' left tibia', imgtitle+' left femur',
                imgtitle+' right tibia',imgtitle+' right femur']
    
    left = auto_crop(img[:,:,:mid_idx])
    right = auto_crop(img[:,:,-1:mid_idx:-1])
    del img

    ##### Save left tibia and femur #####
    z_index_splt_left=knee_join_z_index(left)
    left_tibia = sitk.GetImageFromArray(auto_crop(left[:z_index_splt_left]))
    left_femur = sitk.GetImageFromArray(auto_crop(rotate_by_euler_angles(left[z_index_splt_left:])))
    del left
    # use multiple threads to accelerate writng images to disk.
    # create an iterable to be passed to imreadseq(img,fd,title)
    imagelist = [left_tibia,left_femur]
    logging.info("Writing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(imsaveseq,imagelist,pathlist[:2],titlelist[:2])
    del left_tibia,left_femur
    
    ##### Save right tibia and femur #####
    logging.info("Processing...split RT & RF")
    z_index_splt_right=knee_join_z_index(right)
    right_tibia = sitk.GetImageFromArray(auto_crop(right[:z_index_splt_right]))
    right_femur = sitk.GetImageFromArray(auto_crop(right[z_index_splt_right:]))
    del right
    imagelist = [right_tibia,right_femur]
    logging.info("Writing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(imsaveseq,imagelist,pathlist[2:],titlelist[2:])

    del right_tibia,right_femur
    
if __name__ == "__main__":
    masterfolder = r'/run/user/1000/gvfs/smb-share:server=lywanglab.local,share=micro_ct_data/Micro CT reconstruction/Reconstruction  Heart July-2019'
    masterout = r'/run/user/1000/gvfs/smb-share:server=lywanglab.local,share=micro_ct_data/Micro CT reconstruction/Reconstruction  Heart July-2019/Heart July-2019 LR tibia and femur'
    time1 = time.time()
    count = 0

    #with open("/home/spl/uncompleted.txt", "r") as file:
    #    retry = file.readlines()

    retry = ["384 week 0"]

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    failed = []
    for inputfd in glob.glob(os.path.join(masterfolder,"Reconstruction*")):
        for folder in sorted(os.listdir(inputfd))[:]:
            if folder[:10] in retry:
                count += 1
                ID = os.path.basename(folder)[0:10]
                logging.info('Cropping for {} started.'.format(ID))
                try:
                    splitLRTF(os.path.join(inputfd,folder),ID,masterout)
                    logging.info('Cropping for {} is completed.'.format(ID))
                except Exception:
                    failed.append(folder)
                    logging.info('Cropping for {} failed.'.format(ID))
                    pass
    
    print(failed)
    time2 = time.time()
    logging.info("Average time used: {: >8.1f} seconds".format((time2-time1)/count)) 