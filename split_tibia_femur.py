#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os
import SimpleITK as sitk
from shubow_tools import *
import re
import time
import concurrent.futures
import logging

def splitLRTF(folder,imgtitle,outfd = None):
    logging.info("Reading {}".format(imgtitle))
    img = imreadseq_multithread(folder,sitkimg= False, rmbckgrd=60)
    width = img.shape[2]
    mid_idx = int(width/2)

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
        if not os.path.exists(fd):
            os.mkdir(fd)

    ##### Save right tibia and femur #####
    logging.info("Splitting Left")
    left =  auto_crop(Rotate_by_Euler_angles(img[:,:,:mid_idx]))
    logging.info("Processing...")
    z_index_splt_left=np.argmin((left.mean(axis=(1,2)))[1200:1800])+1200 
    left_tibia = sitk.GetImageFromArray(left[:z_index_splt_left])
    left_femur = sitk.GetImageFromArray(auto_crop(Rotate_by_Euler_angles(left[z_index_splt_left:])))
    del left
    # use multiple threads to accelerate writng images to disk.
    # create an iterable to be passed to imreadseq(img,fd,title)
    titlelist = [imgtitle+' left tibia', imgtitle+' left femur',
                 imgtitle+' right tibia',imgtitle+' right femur']
    imagelist = [left_tibia,left_femur]
    logging.info("Writing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        '''
        for i in range(4):
            executor.submit(imsaveseq,imagelist[i],pathlist[i],titlelist[i])
        '''
        executor.map(imsaveseq,imagelist,pathlist[:2],titlelist[:2])
    del left_tibia,left_femur
    
    ##### Save right tibia and femur #####
    logging.info("Splitting right")
    right = auto_crop(Rotate_by_Euler_angles(img[:,:,-1:mid_idx:-1]))
    logging.info("Processing...")
    z_index_splt_right=np.argmin((right.mean(axis=(1,2)))[1200:1800])+1200
    right_tibia = sitk.GetImageFromArray(right[:z_index_splt_right])
    right_femur = sitk.GetImageFromArray(auto_crop(Rotate_by_Euler_angles(right[z_index_splt_right:])))
    del right
    imagelist = [right_tibia,right_femur]
    logging.info("Writing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(imsaveseq,imagelist,pathlist[2:],titlelist[2:])

    del right_tibia,right_femur

if __name__ == "__main__":
    masterfolder = r'/media/spl/Seagate MicroCT/Yoda1-tumor 1.24.2020/Reconstruction week 3'
    masterout = r'/media/spl/Seagate MicroCT/Yoda1-tumor 1.24.2020/Tibia femur split week 3'
    time1 = time.time()
    count = 0

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    for folder in sorted(os.listdir(masterfolder))[4:]:
        count += 1
        ID = os.path.basename(folder)[0:10]
        logging.info('Cropping for {} started.'.format(ID))
        splitLRTF(os.path.join(masterfolder,folder),ID,masterout)
        logging.info('Cropping for {} is completed.'.format(ID))
    
    time2 = time.time()
    logging.info("Average time used: {: >8.1f} seconds".format((time2-time1)/count)) 