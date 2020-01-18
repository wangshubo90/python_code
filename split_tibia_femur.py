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
    logging.info("start imreading")
    img = imreadseq_multithread(folder,sitkimg= False, rmbckgrd=60)
    width = img.shape[2]
    mid_idx = int(width/2)

    if not outfd is None:
        pass
    else:
        outfd = folder

    logging.info("start splitting")
    
    left_tibia =[]
    right_tibia =[]
    #left_tibia = auto_crop(Rotate_by_Euler_angles(img[150:(mid_idx-50), : ,0:-911]))
    #right_tibia = auto_crop(Rotate_by_Euler_angles(img[-150:(mid_idx+50):-1, : ,0:-911]))
    left_femur = sitk.GetImageFromArray(auto_crop(Rotate_by_Euler_angles(img[-911:-1, : ,150:(mid_idx-50)])))
    
    right_femur = sitk.GetImageFromArray(auto_crop(Rotate_by_Euler_angles(img[-911:-1, : ,-150:(mid_idx+50):-1])))
 
    # use multiple threads to accelerate writng images to disk.
    # create an iterable to be passed to imreadseq(img,fd,title)
    
    titlelist = [imgtitle+' left tibia',imgtitle+' right tibia',
                        imgtitle+' left femur',imgtitle+' right femur']
    imagelist = [left_tibia,right_tibia,left_femur,right_femur]
    pathlist = []
    pathlist.append(os.path.join(outfd,imgtitle+' left tibia'))
    pathlist.append(os.path.join(outfd,imgtitle+' right tibia') )
    pathlist.append(os.path.join(outfd,imgtitle+' left femur'))
    pathlist.append(os.path.join(outfd,imgtitle+' right femur'))
    
    titlelist = titlelist[2:]
    imagelist = imagelist[2:]
    pathlist = pathlist[2:]

    for fd in pathlist:
        if not os.path.exists(fd):
            os.mkdir(fd)

    logging.info("start imwriting")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        '''
        for i in range(4):
            executor.submit(imsaveseq,imagelist[i],pathlist[i],titlelist[i])
        '''
        executor.map(imsaveseq,imagelist,pathlist,titlelist)
    del img,left_tibia,right_tibia,left_femur,right_femur

if __name__ == "__main__":
    masterfolder = '/media/spl/D/MicroCT data/Yoda1-loading/Reconstruction week 4'
    masterout = '/media/spl/D/MicroCT data/Yoda1-loading/Tibia & Femur Rec week 4'
    time1 = time.time()
    count = 0

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    for folder in sorted(os.listdir(masterfolder))[0:1]:
        count += 1
        ID = os.path.basename(folder)[0:10]
        logging.info('Cropping for {} started.'.format(ID))
        splitLRTF(os.path.join(masterfolder,folder),ID,masterout)
        logging.info('Cropping for {} is completed.'.format(ID))
    
    time2 = time.time()
    logging.info("Average time used: {: >8.1f} seconds".format((time2-time1)/count))