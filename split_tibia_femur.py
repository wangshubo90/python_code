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
    img = imreadseq_multithread(folder, rmbckgrd=60)
    width = img.GetWidth()
    mid_idx = int(width/2)

    if not outfd is None:
        pass
    else:
        outfd = folder
    pathlist = []
    pathlist.append(os.path.join(outfd,imgtitle+' left tibia'))
    pathlist.append(os.path.join(outfd,imgtitle+' right tibia') )
    pathlist.append(os.path.join(outfd,imgtitle+' left femur'))
    pathlist.append(os.path.join(outfd,imgtitle+' right femur'))
    
    for fd in pathlist:
        if not os.path.exists(fd):
            os.mkdir(fd)
    logging.info("start splitting")

    left_tibia = img[150:(mid_idx-50),125:-125,0:-911]
    right_tibia = img[-150:(mid_idx+50):-1,125:-125,0:-911]
    left_femur = img[150:(mid_idx-50),125:-125,-1011:-1]
    right_femur = img[-150:(mid_idx+50):-1,125:-125,-1011:-1]

    # use multiple threads to accelerate writng images to disk.
    # create an iterable to be passed to imreadseq(img,fd,title)
    
    titlelist = [imgtitle+' left tibia',imgtitle+' right tibia',
                        imgtitle+' left femur',imgtitle+' right femur']
    imagelist = [left_tibia,right_tibia,left_femur,right_femur]
    logging.info("start imwriting")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        '''
        for i in range(4):
            executor.submit(imsaveseq,imagelist[i],pathlist[i],titlelist[i])
        '''
        executor.map(imsaveseq,imagelist,pathlist,titlelist)
    del img,left_tibia,right_tibia,left_femur,right_femur

if __name__ == "__main__":
    masterfolder = '/media/spl/D/MicroCT data/Yoda1-loading/Reconstruction week 0'
    masterout = '/media/spl/D/MicroCT data/Yoda1-loading/Tibia & Femur Rec week 0'
    time1 = time.time()
    count = 0

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    for folder in sorted(os.listdir(masterfolder)[2:4]):
        count += 1
        ID = os.path.basename(folder)[0:10]
        logging.info('Cropping for {} started.'.format(ID))
        splitLRTF(os.path.join(masterfolder,folder),ID,masterout)
        logging.info('Cropping for {} is completed.'.format(ID))
    
    time2 = time.time()
    logging.info("Average time used: {: >8.1f} seconds".format((time2-time1)/count))