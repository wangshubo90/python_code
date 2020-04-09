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
    z_index = np.argsort(index[1400:2000])[0]+1400

    return z_index

def splitTF(folder,imgtitle,outfd = None):
    logging.info("Reading {}".format(imgtitle))
    img = imreadseq_multithread(folder, sitkimg= False, rmbckgrd=65, thread=2)
    logging.info("Processing...split Tibia and Femur")

    if not outfd is None:
        pass
    else:
        outfd = folder

    pathlist = []
    pathlist.append(os.path.join(outfd,imgtitle+' tibia' ))
    pathlist.append(os.path.join(outfd,imgtitle+' femur' ))

    for fd in pathlist:
        if os.path.exists(fd):
            shutil.rmtree(fd)
            os.mkdir(fd)
        else:
            os.mkdir(fd)

    titlelist = [imgtitle+' tibia', imgtitle+' femur']
    
    img = auto_crop(img)

    ##### Save tibia and femur #####
    z_index_splt=knee_join_z_index(img)
    tibia = sitk.GetImageFromArray(auto_crop(img[:z_index_splt]))
    femur = sitk.GetImageFromArray(auto_crop(img[z_index_splt:]))
    del img

    # use multiple threads to accelerate writng images to disk.
    # create an iterable to be passed to imreadseq(img,fd,title)
    imagelist = [tibia, femur]
    logging.info("Writing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(imsaveseq,imagelist,pathlist,titlelist)
    del tibia, femur
    
if __name__ == "__main__":
    masterfolder = r'/run/user/1000/gvfs/smb-share:server=lywanglab.local,share=micro_ct_data/Micro CT processed data/35n Treadmill Running/Treadmill35 L&R'
    masterout = r'/media/spl/D/MicroCT_data/Machine learning/Treadmill running 35n tibia and registration/Treadmill running 35n tibia'
    os.chdir(masterout)

    time1 = time.time()
    count = 0
    
    with open(r"failed.txt", "r") as file:
        retry = file.readlines()
    retry = [i[:-1] for i in retry]
    
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    failed = []

    for inputfd in glob.glob(os.path.join(masterfolder,"Treadmill*")):
        for folder in sorted(os.listdir(inputfd))[:]:
            if folder in ['351 week 0 left']:
                count += 1
                ID = os.path.basename(folder)
                logging.info('Cropping for {} started.'.format(ID))
                try:
                    splitTF(os.path.join(inputfd,folder),ID,masterout)
                    logging.info('Cropping for {} is completed.'.format(ID))
                except Exception as err:
                    print(err)
                    failed.append(folder)
                    logging.info('Cropping for {} failed.'.format(ID))
                    pass

    print(failed)
    time2 = time.time()
    logging.info("Average time used: {: >8.1f} seconds".format((time2-time1)/count)) 

    with open("failed.txt", "w") as f:
        for i in failed:
            f.write(i+"\n")