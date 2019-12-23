#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os
import SimpleITK as sitk
from Through_time_registration import imreadseq, imsaveseq
import re


def splitLRTF(folder,imgtitle,outfd = None):

    img = imreadseq(folder,rmbckgrd=60)
    width = img.GetWidth()
    mid_idx = int(width/2)

    if not outfd is None:
        pass
    else:
        outfd = folder

    LT_fd = os.path.join(outfd,imgtitle+' left tibia')
    RT_fd = os.path.join(outfd,imgtitle+' right tibia') 
    LF_fd = os.path.join(outfd,imgtitle+' left femur')
    RF_fd = os.path.join(outfd,imgtitle+' right femur')

    for fd in [LT_fd, RT_fd,LF_fd,RF_fd]:
        if not os.path.exists(fd):
            os.mkdir(fd)
    '''
    left_tibia = img[50:mid_idx,100:-100,0:-1111]
    imsaveseq(left_tibia,LT_fd,imgtitle+' left tibia')
    del left_tibia

    right_tibia = img[-50:mid_idx:-1,100:-100,0:-1111]
    imsaveseq(right_tibia,RT_fd,imgtitle+' right tibia')
    del right_tibia
    '''
    left_femur = img[50:mid_idx,100:-100,-1211:-1]
    imsaveseq(left_femur,LF_fd,imgtitle+' left femur')
    del left_femur

    right_femur = img[-50:mid_idx:-1,100:-100,-1211:-1]
    imsaveseq(right_femur,RF_fd,imgtitle+' right femur')
    del right_femur

if __name__ == "__main__":
    masterfolder = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/Reconstruction week 3'

    masterout = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/Tibia Femur fully seg'
    for folder in os.listdir(masterfolder):
        ID = os.path.basename(folder)[0:10]
        splitLRTF(os.path.join(masterfolder,folder),ID,masterout)
        print('Cropping for {} is completed.'.format(ID))