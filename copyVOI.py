#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-
import os
import shutil
import glob
import re

def cp_VOI(fd,dst_fd,z_range):
    '''
    Descriptions: This function copies 2d image sequences in a specified range
    parameters: 
                fd: str, where images come from
                dst_fd: str, where images are copied to
                z_range: tuple or list with two elements
    '''
    imglist = [image for image in sorted(os.listdir(fd))
                 if re.search(r"(00\d{4,6}).(tif|bmp|png)$",image)]
    z_down,z_up = z_range
    imglist=imglist[z_down:z_up]

    if not os.path.exists(dst_fd):
        os.mkdir(dst_fd)
    for img in imglist:
        shutil.copy(os.path.join(fd,img),os.path.join(dst_fd,img))
    
def cp_VOI_batch(master_fd,master_dst,z_range):
    
    if not os.path.exists(master_dst):
        os.mkdir(master_dst)
    
    for folder in os.listdir(master_fd):
        cp_VOI(os.path.join(master_fd,folder),os.path.join(master_dst,folder),z_range)

if __name__ == "__main__":
    fd = r'E:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg\tibia w0w3composite_40-840'
    dst_fd = r'E:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg\tibia w0w3composite_724-774'

    cp_VOI_batch(fd,dst_fd,[724,774])