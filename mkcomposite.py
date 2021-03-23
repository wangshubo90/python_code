#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

from shubow_tools import imreadseq, imsaveseq
import os
import SimpleITK as sitk
import numpy as np
import multiprocessing
import glob
import re
import logging
import concurrent.futures
from pathlib import Path
import shutil

def extractmsk(img,mask):
    
    #mask = np.where(mask>0,1,0)
    #img = img * mask
    mask = mask > 0
    img = img*mask

    return img
    
def mkcomposite(refimg, tarimg, mask = None):
    '''
    Desciption:
    Parameters: refimg, ndarray with dimension = 3
                tarimg, ndarray with the shape as refimg
                mask, ndarray with the same shape as refimg. Default value is None.
    Returns:    comp, ndarray with the shape as refimg
    Note:
                tar only .0
                .....................................-0+120 = 240
                ref only 0-60+120 = 60
                tar-ref overlavp = 120-60+120 = 180
                background = 0-0+120 = 120  -->> 0

                tar [80 : 255]
                ref [(1-60) : (180-240)]
    '''
    if not mask is None:
        [refimg, tarimg] = [extractmsk(img,mask) for img in [refimg,tarimg]]
    else:
        pass

    refimg = np.where(refimg>75,60,0)
    tarimg = np.where(tarimg>75,120,0)
    
    comp = tarimg - refimg +120 
    comp = np.where(comp==120,0,comp) 
    comp.astype(np.uint8) 
    return comp

def batch_mkcomp(tardir,outputmasterdir,mask = None):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info('Thread--{}--started'.format(os.path.basename(tardir)))
    tarwk = re.search(r"week (\d)",tardir)
    refdir = tardir.replace("week 3","week 0",-1)
    tartitle = os.path.basename(tardir)
    logging.info('Thread--{}--mkdir'.format(os.path.basename(tardir)))
    comptitle = tartitle[:-11] + ' w{}w{}composite'.format(0,tarwk.group(1))
    outdir = os.path.join(outputmasterdir,comptitle)
    
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    os.mkdir(outdir)

    refimg = imreadseq(refdir,sitkimg=False,rmbckgrd=75, z_range=(-300,None))
    tarimg = imreadseq(tardir,sitkimg=False,rmbckgrd=75, z_range=(-300,None))
    
    composite = mkcomposite(refimg,tarimg,mask=mask)
    
    imsaveseq(composite,outdir,comptitle,sitkimages=False)
    logging.info('Thread finished for '+comptitle)

if __name__ == "__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    ref = 'week 0'
    tar = 'week 2'
    refimgmasterdir = os.path.join(r'E:\Yoda1-tumor-loading 2.26.2021','Registration '+ref) #pylint: disable=anomalous-backslash-in-string
    tarimgmasterdir = os.path.join(r'E:\Yoda1-tumor-loading 2.26.2021','Registration '+tar) #pylint: disable=anomalous-backslash-in-string
    outputmasterdir = os.path.join(r'E:\Yoda1-tumor-loading 2.26.2021','Tibia w{}w{}composite'.format(ref[-1],tar[-1]))

    if not os.path.exists(outputmasterdir):
        os.mkdir(outputmasterdir)
    
    tibia_only_mask = imreadseq(r'E:\Yoda1-tumor 1.24.2020\Tibia-ROI2', sitkimg=False, z_range=(-300,None)) # if not needed, set this to None

    tardirls = [os.path.join(tarimgmasterdir,i) for i in os.listdir(tarimgmasterdir) if re.search('week 2',i)]
    compdirls = [outputmasterdir]*len(tardirls)

    with concurrent.futures.ProcessPoolExecutor(max_workers = 2) as executor:
        logging.info('ProcessPool started')
        executor.map(batch_mkcomp, tardirls, compdirls, [tibia_only_mask]*len(tardirls))

    logging.info('Done!')