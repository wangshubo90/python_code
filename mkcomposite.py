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

def extractmsk(img,mask):
    
    mask = np.where(mask>0,1,0)
    img = img * mask
    
    return img
    
def mkcomposite(refimg, tarimg, mask = None):
    '''
    Desciption:
    Parameters: refimg, ndarray with dimension = 3
                tarimg, ndarray with the shape as refimg
                mask, ndarray with the same shape as refimg. Default value is None.
    Returns:    comp, ndarray with the shape as refimg
    Note:
                tar only 120-0+120 = 240
                ref only 0-60+120 = 60
                tar-ref overlavp = 120-60+120 = 180
                background = 0-0+120 = 120  -->> 0
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
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    refimg = imreadseq(refdir,sitkimg=False,rmbckgrd=75)
    tarimg = imreadseq(tardir,sitkimg=False,rmbckgrd=75)
    
    composite = mkcomposite(refimg,tarimg,mask=mask)
    
    imsaveseq(composite,outdir,comptitle,sitkimages=False,idx_start=451)
    logging.info('Thread finished for '+comptitle)

if __name__ == "__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    ref = 'week 0'
    tar = 'week 3'
    refimgmasterdir = os.path.join(r'F:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg','VOI450-590_Registered femur '+ref+"_thred75") #pylint: disable=anomalous-backslash-in-string
    tarimgmasterdir = os.path.join(r'F:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg','VOI450-590_Registered femur '+tar) #pylint: disable=anomalous-backslash-in-string
    outputmasterdir = os.path.join(r'F:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg','femur w{}w{}composite_450-590_thred75'.format(ref[-1],tar[-1]))
    if not os.path.exists(outputmasterdir):
        os.mkdir(outputmasterdir)
    #tibia_only_mask = imreadseq('/media/spl/D/MicroCT data/4th batch bone mets loading study/Ref_tibia_ROI',sitkimg=False)

    tardirls = [os.path.join(tarimgmasterdir,i) for i in os.listdir(tarimgmasterdir) if re.search('week 3',i)]
    compdirls = [outputmasterdir]*len(tardirls)

    with concurrent.futures.ProcessPoolExecutor(max_workers = 3) as executor:
        logging.info('ProcessPool started')
        executor.map(batch_mkcomp, tardirls, compdirls)

    logging.info('Done!')