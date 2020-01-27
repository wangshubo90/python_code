#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

from shubow_tools import imreadseq, imsaveseq
import os, sys
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
    '''
    if not mask is None:
        [refimg, tarimg] = [extractmsk(img,mask) for img in [refimg,tarimg]]
    else:
        pass

    refimg = np.where(refimg>65,60,0)
    tarimg = np.where(tarimg>65,120,0)
    ''' 
    tar only 120-0+120 = 240
    ref only 0-60+120 = 60
    tar-ref overlavp = 120-60+120 = 180
    background = 0-0+120 = 120
    '''
    comp = tarimg - refimg +120 
    #set background==120 to 0
    comp = np.where(comp==120,0,comp) 
    comp.astype(np.uint8) 
    return comp

def batch_mkcomp(tardir,outputmasterdir,mask = None):
    logging.info('Thread started for %s',os.path.basename(tardir))
    pat = re.compile(r'week \d')
    refdir = re.sub(pat,"week 0",tardir,count = 3)
    refimg = imreadseq(refdir,sitkimg=False)
    tarimg = imreadseq(tardir,sitkimg=False)
    tartitle = os.path.basename(tardir)
    composite = mkcomposite(refimg,tarimg,mask=mask)
    comptitle = tartitle[:-11] + ' w{}w{}composite'.format(ref[-1],tar[-1])
    outdir = os.path.join(outputmasterdir,comptitle)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    imsaveseq(composite,outdir,comptitle,sitkimages=False)
    logging.info('Thread finished for '+comptitle)


if __name__ == "__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    ref = 'week 0'
    tar = 'week 3'
    refimgmasterdir = os.path.join('E:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg','Registered femur '+ref) #pylint: disable=anomalous-backslash-in-string
    tarimgmasterdir = os.path.join('E:\MicroCT data\Yoda1 small batch\Tibia Femur fully seg','Registered femur '+tar) #pylint: disable=anomalous-backslash-in-string
    outputmasterdir = os.path.join(tarimgmasterdir,'..','femur w{}w{}composite'.format(ref[-1],tar[-1]))
    if not os.path.exists(outputmasterdir):
        os.mkdir(outputmasterdir)
    #tibia_only_mask = imreadseq('/media/spl/D/MicroCT data/4th batch bone mets loading study/Ref_tibia_ROI',sitkimg=False)

    tardirls = [os.path.join(tarimgmasterdir,i) for i in os.listdir(tarimgmasterdir) if re.search('week 3',i)]
    compdirls = [outputmasterdir]*len(tardirls)

    with concurrent.futures.ProcessPoolExecutor(max_workers = 3) as executor:
        logging.info('ProcessPool started')
        executor.map(batch_mkcomp,tardirls,compdirls)
        #for i in range(len(compdirls)):
        #    executor.submit(lambda:batch_mkcomp(tardirls[i],compdirls[i]))
        #for a, b in zip(tardirls,compdirls):
        #    executor.submit(batch_mkcomp,a,b)

    logging.info('Done!')
    