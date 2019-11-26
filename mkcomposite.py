#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

from Through_time_registration import imreadseq, imsaveseq
import os, sys
import SimpleITK as sitk
import numpy as np
from joblib import Parallel,delayed
import multiprocessing
import glob

def extractmsk(img,mask):
    
    mask = np.where(mask>0,1,0)
    img = img * mask
    
    return img
    
def mkcomposite(refimg, tarimg, mask = None):
    if not mask is None:
        [refimg, tarimg] = [extractmsk(img,mask) for img in [refimg,tarimg]]
    else:
        pass

    refimg = np.where(refimg>65,80,0)
    tarimg = np.where(tarimg>65,120,0)
    ''' 
    tar only 120-0+120 = 240
    ref only 0-80+120 = 40
    tar-ref overlavp = 120-80+120 = 160
    background = 0-0+120 = 120
    '''
    comp = tarimg - refimg +120 
    #set background==120 to 0
    comp = np.where(comp==120,0,comp) 
    comp.astype(np.uint8) 
    return comp

def batch_mkcomp(folder):
    global ref, tar, refimgmasterdir, tarimgmasterdir, outputmasterdir,tibia_only_mask
    tartitle = folder
    reftitle = folder.replace(tar,ref)
    refimg = imreadseq(os.path.join(refimgmasterdir,reftitle),sitkimg=False)
    tarimg = imreadseq(os.path.join(tarimgmasterdir,tartitle),sitkimg=False)
    composite = mkcomposite(refimg,tarimg,mask = tibia_only_mask)
    comptitle = tartitle[:-11] + ' w{}w{}composite'.format(ref[-1],tar[-1])
    outputdir = os.path.join(outputmasterdir,comptitle)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        imsaveseq(composite,outputdir,comptitle,sitkimages=False)
    print(comptitle+' is saved!')


if __name__ == "__main__":
    ref = 'week 0'
    tar = 'week 4'
    refimgmasterdir = os.path.join('/media/spl/D/MicroCT data/4th batch bone mets loading study','Registration '+ref)
    tarimgmasterdir = os.path.join('/media/spl/D/MicroCT data/4th batch bone mets loading study','Registration '+tar)
    outputmasterdir = os.path.join(tarimgmasterdir,'..','w{}w{}composite'.format(ref[-1],tar[-1]))
    if not os.path.exists(outputmasterdir):
        os.mkdir(outputmasterdir)
    tibia_only_mask = imreadseq('/media/spl/D/MicroCT data/4th batch bone mets loading study/Ref_tibia_ROI',sitkimg=False)
    
    fdlist = []
    for folder in os.listdir(tarimgmasterdir):
        if 'week 4' in folder:
            fdlist.append(folder)

    # we parallel the for loop by multiprocessing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(batch_mkcomp)(i) 
                        for i in sorted(fdlist)) 
    
    # original for loop
    '''
    for folder in sorted(os.listdir(tarimgmasterdir)):
        if tar in folder:
            tartitle = folder
            reftitle = folder.replace(tar,ref)
            refimg = imreadseq(os.path.join(refimgmasterdir,reftitle),sitkimg=False)
            tarimg = imreadseq(os.path.join(tarimgmasterdir,tartitle),sitkimg=False)
            composite = mkcomposite(refimg,tarimg,mask = tibia_only_mask)
            comptitle = tartitle[:-11] + ' w{}w{}composite'.format(ref[-1],tar[-1])
            outputdir = os.path.join(outputmasterdir,comptitle)
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
            imsaveseq(composite,outputdir,comptitle,sitkimages=False)
            print(comptitle+' is saved!')
    '''
    
    print('Done!')