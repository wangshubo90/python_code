#! /home/spl/ml/sitk/bin/python

import os
import shutil
import glob

masterdir = os.getcwd()
sampledir = os.path.join(masterdir,'sampleImg')

if not os.path.exists(sampledir):
    os.mkdir(sampledir)

failed = [] # use this one to output a list of failed registration

for folder in os.listdir(masterdir):

    if 'week' in folder:
        images = sorted(glob.glob(os.path.join(masterdir,folder,'*00*.tif')))
        try:
            distal=images[0]
            midline = images[-303]
            proximal=images[-1]
            sampleimg = [proximal,distal,midline]
            for img in sampleimg:
                shutil.copyfile(img,os.path.join(sampledir,os.path.basename(img)))
        except IndexError:
            print('No files for {}'.format(folder))
            failed.append(folder)
            continue

print('Done!')