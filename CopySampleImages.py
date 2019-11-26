#! /home/spl/ml/sitk/bin/python

import os
import shutil

masterdir = '/media/spl/D/MicroCT data/4th batch bone mets loading study/Registration week 4'
sampledir = os.path.join(masterdir,'sampleImg')

if not os.path.exists(sampledir):
    os.mkdir(sampledir)

failed = [] # use this one to output a list of failed registration

for folder in os.listdir(masterdir):

    if 'week' in folder:
        images = sorted(os.listdir(os.path.join(masterdir,folder)))

        try:
            distal=images[0]
            midline = images[-303]
            proximal=images[-3]
            sampleimg = [proximal,distal,midline]
            for img in sampleimg:
                shutil.copyfile(os.path.join(masterdir,folder,img),os.path.join(sampledir,img))
        except IndexError:
            print('No files for {}'.format(folder))
            failed.append(folder)
            continue

print('Done!')