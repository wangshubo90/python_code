#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
import sys
from pathlib import Path
import re
from shubow_tools import imreadseq_multithread

if len(sys.argv) == 1:
    dir = Path.cwd()
elif len(sys.argv) == 2:
    dir = sys.argv[1]
    tar = Path.cwd()
elif len(sys.argv) == 3:
    dir, tar = sys.argv[1:]

pat = re.compile(r"(\d{3}).(week.\d).(left|right)")

for fd in os.listdir(dir):
    name = pat.match(fd)
    if not name is None:
        animalET=name.group(1)
        LR = name.group(3)[0].capitalize()+"T"
        time = name.group(2)[-1]
        ID = animalET+LR+"_w"+time
        img = imreadseq_multithread(os.path.join(dir,fd),rmbckgrd=65)
        sitk.WriteImage(img,os.path.join(tar,ID+".nii"))
    else:
        pass