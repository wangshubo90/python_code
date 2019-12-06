#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import shutil
import os

masterdir = '/media/spl/D/MicroCT data/Yoda1 11.13.2019/L & R week 3'

for folder in os.listdir(masterdir):
    src = os.path.join(masterdir,folder)
    dest = os.path.join(masterdir,folder+" tibia")
    shutil.move(src,dest)