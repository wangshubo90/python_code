#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os

os.chdir(r'G:\Treadmill HIF\3.14.2020')

for group in os.listdir():
    for slice in os.listdir(os.path.join(os.getcwd(),group)):
        for image in os.listdir(os.path.join(os.getcwd(),group,slice)):
            os.rename(os.path.join(os.getcwd(),group,slice,image),
                            os.path.join(os.getcwd(),group,slice,slice+'_'+image[-6:]))
