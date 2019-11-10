#! /home/spl/ml/ihc/bin/python

# -*- coding: utf-8 -*-

import numpy as np
import os
import re

cwd = os.getcwd()

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)',s)]

def sort_nicely(l):
    return sorted(l,key = alphanum_key)

filelist = os.listdir(cwd)

for file in sort_nicely(filelist):
    if file.endswith(".tif"):
        print(file)
        continue
    else:
        continue

print (cwd)