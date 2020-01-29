#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import os
import re


os.chdir(r'C:\Users\wangs\Desktop\Yoda1 small batch')

def read_ctan(file): 
    df = pd.read_csv(file,sep=",", header =1,
                usecols=[0,6,7,8,14,15,44],skiprows=[2,3])
    return df


femur_w0 = read_ctan(r'femural cortex week 0 Yoda1 small batch 1.28.2020.txt')
femur_w3 = read_ctan(r'femural cortex week 3 Yoda1 small batch 1.28.2020.txt')
tibia_w0 = read_ctan(r'tibial cortex week 0 Yoda1 small batch 1.28.2020.txt')
tibia_w3 = read_ctan(r'tibial cortex week 3 Yoda1 small batch 1.28.2020.txt')

dtype=["str",'float32','float32','float32','float32','float32','float32'] 
dtypedict={}
for i, colname in enumerate(femur_w0.columns):
    dtypedict[colname]=dtype[i]

femur_w0 = femur_w0.astype(dtypedict)
femur_w3 = femur_w3.astype(dtypedict)
tibia_w0 = tibia_w0.astype(dtypedict)
tibia_w3 = tibia_w3.astype(dtypedict)

pattern = re.compile(r"(\d{3}).(week.\d).(left|right)")
sample_list = []
for sample in iter(femur_w0.Dataset):
    sample_list.append(pattern.search(sample).groups())

animal, time, LR = zip(*sample_list)

femur_w0['Dataset'] = list(map(lambda x,y: x+" "+y, animal,LR))
femur_w3['Dataset'] = list(map(lambda x,y: x+" "+y, animal,LR))
tibia_w0['Dataset'] = list(map(lambda x,y: x+" "+y, animal,LR))
tibia_w3['Dataset'] = list(map(lambda x,y: x+" "+y, animal,LR))

femur_w0.insert(1,"Time",time)
femur_w3.insert(1,"Time",time)
tibia_w0.insert(1,"Time",time)
tibia_w3.insert(1,"Time",time)

with pd.ExcelWriter("Yoda1_CTan_results.xlsx") as writer:
    femur_w0.to_excel(writer, sheet_name = "Femur week 0")
    femur_w3.to_excel(writer, sheet_name = "Femur week 3")
    tibia_w0.to_excel(writer, sheet_name = "Tibia week 0")
    tibia_w3.to_excel(writer, sheet_name = "Tibia week 3")