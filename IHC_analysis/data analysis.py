#! /home/spl/ml/ihc/bin/python

# -*- coding: utf-8 -*-

import os
import pandas as pd 

file = r'/media/spl/D/IHC data/3rd batch bone mets IHC/week 3/ki67/segmentation/Ki67_Results.csv'

df = pd.read_csv(file, sep=',', header = 0)
sample_list = []


for i, row in df.iterrows():
    sample = row["# ID"].split(sep = ' ')
    
    sample_list.append(sample[0] + ' ' + sample[1])

sample_set = set(sample_list)
df['Sample'] = sample_list

df = df.astype({'DAB':float, 'Hemotaxylin': float, 'Background': float
                            , 'DAB_ratio': float, })
'''
with pd.ExcelWriter("Results.xlsx") as writer:
    df.to_excel(writer, sheet_name = 'Endomucin')
'''
df.to_csv(file, index = False, header = True)

with open(file[:-4]+'2.csv', 'w') as f:
    for i in sample_set:
        f.write(i+'\n')

