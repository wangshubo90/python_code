#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import os
import re

def get_metadata(dataframe, col="Dataset", pattern=None, groupdf=None):

    """
    Desctiption: Get metadata from a given column in dataframe
        dataframe: Pandas dataframe
        col: column name
        pattern: re search pattern
    """
    if pattern == None:
        pattern = re.compile(r"(\d{3}).(week.\d).(left tibia|right tibia)")

    sample_list = []

    for sample in iter(dataframe[col]):
        sample_list.append(pattern.search(sample).groups())

    animal, time, LR = zip(*sample_list)

    dataframe.insert(0, "Limb", LR)
    dataframe.insert(0, "time (wk)", time)
    dataframe.insert(0, "Animal ET", animal)
    dataframe.insert(3, "LongName", list(map(lambda x,y: x+" "+y, animal,LR)))

    if not groupdf is None:
        dataframe=pd.merge(dataframe, groupdf, on="LongName", validate="m:1")

    return dataframe

def read_ctan(file): 
    df = pd.read_csv(file,sep=",", header =1,
                usecols=[0,6,7,8,14,15,16, 17,44], skiprows=[2,3])
    return df

if __name__ == "__main__":
    os.chdir(r'C:\Users\wangs\Google Drive\Yoda1 project')
    
    files = [
        [
            "Cortical bone lesions week 1 4.19.2021.txt",
            "Cortical bone lesions week 2 4.19.2021.txt",
            "Cortical bone lesions week 3 4.19.2021.txt",
            "Cortical bone lesions week 4 4.19.2021.txt"
        ]]
    #     [
    #         "Trabecular bone w0 3.16.2021 Yoda1 tumor loading.txt",
    #         "Trabecular bone w1 3.16.2021 Yoda1 tumor loading.txt",
    #         "Trabecular bone w2 3.19.2021 Yoda1 tumor loading.txt",
    #         "Trabecular bone w3 4.18.2021 Yoda1 tumor loading.txt",
    #         "Trabecular bone w4 4.18.2021 Yoda1 tumor loading.txt"
    #     ]
    # ]
    sheet_names = ['Cort Lesion']
    # sheet_names = [
    #     "Cort",
    #     "Trab"
    # ]

    output_excel = "Yoda1 Tumor Loading Microct results Cort Lesion.xlsx"
    groupdf = pd.read_excel(r"C:\Users\wangs\Google Drive\Yoda1 project\Yoda1 Loading Tumor Perforation Analysis.xlsx", usecols=range(5), header=0)


    with pd.ExcelWriter(output_excel) as writer:
        for file_list,sheet in zip(files, sheet_names):
            for file in file_list:
                df = read_ctan(file)
                df = get_metadata(df, groupdf=groupdf)
                
                try:
                    startrow = writer.sheets[sheet].max_row
                except KeyError:
                    startrow = 0

                header = True if startrow == 0 else False

                df.to_excel(
                        writer, 
                        sheet_name = sheet, 
                        index=False,
                        header=header,
                        startrow=startrow

            )