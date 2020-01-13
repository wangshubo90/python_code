#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.io import imsave
from pandas.plotting import parallel_coordinates

os.chdir("/media/spl/D/Others/CTan_results_analysis")

cort_df = pd.read_excel("4th batch cort CTan results.xlsx",sheet_name="Tumor",usecols="A,B,C,J")

# cast correct data type to all columns
# df.astype() only accept data type/dictionary as argument for column/columns

dtypedict = {} #create empty dict
dtype_value = ["str","str","str","float32"] # specify dtypes for each column

for i,colname in enumerate(cort_df.columns):
    dtypedict[colname] = dtype_value[i] 

cort_df = cort_df.astype(dtypedict) # cast dtypes

cort_df = cort_df.set_index([cort_df.columns[i] for i in range(3)])
cort_df = cort_df.unstack("Time_wk")
cort_df.columns = [cort_df.columns[i][1] for i in range(len(cort_df.columns))]
cort_df = cort_df.T.reset_index().drop(('index',''), axis = 1)
cort_df.columns = list(map(lambda x: x[0]+' '+x[1],cort_df.columns))

cort_df[["439 left  Loaded","439 right Nonloaded"]] = cort_df[["439 right Nonloaded","439 left  Loaded"]] 
cort_df[["450 left  Loaded","450 right Nonloaded"]] = cort_df[["450 right Nonloaded","450 left  Loaded"]] 

# Initialize the figure
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')
plt.figure(figsize=(18,18))
# multiple line plot


for i in range(0,len(cort_df.columns),2):
    num = int(i/2)+1
    plt.subplot(5,5,num)
    plt.plot(list(cort_df.index),cort_df.iloc[:,i],'r--',
                list(cort_df.index),cort_df.iloc[:,i+1],'b-',
                    marker='', linewidth=1.9, alpha=0.9,label = cort_df.columns[i][0][:3])
    plt.xlim(-0.5,5)
    plt.ylim(0,100)
    plt.xticks(range(5))
    plt.yticks(range(0,110,20))

    if num in range(18) :
        plt.tick_params(labelbottom='off')
        
    if num not in list(range(1,22,5)) :
        plt.tick_params(labelleft='off')
    plt.title(cort_df.columns[i][:3], loc='center', fontsize=12, fontweight=0, color="black" )
    plt.legend(("Loaded","Nonloaded"), loc = "lower left")
    
#plt.savefig("diff.png")


loaded_df = cort_df.iloc[:,::2]
nonloaded_df = cort_df.iloc[:,1::2]

loaded_df.columns = [i[0:3] for i in loaded_df.columns]
nonloaded_df.columns = loaded_df.columns

diff_df = nonloaded_df.subtract(loaded_df)
n_colors = len(diff_df.columns)
plt.figure(figsize=(6,6))
palette = plt.get_cmap('Set1')
#f.set_color_cycle([palette(1.*i/n_colors) for i in range (n_colors)])

for i in range(len(diff_df.columns)):
    #plt.subplot(5,5,i+1)
    plt.plot(list(diff_df.index),diff_df.iloc[:,i],'b-',
            marker='',color=palette(i), linewidth=1.9, alpha=0.9,label = diff_df.columns)
    
    plt.legend(diff_df.columns,bbox_to_anchor=(1.2, 1), loc = "upper right",borderaxespad=0.)
plt.savefig("diff.png")
'''
new_cort_df = cort_df[["week 1","week 2","week 3", "week 4","load_nload"]]
plt.figure(figsize=(8,7))
plt.grid(None)
plt.rc('font',size=20)
parallel_coordinates(new_cort_df,"load_nload",color=('firebrick', 'steelblue'))
plt.legend(loc = 'lower left')
plt.ylabel("Osteolytic lesion Bv")
#plt.savefig('Osteolytic lesion Bv.png')
'''

'''
bldf = cort_df.iloc[:, 0:4]
bfdf = cort_df.iloc[:, [0,1,2,4]]
bldf = bldf.set_index(['Time_wk']+metas).unstack("Time_wk")
bfdf = bfdf.set_index(['Time_wk']+metas).unstack("Time_wk")

bldf = bldf.rename_axis(index = {"Time_wk": None}, columns={ 'Bone_lesion_volume': None}).reset_index()
bfdf = bfdf.rename_axis(index = {"Time_wk": None}, columns={ "Bone_formation_ volume": None}).reset_index()
bldf.columns = ["Dataset","load_nload", "week 1","week 2","week 3", "week 4"]
bfdf.columns = ["Dataset","load_nload", "week 1","week 2","week 3", "week 4"]
bldf=bldf.iloc[:,[2,3,4,5,1]]

plt.figure(figsize=(8,7))
plt.grid(None)
plt.rc('font',size=20)
parallel_coordinates(bldf,"load_nload",color=('firebrick', 'steelblue'))
plt.legend(loc = 'upper left')
plt.ylabel("Osteolytic lesion Bv")
plt.savefig('Osteolytic lesion Bv.png')

'''