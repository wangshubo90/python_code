import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.io import imsave
from pandas.plotting import parallel_coordinates

os.chdir("D:\\Others\\CTan_results_analysis")

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

# Initialize the figure
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')
plt.figure(figsize=(20,20))
# multiple line plot
for i in range(0,len(cort_df.columns),2):
    num = int(i/2)+1
    plt.subplot(6,4,num)
    plt.plot(list(cort_df.index),cort_df.iloc[:,i],'b-',
                list(cort_df.index),cort_df.iloc[:,i+1],'r--',
                    marker='', linewidth=1.9, alpha=0.9,label = cort_df.columns[i][0][:3])
    plt.xlim(0,5)
    plt.ylim(0,100)
    plt.xticks(range(5))
    plt.yticks(range(0,100,10))

    if num in range(18) :
        plt.tick_params(labelbottom='off')
        
    if num not in list(range(1,22,4)) :
        plt.tick_params(labelleft='off')
    plt.title(cort_df.columns[i][0][:3], loc='center', fontsize=12, fontweight=0, color=palette(num) )

plt.savefig("Cort BvTv individual animal plots.png")
plt.legend()
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