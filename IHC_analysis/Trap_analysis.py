import numpy as np 
from skimage.io import imread, imsave
import glob
import os

dir = r'E:\DATA\4th batch IHC\TRAP\Measurement'
os.chdir(dir)
files = glob.glob(r"*.tif")

img_list = []
dtype = [('Animal',(np.str_, 4)),('Group', (np.str_, 10)),('File', (np.str_, 35)) ,('Bone surface count', np.int64), ('Trap surface count', np.int64), ('Trap percentage', np.float32)]
results = []

for image in files:
    p = glob.glob(r'../*slice*/'+image[:-3]+'tif')
    p = p[0]
    p = os.path.dirname(p)
    fn = p[3:]
    #fn = os.path.basename(image)
    sample = fn[:4]
    group = fn[3]
    if group == 'L':
        group = 'Loaded'
    elif group == 'R':
        group = 'Nonloaded'

    image = imread(image)

    trap_count = 0
    trap_neg_count = 0
    img_list.append(fn)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            if image[i,j,0] >= 245 and image[i,j,1] <= 30 and image[i,j,2] <= 30:
                trap_count +=1

            if image[i,j,0] <= 30 and image[i,j,1] >= 245 and image[i,j,2] <= 30:
                trap_neg_count +=1
    
    print(trap_count, trap_neg_count)

    bone_count = trap_count + trap_neg_count
    trap_percent = trap_count / bone_count

    result = np.array([(sample, group, fn, bone_count, trap_count, trap_percent)], dtype=dtype)
    results.append(result[0])

np.savetxt('./results.txt', results, fmt=['%s','%s','%s', '%d', '%d', '%f'], delimiter=',',
         header = 'Animal,Group,Image name, Bone surface, Trap +, Trap + Percentage')