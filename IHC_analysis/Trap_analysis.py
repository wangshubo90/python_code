import numpy as np 
from skimage.io import imread, imsave
import glob
import os

dir = r'/media/spl/D/IHC data/3rd batch bone mets IHC/week 3/Trap 3rd batch w3/Bone surface/Mark'
os.chdir(dir)
files = glob.glob(r"*.png")

img_list = []
dtype = [('Animal',np.int32),('Group', (np.str_, 10)),('File', (np.str_, 35)) ,('Bone surface count', np.int64), ('Trap surface count', np.int64), ('Trap percentage', np.float32)]
results = []

for image in files:

    fn = os.path.basename(image)
    sample = fn[:3]
    group = fn[4]
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

np.savetxt('./results.txt', results, fmt=['%d','%s','%s', '%d', '%d', '%f'], delimiter=',',
         header = 'Animal,Group,Image name, Bone surface, Trap +, Trap + Percentage')

    