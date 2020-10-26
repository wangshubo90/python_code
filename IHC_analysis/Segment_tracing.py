from skimage.io import imread, imsave
import os
import numpy as np
import sys
import matplotlib.pyplot as plt 

os.chdir(r"E:\DATA\Human Sample Silver Staining 9.28.2020\Lacuna measure")

for image_file in os.listdir():
    image=imread(image_file)
    shape=image.shape
    mask = np.ones(shape=(shape[0], shape[1]), dtype=np.uint8)
    print(mask.shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if image[y,x,0] > 240 and image[y,x,1] < 20 and image[y,x,2] <20:
                mask[y,x]=0


    imsave("mask_"+image_file, mask*255)
    