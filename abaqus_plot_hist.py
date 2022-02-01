import os, glob, re
import matplotlib.pyplot as plt 
import numpy as np
from skimage.io import imread

file = np.load(r"C:\Users\wangs\Downloads\241RT_w4_3um_stress.npy", allow_pickle=True)
fd = r"\\192.168.2.2\data_share\temporalAAEv5-2nd-16-32-64-128-blend100-dice_squared-moredata-bidirect-Att\visualize_pred"
os.chdir(fd)

def imshow3Dmodel(imgf1, imgf2):
    
    img1=imread(imgf1)
    img2=imread(imgf2)
    figure, axes = plt.subplots(1,2)
    axes[0].imshow(img1[:, 135:450])
    axes[0].set_axis_off()
    axes[1].imshow(img2[:, 135:450])
    axes[1].set_axis_off()
    plt.tight_layout()
    figure.suptitle(imgf1[:-4])
    return figure, axes

pngs = glob.glob("[0-9m]*.png")
ids = set([re.search(r"(.\d{2}.T)", i).groups()[0] for i in pngs])

for i in pngs:
    fig, axes = imshow3Dmodel(i, "pred"+i)
    fig.savefig("compare"+i, dpi=150)
    plt.show()
    plt.close()