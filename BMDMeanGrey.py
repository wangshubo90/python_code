from operator import index
import os, glob, re
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import mean, std
from skimage.io import imread
import tqdm
import pandas as pd

def imageToHist(image:np.ndarray, bins=75):
    
    image1d = image.flatten()
    masked = image1d[(image1d > 3) * (image1d < 75)]
    hist, bin = np.histogram(masked, bins=bins, range=(0,255), density=False)

    return hist, bin, masked

def checkReconHist(reconfd):
    imagels = sorted(os.listdir(reconfd))
    samplels = imagels[200:-200:100]
    images = [imread(os.path.join(reconfd, i)) for i in samplels]
    histls = []
    means = []
    stds = []
    for image in images:
        hist, _, masked= imageToHist(image, bins = 60)
        histls.append(hist)
        means.append(masked.mean())
        stds.append(masked.std())

    mean = np.array(means).mean()
    std = np.array(stds).mean()

    histavg = np.array(histls).sum(axis=0)
    histavg = histavg/histavg.sum()

    return hist, mean, std

def batchRun(masterdir, dict):

    for fd in tqdm.tqdm(sorted(os.listdir(masterdir))):
        ID = re.search(r"(\d{3}).(week \d)", fd).group(0)
        hist , mean, std= checkReconHist(os.path.join(masterdir, fd))
        
        dict["ID"].append(ID)
        dict["mean"].append(mean)
        dict["stdev"].append(std)
        dict["hist"].append(list(hist))

    return 

if __name__=="__main__":
    import matplotlib.pyplot as plt

    from skimage.io import imread
    #fd =  r'D:\MicroCT_data\4th batch bone mets loading study\Reconstruction week 0\418 week 0 _Rec'
    masterdir = r"\\lywanglab\Micro_CT_Data\Micro CT reconstruction\3rd batch reconstruction Dec 2019 8.0N\3rd batch week 2 reconstruction"
    
    dict = {
        "ID":[], "mean":[], "stdev":[], "hist":[]
        }

    for i in range(2,5):
        batchRun(masterdir.replace("week 2", "week "+str(i)), dict)
    
    df = pd.DataFrame(dict)
    df.describe()
    df.to_csv(r"d:\8.0N loading softissue mean grey value.csv", index=False, mode='w')
    # plt.plot(np.linspace(0, 255, 60), hist)




