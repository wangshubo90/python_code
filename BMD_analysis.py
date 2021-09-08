import os, glob, re
import SimpleITK as sitk
from matplotlib.colors import Colormap
from numpy.core.fromnumeric import mean
from shubow_tools import imreadseq_multithread as imread
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def greyToBMD(meanGreyValue):
    BMD = (meanGreyValue - 11.868) / 165.05
    return BMD

def measureMineralDensity(image, mask, greyToBMD, addnoise=True, noisemean=0, noisestd=1):

    mask = mask.flatten()
    image1d = image.flatten()
    if addnoise:
        backgroundmask = image1d == 0
        image1d[backgroundmask] = np.clip(np.random.normal(noisemean, noisestd, size=backgroundmask.sum()), 0.0, 75.0)
    # print(image1d.shape)
    # plt.figure()
    # plt.imshow(image1d.reshape(image.shape)[0], cmap="gray")
    image1d = image1d * (mask>0)
    # plt.figure()
    # plt.imshow(image1d.reshape(image.shape)[0], cmap="gray")
    meanGreyBMD = image1d[image1d>0].mean()
    BMD = greyToBMD(meanGreyBMD)

    meanGreyTMD = image1d[image1d > 75].mean()
    # print(image1d[image1d > 0].shape)
    TMD = greyToBMD(meanGreyTMD)

    return meanGreyBMD, BMD, meanGreyTMD, TMD

def measurements(imagefd, maskfd):

    image = imread(imagefd)
    mask = imread(maskfd)

def addNoise(image, mean, std):
    image1d = image.flatten()
    backgroundmask = image1d == 0
    image1d[backgroundmask] = np.random.normal(mean, std, size=len(backgroundmask))

if __name__=="__main__":

    mask = sitk.GetArrayFromImage(imread(r"D:\MicroCT_data\4th batch bone mets loading study\Ref_tibia_ROI", z_range=[-300,-10]))

    Mdir = r"D:\MicroCT_data\Machine learning\SITK_reg_7um" 
    maskMdir = r"D:\MicroCT_data\Machine learning\SITK_reg_7um" 
    # MMdir = r"D:\MicroCT_data\4th batch bone mets loading study\Registration week "
    # maskMdir = r"D:\MicroCT_data\4th batch bone mets loading study\Registration week 0"
    results = []
    key = ["ID", "time", "limb", "loading", "loading_sign", "trabMean_BMD", "trab_BMD", "trabMean_TMD", "trab_TMD", "cortMean_BMD", "cort_BMD", "cortMean_TMD", "cort_TMD"]
    bgdf = pd.read_csv(r"d:\8.0N loading softissue mean grey value.csv").set_index("ID")

    for i in range(1):
        # Mdir = MMdir+str(i)
        for fd in tqdm.tqdm(sorted(os.listdir(Mdir))):
            if re.search(r"(3[0-2]\d).(week \d).(left|right)", fd):

                match = re.search(r"(\d{3}).(week \d).(left|right)", fd)
                ID = match.group(1)
                time = match.group(2)
                limb = match.group(3) + " tibia"
                loading = "Loaded" if limb == "left tibia" else "Nonloaded"
                loading_sign = "+" if limb == "left tibia" else "-"

                #print("\t".join([ID, time, limb, loading])+"\n")
                first_time = "week 0" if int(ID) > 330 else "week 1"

                imagefile = fd
                trabmaskfile = os.path.join(maskMdir, re.sub(r"week \d", first_time, fd), "Trab-ROI")
                cortmaskfile = os.path.join(maskMdir, re.sub(r"week \d", first_time, fd), "Cort-ROI")

                trabmask = imread(os.path.join(Mdir, trabmaskfile), z_range=[-300,-10])
                cortmask = imread(os.path.join(Mdir, cortmaskfile), z_range=[-300,-10])
                image = imread(os.path.join(Mdir, imagefile), z_range=[-300,-10])
                trabmask, cortmask, image = [sitk.GetArrayFromImage(i) for i in [trabmask, cortmask, image]]

                image = image * (mask>0)
                mean = bgdf.loc[ID + " " + time, "mean"]
                std = bgdf.loc[ID + " " + time, "stdev"]
                trabMean_BMD, trab_BMD, trabMean_TMD, trab_TMD = measureMineralDensity(image[-50:], trabmask[-50:], greyToBMD, addnoise=True, noisemean=mean, noisestd=std)
                cortMean_BMD, cort_BMD, cortMean_TMD, cort_TMD = measureMineralDensity(image[:250], cortmask[:250], greyToBMD, addnoise=True, noisemean=mean, noisestd=std)

                result = [ID, time, limb, loading, loading_sign, trabMean_BMD, trab_BMD, trabMean_TMD, trab_TMD, cortMean_BMD, cort_BMD, cortMean_TMD, cort_TMD]
                print("  ".join([f'{i}:{j:.5}' for i, j in zip(key, result)]) + "\n")
                results.append(result)

    df = pd.DataFrame(results, columns=key)
    df.to_csv(r"d:\8.0N loading BMD.csv", index = False, mode="w", header=True)
    # print("\t:".join([str(i) for i in results]))

    # from shubow_tools import imsaveseq
    # imsaveseq(sitk.GetImageFromArray(image*cortmask), r"D:\MicroCT_data\4th batch bone mets loading study\Registration week 0\test", "420 week 0 left registered cort")

