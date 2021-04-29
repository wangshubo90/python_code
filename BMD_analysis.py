import os, glob, re
import SimpleITK as sitk
from shubow_tools import imreadseq_multithread as imread
import pandas as pd

def greyToBMD(meanGreyValue):
    BMD = (meanGreyValue - 30.051) / 135.94
    return BMD

def measureMineralDensity(image, mask, greyToBMD):

    image = image * mask
    image1d = image.flatten()
    # print(image1d.shape)
    meanGreyTMD = image1d.mean()
    TMD = greyToBMD(meanGreyTMD)

    meanGreyBMD = image1d[image1d > 0].mean()
    # print(image1d[image1d > 0].shape)
    BMD = greyToBMD(meanGreyBMD)

    return meanGreyBMD, BMD, meanGreyTMD, TMD

def measurements(imagefd, maskfd):

    image = imread(imagefd)
    mask = imread(maskfd)

if __name__=="__main__":

    mask = sitk.GetArrayFromImage(imread(r"D:\MicroCT_data\4th batch bone mets loading study\Ref_tibia_ROI", z_range=[-300,None]))

    Mdir = r"D:\MicroCT_data\Machine learning\SITK_reg_7um" 
    maskMdir = r"D:\MicroCT_data\Machine learning\SITK_reg_7um" 
    # Mdir = r"D:\MicroCT_data\4th batch bone mets loading study\Registration week 0"
    # maskMdir = r"D:\MicroCT_data\4th batch bone mets loading study\Registration week 0"
    results = []
    key = ["ID", "time", "limb", "loading", "loading_sign", "trabMean_BMD", "trab_BMD", "trabMean_TMD", "trab_TMD", "cortMean_BMD", "cort_BMD", "cortMean_TMD", "cort_TMD"]

    for fd in sorted(os.listdir(Mdir)):
        if re.search(r"(\d{3}).(left|right).*?(week \d)", fd):

            match = re.search(r"(\d{3}).(left|right).*?(week \d)", fd)
            ID = match.group(1)
            time = match.group(3)
            limb = match.group(2) + " tibia"
            loading = "Loaded" if limb == "left tibia" else "Nonloaded"
            loading_sign = "+" if limb == "left tibia" else "-"

            #print("\t".join([ID, time, limb, loading])+"\n")
            first_time = "week 0" if int(ID) > 330 else "week 1"

            imagefile = fd
            trabmaskfile = os.path.join(maskMdir, re.sub(r"week \d", first_time, fd), "Trab-ROI")
            cortmaskfile = os.path.join(maskMdir, re.sub(r"week \d", first_time, fd), "Cort-ROI")

            trabmask = imread(os.path.join(Mdir, trabmaskfile), z_range=[-300,None])
            cortmask = imread(os.path.join(Mdir, cortmaskfile), z_range=[-300,None])
            image = imread(os.path.join(Mdir, imagefile), z_range=[-300,None])
            trabmask, cortmask, image = [sitk.GetArrayFromImage(i) for i in [trabmask, cortmask, image]]

            image = image * mask
            trabMean_BMD, trab_BMD, trabMean_TMD, trab_TMD = measureMineralDensity(image[-100:], trabmask[-100:], greyToBMD)
            cortMean_BMD, cort_BMD, cortMean_TMD, cort_TMD = measureMineralDensity(image[:100], cortmask[:100], greyToBMD)

            result = [ID, time, limb, loading, loading_sign, trabMean_BMD, trab_BMD, trabMean_TMD, trab_TMD, cortMean_BMD, cort_BMD, cortMean_TMD, cort_TMD]
            print("  ".join([f'{i}:{j:.5}' for i, j in zip(key, result)]) + "\n")
            results.append(result)

    df = pd.DataFrame(results, columns=key)
    df.to_csv(r"d:\8.0N loading BMD_2.csv", index = False, mode="w", header=True)
    # print("\t:".join([str(i) for i in results]))

    # from shubow_tools import imsaveseq
    # imsaveseq(sitk.GetImageFromArray(image*cortmask), r"D:\MicroCT_data\4th batch bone mets loading study\Registration week 0\test", "420 week 0 left registered cort")

