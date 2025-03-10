from math import log10, sqrt
import SimpleITK as sitk
from skimage.metrics import structural_similarity, normalized_mutual_information
from sklearn.metrics import jaccard_score
import numpy as np
import torch as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(gt, pred):
    
    return structural_similarity(gt, pred, data_range=pred.max()-pred.min())

def Dice(gt, pred):
    j = jaccard_score(gt.flatten()>20, pred.flatten()>20)
    return 2*j / (j+1)

class LPIPS():
    def __init__(self) -> None:
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        
    def __call__(self, gt_img, pred_img):
        return self.lpips(gt_img, pred_img)

if __name__=="__main__":
    
    import os, glob, tqdm
    import pandas as pd

    DATA_ROOT = r"E:\35_um_data_100x100x48 niis\Data"
    
    predFileList = glob.glob(r"E:\35_um_data_100x100x48 niis\meshes\visualize_pred2\pred*.nii.gz")
    gtFileList = [os.path.join(os.path.dirname(i), os.path.basename(i)[4:]) for i in predFileList]
    
    results = {"PSNR": [], "SSIM": [], "NMI": [],
               "LPIPS": [], "Dice": [], "Tn": [], "Twk": []}
    
    lpips_f = lpips.LPIPS(net='alex')
    for p, g in tqdm.tqdm(zip(predFileList, gtFileList), total=len(gtFileList)):
        
        sample_name = os.path.basename(g)
        sample_id = sample_name.replace(".nii.gz", "")
        t0 = 0
        t = int(sample_id[-1])
        t_zero_nii = ""
        while t0 < 3:
            t_zero_nii = os.path.join(
                DATA_ROOT, sample_id[:-1]+f"{t0}.nii.gz")
            if os.path.exists(t_zero_nii):
                results["Tn"].append(
                    t-t0)
                break
            t0 += 1

        results["Twk"].append(t)

        pimg = sitk.GetArrayFromImage(sitk.ReadImage(p))
        gimg = sitk.GetArrayFromImage(sitk.ReadImage(g))
        
        results["PSNR"].append(PSNR(pimg, gimg))
        results["SSIM"].append(structural_similarity(gimg, pimg))
        results["NMI"].append(normalized_mutual_information(np.expand_dims(gimg, 0), np.expand_dims(pimg, 0)))
        gtensor = nn.Tensor(np.expand_dims(gimg, 0).astype(np.float32)[:, [12,24,36]]/255)
        ptensor = nn.Tensor(np.expand_dims(pimg, 0).astype(np.float32)[:, [12,24,36]]/255)
        results["LPIPS"].append(np.squeeze(
            lpips_f(gtensor.float(), ptensor.float()).detach().numpy()))
        results["Dice"].append(Dice(gimg, pimg))
        
    for k, v in results.items():
        v = np.array(v)
        print(f"{k}:{v.mean()} -std- {v.std()} -min- {v.min()} -max- {v.max()}")
        if k=="Dice":
            print(gtFileList[v.argmin()])

    df = pd.DataFrame(results)
    print(df.groupby("Twk").mean())
    print(df.groupby("Twk").count())
    print(df.groupby("Twk").std())
    print(df.describe())
