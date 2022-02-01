import os, glob
import json
from random import sample
import pandas as pd
import numpy as np
from scipy import interpolate

def findStepByDisplacement(df, U3, n_index=2):
    u3np = np.abs(df["U3"].to_numpy() - U3)
    index = np.argsort(u3np)[0:n_index]
    subdf = df.loc[list(index)].sort_index()
    
    return subdf["step"].to_numpy(), subdf["U3"].to_numpy()

def interpolateOne(x, y, newspace, kind="linear"):
    interpolation1d = interpolate.interp1d(x, y, kind=kind) 
    y = interpolation1d(newspace)
    return y

def interpolateAll(x, allelement, newx):
    """
    x : shape(length_of_x)
    allelement: shape(length_of_x, elements, channels)
    newx: [] of discrete desired new x values
    """
    newy = np.zeros(shape=[len(newx), allelement.shape[1], allelement.shape[2]+1])
    newy[:,:,0] = newx
    for k, nx in enumerate(newx):
        for i in range(allelement.shape[1]):
            for j in range(allelement.shape[2]):
                y = interpolateOne(x, allelement[:, i, j], nx, kind="linear")
                newy[k,i,j+1] = y
    return newy

def interpolateWrapper(df, newx, strain, stress):
    """
    newx = [1000,2000,3000,5000]
    """
    ystrain_new = []
    ystress_new = []

    for nx in newx:
        index, x = findStepByDisplacement(df, nx, n_index=5)
        ystrain = strain[index]
        ystress = stress[index]

        ystrain_new.append(interpolateAll(x, ystrain, [nx,]))
        ystress_new.append(interpolateAll(x, ystress, [nx,]))

    ystrain_new = np.concatenate(ystrain_new, axis=0)
    ystress_new = np.concatenate(ystress_new, axis=0)

    return ystrain_new, ystress_new

def data_transform(df, tare_strain=0.0125):
    df.loc[:, "U3"] = df["U3"].apply(lambda x: (x*-1-tare_strain)/1.44*1E6)
    df.loc[:, "RF3"] = df["RF3"].apply(lambda x: x*-1)
    return df

if __name__=="__main__":
    fd = r"C:\Users\wangs\Documents\35_um_data_100x100x48 niis\abaqus_results\U3RF3jsons"
    data_list = glob.glob(os.path.join(fd, "[0-9]*.json"))  

    tare_strain = 0.0125
    # newx for interpolation
    newx = [1000, 2000, 3000, 5000]

    for i, dataf in enumerate(data_list):
        
        sample_name = os.path.basename(dataf)
        preddataf = os.path.join(os.path.dirname(dataf),"pred"+sample_name)
        if sample_name in ["431RT_w3_3um.json", "324RT_w4_3um.json", "322LT_w3_3um.json"]:
            data = json.load(open(dataf, "r"))
            preddata = json.load(open(preddataf, "r"))

            df = data_transform(pd.DataFrame(data), tare_strain=tare_strain)
            pdf = data_transform(pd.DataFrame(preddata), tare_strain=tare_strain)

            print(sample_name+"----")
            for nx in newx:
                print(str(nx)+"--"+str(findStepByDisplacement(df, nx, n_index=1)))
            print("pred"+sample_name+"----")
            for nx in newx:
                print(str(nx)+"--"+str(findStepByDisplacement(pdf, nx, n_index=1)))

        # strain = np.load(dataf[:-5]+"_strain.npy", allow_pickle=True)
        # stress = np.load(dataf[:-5]+"_stress.npy", allow_pickle=True)
        # pstrain = np.load(preddataf[:-5]+"_strain.npy", allow_pickle=True)
        # pstress = np.load(preddataf[:-5]+"_stress.npy", allow_pickle=True)
        
        # results = interpolateWrapper(df, newx, strain, stress)
        # presults = interpolateWrapper(pdf, newx, pstrain, pstress)       

        # np.save(dataf[:-5]+"_strain_1235k.npy", results[0])
        # np.save(dataf[:-5]+"_stress_1235k.npy", results[1])
        # np.save(preddataf[:-5]+"_strain_1235k.npy", presults[0])
        # np.save(preddataf[:-5]+"_stress_1235k.npy", presults[1])