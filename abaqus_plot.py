import os, glob
import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import r2_score
import matplotlib.transforms as transforms

def plot_force_displacement(data, preddata, title, axes):
    axes.plot(data["U3"].to_numpy(), data["RF3"].to_numpy(), "-b")
    axes.plot(preddata["U3"].to_numpy(), preddata["RF3"].to_numpy(), "--r")
    axes.set_title(title)
    axes.set_xlim(left=0, right=5000)
    axes.relim()
    axes.autoscale_view()
    return axes

def energy_fraction(data, preddata, title, axes):
    axes.plot(data["U3"].to_numpy(), data["ALLSD"].to_numpy() / (data["ALLSE"].to_numpy()+1e-8), "-b")
    axes.plot(preddata["U3"].to_numpy(), preddata["ALLSD"].to_numpy() / (preddata["ALLSE"].to_numpy()+1e-8), "--r")
    axes.set_title(title)
    axes.set_xlim(left=0, right=5000)
    axes.set_ylim(top=1, bottom=0)
    axes.relim()
    axes.autoscale_view()
    return axes

def energy(data, preddata, title, axes):
    axes.plot(data["U3"].to_numpy(), data["ALLSD"].to_numpy(), "-b")
    axes.plot(data["U3"].to_numpy(), data["ALLSE"].to_numpy(), "-r")
    axes.plot(preddata["U3"].to_numpy(), preddata["ALLSD"].to_numpy() , "--b")
    axes.plot(preddata["U3"].to_numpy(), preddata["ALLSE"].to_numpy(), "--r")
    axes.set_xlim(left=0, right=5000)
    axes.relim()
    axes.set_title(title)
    axes.autoscale_view()
    return axes

def data_transform(df, tare_strain=0.0125):
    df.loc[:, "U3"] = df["U3"].apply(lambda x: (x*-1-tare_strain)/1.44*1E6)
    df.loc[:, "RF3"] = df["RF3"].apply(lambda x: x*-1)
    return df

def interpolate_force(df, newlinspace):
    x = df["U3"].to_numpy()
    y = df["RF3"].to_numpy()
    interpolation1d = interpolate.interp1d(x, y, kind='linear') 
    newRF3 = interpolation1d(newlinspace)
    newdf = pd.DataFrame({"U3":newlinspace, "RF3": newRF3}).set_index("U3")
    return newdf

def plot_scatter_with_trendline(axes, x, y):
    x, y = np.array(x), np.array(y)
    axes.scatter(x,y, c="b")
    x = x[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x, y)

    axes.plot(x, a*x, 'r-')
    text = f"$R^2 = {r2_score(y,x):0.3f}$"
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)

    axes.text(0.8, 0.8, text,
        fontsize=14, verticalalignment='top')
    return axes

def plot_scatter_with_trendline2(axes, x, y):
    axes.scatter(x,y)
    z = np.polyfit(x, y, 1)
    y_hat = np.poly1d(z)(x)

    axes.plot(x, y_hat, "r--", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)

    trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)

    axes.text(0.05, 0.95, text, transform=trans,
        fontsize=18, verticalalignment='top')
    return axes

def findStepByDisplacement(df, U3, n_index=2):
    u3np = np.abs(df["U3"].to_numpy() - U3)
    index = np.argsort(u3np)[0:n_index]
    subdf = df.loc[list(index)].sort_index()
    
    return subdf["step"].to_numpy(), subdf["U3"].to_numpy()

if __name__=="__main__":
    fd = r"C:\Users\wangs\Documents\35_um_data_100x100x48 niis\abaqus_results\U3RF3jsons"
    data_list = glob.glob(os.path.join(fd, "[0-9]*.json"))

    ultimate_force1k = {"GT":[], "Pred":[]}
    ultimate_force2k = {"GT":[], "Pred":[]}
    ultimate_force3k = {"GT":[], "Pred":[]}
    ultimate_force5k = {"GT":[], "Pred":[]}

    tare_strain = 0.012
    # newx for interpolation
    xleft = int(((0-tare_strain)/1.44*1E6//100+1)*100)
    xright = 11000
    newx = np.linspace(xleft, xright, (xright - xleft)//100+1)

    for i, dataf in enumerate(data_list):
        sample_name = os.path.basename(dataf)
        preddataf = os.path.join(os.path.dirname(dataf),"pred"+sample_name)
        data = json.load(open(dataf, "r"))
        preddata = json.load(open(preddataf, "r"))
        
        #====transform data====
        df = data_transform(pd.DataFrame(data), tare_strain=tare_strain)
        pdf = data_transform(pd.DataFrame(preddata), tare_strain=tare_strain)

        # print("--".join([sample_name,str(df.loc[df.shape[0]-1, "U3"]), str(pdf.loc[pdf.shape[0]-1, "U3"])]))
        # print("--".join([sample_name,str(df.loc[0, "U3"]), str(df.loc[0, "U3"])]))
        #====interpolate U3 and RF3====
        dfinterp = interpolate_force(df, newx)
        pdfinterp = interpolate_force(pdf, newx)

        #====extract RF3-GT vs RF3-pred at different U3 for each sample====
        a = dfinterp.loc[5000, "RF3"]/pdfinterp.loc[5000, "RF3"]
        if a<0.5 or a > 1.5: # remove some outliers
            print(dataf+":"+str(a))
        else:
            ultimate_force1k["GT"].append(dfinterp.loc[1000, "RF3"] )
            ultimate_force1k["Pred"].append(pdfinterp.loc[1000, "RF3"] )

            ultimate_force2k["GT"].append(dfinterp.loc[2000, "RF3"] )
            ultimate_force2k["Pred"].append(pdfinterp.loc[2000, "RF3"] )
            
            ultimate_force3k["GT"].append(dfinterp.loc[3000, "RF3"] )
            ultimate_force3k["Pred"].append(pdfinterp.loc[3000, "RF3"] )
            
            ultimate_force5k["GT"].append(dfinterp.loc[5000, "RF3"] )
            ultimate_force5k["Pred"].append(pdfinterp.loc[5000, "RF3"] )

        #====plot RF3, energy vs U3 for each pair of samples====
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        _=plot_force_displacement(df, pdf, "Force vs Displacement", axes[0])
        _=energy_fraction(df, pdf, "ALLSD/ALLSE", axes[1])
        _=energy(df, pdf, "ALLSD & ALLSE vs Displacement", axes[2])

        plt.tight_layout()
        figure.savefig(dataf.replace(".json", ".png").replace("U3RF3jsons", "U3RF3plots"), dpi=150)
        plt.close()

    #====plot RF3-GT vs RF3-pred at different U3 for all samples====
    plt.style.use("ggplot")

    figure, axes = plt.subplots(2,2, figsize=(12,10))
    ax = plot_scatter_with_trendline2(axes[0,0], ultimate_force1k["GT"], ultimate_force1k["Pred"])
    ax.set_title("$\epsilon=1000$", fontweight="bold", fontsize=16)
    ax = plot_scatter_with_trendline2(axes[0,1], ultimate_force2k["GT"], ultimate_force2k["Pred"])
    ax.set_title("$\epsilon=2000$", fontweight="bold", fontsize=16)
    ax = plot_scatter_with_trendline2(axes[1,0], ultimate_force3k["GT"], ultimate_force3k["Pred"])
    ax.set_title("$\epsilon=3000$", fontweight="bold", fontsize=16)
    ax = plot_scatter_with_trendline2(axes[1,1], ultimate_force5k["GT"], ultimate_force5k["Pred"])
    ax.set_title("$\epsilon=5000$", fontweight="bold", fontsize=16)
    figure.text(0.5, 0.06, 'Reaction Force (N) - GT Model', ha='center', fontweight="bold", fontsize=16)
    figure.text(0.06, 0.5, 'Reaction Force (N) - Pred Model', va='center', rotation='vertical', fontweight="bold", fontsize=16)
    plt.show()
    figure.savefig(r"C:\Users\wangs\Google Drive\Dissertation\finite element project\ForcePredVsGT.png", dpi=300)
