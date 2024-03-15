import os, glob
import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import r2_score
from scipy.stats import linregress
import matplotlib.transforms as transforms
from functools import reduce
import SimpleITK as sitk
import cv2
from shubow_tools import imsaveseq

def plot_force_displacement(data, preddata, title, axes, ylim=(0, 50)):
    axes.plot(data["U3"].to_numpy(), data["RF3"].to_numpy(), "-b")
    axes.plot(preddata["U3"].to_numpy(), preddata["RF3"].to_numpy(), "--r")
    axes.set_title(title)
    axes.set_ylim(bottom=ylim[0], top=ylim[1])
    axes.set_xlim(left=0, right=5000)
    axes.set_xlabel("Displacement ($\mu\epsilon$)")
    axes.set_ylabel("Reaction Force (N)")
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

def _set_label_tick_color(ax:plt.axes, color:str):
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors=color, labelsize=14)
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors=color, labelsize=14)
    
def plot_scatter_with_trendline(axes, x, y):
    x, y = np.array(x), np.array(y)
    axes.scatter(x,y, c="b")
    x = x[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x, y)

    axes.plot(x, a*x, 'r-')
    text = f"$R^2 = {r2_score(y,x):0.3f}$"
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)
    _set_label_tick_color(axes, "black")

    axes.text(0.8, 0.8, text,
        fontsize=14, verticalalignment='top')
    return axes

def plot_scatter_with_trendline2(axes, x, y):
    axes.scatter(x,y)
    z = np.polyfit(x, y, 1)
    y_hat = np.poly1d(z)(x)
    x_line = np.linspace(0, max(x), 20)
    y_line = np.poly1d(z)(x_line)

    ccc = CCC(x, y)

    axes.plot(x_line, y_line, "r--", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$\nCCC = {ccc:0.3f}"

    _set_label_tick_color(axes, "black")
    axes.set_ylim(bottom=0, top=int(max(x)*1.2))
    axes.set_xlim(left=0, right=int(max(x)*1.2))

    trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)

    axes.text(0.05, 0.95, text, transform=trans,
        fontsize=18, verticalalignment='top')
    return axes


def add_trendline(axis, x, y, style="r--", label=None):
    z = np.polyfit(x, y, 1)

    x_line = np.linspace(0, max(x), 20)
    y_line = np.poly1d(z)(x_line)
    line = axis.plot(x_line, y_line, style, lw=1, label=label)

    print(f"y={z[0]:0.3f}x{z[1]:+0.3f}")
    r2 = r2_score(y, np.poly1d(z)(x))
    print(f"R^2 = {r2}")
    return line, z, r2


def findStepByDisplacement(df, U3, n_index=2):
    u3np = np.abs(df["U3"].to_numpy() - U3)
    index = np.argsort(u3np)[0:n_index]
    subdf = df.loc[list(index)].sort_index()
    
    return subdf["step"].to_numpy(), subdf["U3"].to_numpy()

def calc_compliance(x, y, threshold=1.0):
    
    x = x[y>threshold]
    y = y[y>threshold]
    results = linregress(x , y)
    return results.slope, results.pvalue
    
def plot_scatter_compliances(axes, x, y):
    axes.scatter(x,y)
    z = np.polyfit(x, y, 1)
    y_hat = np.poly1d(z)(x)
    x_line = np.linspace(0, max(x), 20)
    y_line = np.poly1d(z)(x_line)

    ccc = CCC(x, y)

    axes.plot(x_line, y_line, "r--", lw=1)
    axes.plot(x_line, x_line, "b-", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$\nCCC = {ccc:0.3f}"
    axes.set_xlabel("Compliance ($\mu\epsilon/N$) - GT Model", fontweight="bold", fontsize=16)
    axes.set_ylabel("Compliance ($\mu\epsilon/N$) - Pred Model",fontweight="bold", fontsize=16)
    _set_label_tick_color(axes, "black")

    trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)

    axes.text(0.05, 0.95, text, transform=trans,
        fontsize=18, verticalalignment='top')
    return axes


def CCC(y_true, y_pred):
    """
    Lin’s Concordance Correlation Coefficient
    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Lins_Concordance_Correlation_Coefficient.pdf
    https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    ccc = numerator / denominator

    return ccc


def get_lesion_v(base_image_f, lytic_image_f):

    assert os.path.exists(base_image_f), f"{base_image_f} does not exist"
    assert os.path.exists(lytic_image_f), f"{lytic_image_f} does not exist"

    base_image = sitk.GetArrayFromImage(
        sitk.ReadImage(base_image_f))[:-3, 2:-2, 2:-2]
    lytic_image = sitk.GetArrayFromImage(sitk.ReadImage(lytic_image_f))

    base_image = (base_image > 10).astype(np.uint8)
    lytic_image = (lytic_image > 10).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    base_image = process_3D_wrapper(
        base_image, cv2.morphologyEx, cv2.MORPH_OPEN, kernel, iterations=1)
    base_image = process_3D_wrapper(
        base_image, keep_largest_island)
    # lytic_image = process_3D_wrapper(
    #     lytic_image, cv2.morphologyEx, cv2.MORPH_OPEN, kernel, iterations=1)

    # print(f"/n{base_image_f}-{np.sum(base_image)}\n")
    # print(f"{lytic_image_f}-{np.sum(lytic_image)}\n")

    lesions = (base_image - lytic_image[:-3]) * base_image

    lesion_percent = np.sum(lesions) / np.sum(base_image) * 100
    return lesion_percent


def draw_colors_from_colormap(n, colormap_name):
    colormap = plt.get_cmap(colormap_name)
    colors = colormap(np.linspace(0, 1, n))
    return colors


def keep_largest_island(image):
    """
    Keep the largest island in a binary image
    """
    num_labels, labels_im = cv2.connectedComponents(image.astype(np.uint8))
    if num_labels > 1:
        label_count = np.bincount(labels_im.flatten())
        label_count[0] = 0
        max_label = np.argmax(label_count)
        image = (labels_im == max_label).astype(np.uint8)
    return image


def process_3D_wrapper(image, process_3D, *args, **kwargs):
    """ 
    Wrapper for processing 3D image
    """
    return np.array([process_3D(image[i], *args, **kwargs) for i in range(image.shape[0])])


if __name__=="__main__":
    fd = r"E:\35_um_data_100x100x48 niis\abaqus_results\U3RF3jsons"
    DATA_ROOT = r"E:\35_um_data_100x100x48 niis\Data"
    DATA_ROOT_FE = r"E:\35_um_data_100x100x48 niis\abaqus_results\Nfiti"
    data_list = glob.glob(os.path.join(fd, "[0-9]*.json"))

    ultimate_force1k = {"GT":[], "Pred":[]}
    ultimate_force2k = {"GT":[], "Pred":[]}
    ultimate_force3k = {"GT":[], "Pred":[]}
    ultimate_force5k = {"GT":[], "Pred":[]}
    compliances = []
    compliances_lesion_v = {"compliance_gt": [],
                            "compliance_pred": [], "ids": []}

    tare_strain = 0.012
    # newx for interpolation
    xleft = int(((0-tare_strain)/1.44*1E6//100+1)*100)
    xright = 11000
    newx = np.linspace(xleft, xright, (xright - xleft)//100+1)

    n = 0
    for i, dataf in enumerate(data_list):
        sample_name = os.path.basename(dataf)
        sample_id = sample_name.replace("_3um.json", "")

        preddataf = os.path.join(os.path.dirname(dataf),"pred"+sample_name)
        data = json.load(open(dataf, "r"))
        preddata = json.load(open(preddataf, "r"))
        
        #====transform data====
        df = data_transform(pd.DataFrame(data), tare_strain=tare_strain)
        pdf = data_transform(pd.DataFrame(preddata), tare_strain=tare_strain)

        print("--".join([sample_name,str(df.loc[df.shape[0]-1, "U3"]), str(pdf.loc[pdf.shape[0]-1, "U3"])]))
        print("--".join([sample_name,str(df.loc[0, "U3"]), str(df.loc[0, "U3"])]))
        #====interpolate U3 and RF3====
        dfinterp = interpolate_force(df, newx)
        pdfinterp = interpolate_force(pdf, newx)

        #====extract RF3-GT vs RF3-pred at different U3 for each sample====
        a = dfinterp.loc[5000, "RF3"]/pdfinterp.loc[5000, "RF3"]
        if a<0.5 or a > 2.0: # remove some outliers
            print(dataf+":"+str(a))
        else:
            n += 1
            ultimate_force1k["GT"].append(dfinterp.loc[1000, "RF3"] )
            ultimate_force1k["Pred"].append(pdfinterp.loc[1000, "RF3"] )

            ultimate_force2k["GT"].append(dfinterp.loc[2000, "RF3"] )
            ultimate_force2k["Pred"].append(pdfinterp.loc[2000, "RF3"] )
            
            ultimate_force3k["GT"].append(dfinterp.loc[3000, "RF3"] )
            ultimate_force3k["Pred"].append(pdfinterp.loc[3000, "RF3"] )
            
            ultimate_force5k["GT"].append(dfinterp.loc[5000, "RF3"] )
            ultimate_force5k["Pred"].append(pdfinterp.loc[5000, "RF3"] )
            
            compliance, p = calc_compliance(df["RF3"],df["U3"], 2)
            compliance_pred, pp = calc_compliance(pdf["RF3"],pdf["U3"], 2)
            print(f"GT_compliance={compliance}, p={p}\tPred_compliance={compliance_pred}, p={pp}")
            compliances.append((compliance, compliance_pred))

            gt_nii = os.path.join(
                DATA_ROOT_FE, sample_id+".nii.gz")
            pred_nii = os.path.join(
                DATA_ROOT_FE, "pred"+sample_id+".nii.gz")

            t0 = 0
            t_zero_nii = ""
            while t0 < 3:
                t_zero_nii = os.path.join(
                    DATA_ROOT, sample_id[:-1]+f"{t0}.nii.gz")
                if os.path.exists(t_zero_nii):
                    compliances_lesion_v["ids"].append(
                        (sample_id, sample_id[:-1]+f"{t0}"))
                    break
                t0 += 1

            t0_img = sitk.ReadImage(t_zero_nii)
            t0_out_dir = os.path.join(
                r"E:\35_um_data_100x100x48 niis\abaqus_results\2dseq_t0", sample_id[
                    :-1]+f"{t0}"
            )
            os.makedirs(t0_out_dir, exist_ok=True)
            imsaveseq(t0_img, t0_out_dir, sample_id[:-1]+f"{t0}")

            compliances_lesion_v["compliance_gt"].append(
                (compliance, get_lesion_v(t_zero_nii, gt_nii)))
            compliances_lesion_v["compliance_pred"].append(
                (compliance_pred, get_lesion_v(t_zero_nii, pred_nii)))

        #====plot RF3, energy vs U3 for each pair of samples====
        # figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        # _=plot_force_displacement(df, pdf, "Force vs Displacement", axes[0], ylim=(-1, 1.2*max([dfinterp.loc[5000, "RF3"], pdfinterp.loc[5000, "RF3"]])))
        # _=energy_fraction(df, pdf, "ALLSD/ALLSE", axes[1])
        # _=energy(df, pdf, "ALLSD & ALLSE vs Displacement", axes[2])

        # plt.tight_layout()
        # # figure.savefig(dataf.replace(".json", ".png").replace("U3RF3jsons", "U3RF3plots"), dpi=150)
        # plt.show()
        # plt.close()
        


    #====plot RF3-GT vs RF3-pred at different U3 for all samples====
    plt.style.use("ggplot")

    figure, axes = plt.subplots(2,2, figsize=(12,11))
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
    figure.savefig(r"C:\Users\wangs\My Drive\Dissertation\finite element project\ForcePredVsGT.png", dpi=300)
    print(f"Number of samples {n}")
    
    #====plot compliances====
    gtcompliances, predcompliances = zip(*compliances)
    mape = np.array(list(map(lambda x: abs(x[0]-x[1])/x[0]*100, compliances)))
    mapestd = mape.std()
    print(f"Mean percentage error: {mape.mean():4.2f}% , std={mapestd:4.2f}")
    figure, ax = plt.subplots(1,1, figsize=(6,5))
    ax = plot_scatter_compliances(ax, gtcompliances, predcompliances)
    # ax.set_title("$\epsilon=1000$", fontweight="bold", fontsize=16)
    plt.tight_layout()
    plt.show()
    figure.savefig(r"C:\Users\wangs\My Drive\Dissertation\finite element project\CompliancePredVsGT.png", dpi=300)

    # ====plot compliances vs lesion volume====

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerTuple

    figure, ax = plt.subplots(1, 1, figsize=(6, 5))

    n_points = len(compliances_lesion_v["compliance_gt"])
    colors = draw_colors_from_colormap(n_points, "gist_rainbow")

    for i in range(n_points):
        ax.scatter(compliances_lesion_v["compliance_gt"][i][1],
                   compliances_lesion_v["compliance_gt"][i][0], color=colors[i], marker="o")
        ax.scatter(compliances_lesion_v["compliance_pred"][i][1],
                   compliances_lesion_v["compliance_pred"][i][0], color=colors[i], marker="x")
        ax.plot([compliances_lesion_v["compliance_gt"][i][1], compliances_lesion_v["compliance_pred"][i][1]],
                [compliances_lesion_v["compliance_gt"][i][0], compliances_lesion_v["compliance_pred"][i][0]], color="k", linestyle="-", lw=1)

    gt_y, gt_x = zip(*compliances_lesion_v["compliance_gt"])
    pred_y, pred_x = zip(*compliances_lesion_v["compliance_pred"])
    
    plt.xlim(left=0, right=100)
    plt.ylim(bottom=0, top=1000)
    plt.xlabel("Lesion Volume (%)", fontweight="bold", fontsize=16)
    plt.ylabel("Compliance ($\mu\epsilon/N$)", fontweight="bold", fontsize=16)
    _set_label_tick_color(ax, "black")
    line_gt, _, r2_gt = add_trendline(ax, gt_x, gt_y, "b-", "GT")
    line_pred, _, r2_pred = add_trendline(ax, pred_x, pred_y, "r--", "Pred")
    
    legend_handle_gt = mlines.Line2D([], [], color='b', marker='o', linestyle='-',
                                     markersize=6, label=f"GT: $R^2$ = {r2_gt:3.2f}")
    legend_handle_pred = mlines.Line2D([], [], color='r', marker='x', linestyle='--',
                                       markersize=6, label=f"Pred: $R^2$ = {r2_pred:3.2f}")
    
    plt.legend(handles=[legend_handle_gt,
               legend_handle_pred], loc="upper left")
    plt.tight_layout()
    plt.show()
    figure.savefig(
        r"C:\Users\wangs\My Drive\Dissertation\finite element project\ComplianceVsLesionVolume.png", dpi=300)
