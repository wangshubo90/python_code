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

LABELFONT = {'fontname': 'Times New Roman'}
AXISLABELSIZE = 18
plt.rcParams['font.family'] = 'Times New Roman'

def plot_force_displacement(data, preddata, title, axes, ylim=(0, 50)):
    axes.plot(data["U3"].to_numpy(), data["RF3"].to_numpy(), "-b")
    axes.scatter(data["U3"].to_numpy(),
                 data["RF3"].to_numpy(), color="b", marker="x", s=15.0)
    axes.plot(preddata["U3"].to_numpy(), preddata["RF3"].to_numpy(), "--r")
    axes.scatter(preddata["U3"].to_numpy(),
                 preddata["RF3"].to_numpy(), color="r", marker="v", s=15.0)
    axes.set_title(title, **LABELFONT)
    axes.set_ylim(bottom=ylim[0], top=ylim[1])
    axes.set_xlim(left=0, right=5000)
    axes.set_xlabel("Strain ($\mu\epsilon$)", **LABELFONT)
    axes.set_ylabel("Reaction Force (N)", **LABELFONT)
    axes.relim()
    axes.autoscale_view()
    return axes

def energy_fraction(data, preddata, title, axes):
    axes.plot(data["U3"].to_numpy(), data["ALLSD"].to_numpy() / (data["ALLSE"].to_numpy()+1e-8), "-b")
    axes.plot(preddata["U3"].to_numpy(), preddata["ALLSD"].to_numpy() / (preddata["ALLSE"].to_numpy()+1e-8), "--r")
    axes.set_title(title, **LABELFONT)
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
    axes.set_title(title, **LABELFONT)
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
    x = np.asarray(x)
    y = np.asarray(y)
    axes.scatter(x, y)

    # calculate CCC
    ccc = CCC(x, y)

    # fit y = ax, b=0
    slope = np.dot(x, y) / np.dot(x, x)
    y_hat = slope*x
    x_line = np.linspace(0, max(x), 20)
    y_line = slope*x_line
    text = f"y = {slope:0.3f} x\n$R^2$ = {r2_score(y,y_hat):0.3f}\nCCC = {ccc:0.3f}"

    # # fit y = ax+b
    # z = np.polyfit(x, y, 1)
    # y_hat = np.poly1d(z)(x)
    # x_line = np.linspace(0, max(x), 20)
    # y_line = np.poly1d(z)(x_line)
    # text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$\nCCC = {ccc:0.3f}"

    axes.plot(x_line, y_line, "r--", lw=1)

    _set_label_tick_color(axes, "black")
    axes.set_ylim(bottom=0, top=int(max(x)*1.2))
    axes.set_xlim(left=0, right=int(max(x)*1.2))
    axes.tick_params(axis='x', labelsize=AXISLABELSIZE)
    axes.tick_params(axis='y', labelsize=AXISLABELSIZE)

    trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)
    x_pos = (axes.get_xlim()[1] - axes.get_xlim()[0]) * 0.05 + axes.get_xlim()[0]
    # y_pos = (axes.get_ylim()[1] - axes.get_ylim()[0]) * 0.9 + axes.get_ylim()[0]
    axes.text(x_pos, 0.8, text, transform=trans,
              fontsize=14, horizontalalignment = "left", **LABELFONT)
    return axes


def add_trendline(axis, x, y, style="r--", label=None):
    x = np.asarray(x)
    y = np.asarray(y)
    slope = np.dot(x, y) / np.dot(x, x)
    x_line = np.linspace(0, max(x), 20)
    # z = np.polyfit(x, y, 1)
    # y_line = np.poly1d(z)(x_line)
    y_line = slope*x_line
    line = axis.plot(x_line, y_line, style, lw=1, label=label)

    # print(f"y={z[0]:0.3f}x{z[1]:+0.3f}")
    print(f"y={slope:0.3f}x")
    # r2 = r2_score(y, np.poly1d(z)(x))
    r2 = r2_score(y, slope*x)
    print(f"R^2 = {r2}")
    # return line, z, r2
    return line, slope, r2


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
    x = np.asarray(x)
    y = np.asarray(y)
    axes.scatter(x,y)
    # calculate CCC
    ccc = CCC(x, y)

    # fit y = ax, b=0
    slope = np.dot(x, y) / np.dot(x, x)
    y_hat = slope*x
    x_line = np.linspace(0, max(x), 20)
    y_line = slope*x_line
    text = f"$y=${slope:0.3f}$\;x$\n$R^2 = ${r2_score(y,y_hat):0.3f}\nCCC = {ccc:0.3f}"

    # fit y = ax+b
    # z = np.polyfit(x, y, 1)
    # y_hat = np.poly1d(z)(x)
    # x_line = np.linspace(0, max(x), 20)
    # y_line = np.poly1d(z)(x_line)
    # text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$\nCCC = {ccc:0.3f}"

    axes.plot(x_line, y_line, "r--", lw=1)
    axes.plot(x_line, x_line, "b-", lw=1)
    axes.set_xlabel("Compliance ($\mu\epsilon/N$) - GT Model", fontsize=AXISLABELSIZE, **LABELFONT)
    axes.set_ylabel("Compliance ($\mu\epsilon/N$) - Pred Model", fontsize=AXISLABELSIZE, **LABELFONT)
    _set_label_tick_color(axes, "black")

    trans = transforms.blended_transform_factory(axes.transData, axes.transAxes)

    axes.text(0.05, 0.95, text, transform=trans,
              fontsize=14, verticalalignment='top', **LABELFONT)
    axes.tick_params(axis='x', labelsize=18)
    axes.tick_params(axis='y', labelsize=18)
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
    if lytic_image.shape[1] != 96:
        lytic_image = lytic_image[:, 2:-2, 2:-2]
    lytic_image = lytic_image[:-3, :, :]

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

    lesions = (base_image - lytic_image) * base_image

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


def set_axes_style(ax: plt.axis):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)

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
    compliances_lesion_v = {
        "compliance_gt": [],
        "compliance_pred": [],
        "ids": []
    }

    compliances_lesion_v_n1 = {
        "lesions": [],
        "ids": []
    }

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
            tm1_id = sample_id[:-1]+str(int(sample_id[-1])-1)
            tm1_f = os.path.join(
                DATA_ROOT, tm1_id + ".nii.gz")

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

            if tm1_f != t_zero_nii and os.path.exists(tm1_f):
                tm1_img = sitk.ReadImage(tm1_f)
                tm1_out_dir = os.path.join(
                    r"E:\35_um_data_100x100x48 niis\abaqus_results\2dseq_t0", tm1_id
                )
                os.makedirs(tm1_out_dir, exist_ok=True)
                imsaveseq(tm1_img, tm1_out_dir,
                          sample_id[:-1]+f"{int(sample_id[-1])-1}")

                compliances_lesion_v_n1["ids"].append(
                    (sample_id, tm1_id))
                compliances_lesion_v_n1["lesions"].append(
                    (get_lesion_v(t_zero_nii, gt_nii), get_lesion_v(t_zero_nii, pred_nii), get_lesion_v(t_zero_nii, tm1_f)))
            else:
                print(f"-1:{tm1_f} vs 0:{t_zero_nii}")

            os.makedirs(t0_out_dir, exist_ok=True)
            imsaveseq(t0_img, t0_out_dir, sample_id[:-1]+f"{t0}")

            compliances_lesion_v["compliance_gt"].append(
                (compliance, get_lesion_v(t_zero_nii, gt_nii)))
            compliances_lesion_v["compliance_pred"].append(
                (compliance_pred, get_lesion_v(t_zero_nii, pred_nii)))

        # ====plot RF3, energy vs U3 for each pair of samples====
        pd.DataFrame({"x":df["U3"].to_numpy(), "y":df["RF3"].to_numpy()}).to_csv(dataf.replace(".json", "_GT.csv"))
        pd.DataFrame({"x":pdf["U3"].to_numpy(), "y":pdf["RF3"].to_numpy()}).to_csv(dataf.replace(".json", "_PRED.csv"))
        # figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        # _ = plot_force_displacement(df, pdf, "Reaction Force vs Strain", axes[0], ylim=(
        #     -1, 1.2*max([dfinterp.loc[5000, "RF3"], pdfinterp.loc[5000, "RF3"]])))
        # _ = energy_fraction(df, pdf, "ALLSD/ALLSE", axes[1])
        # _ = energy(df, pdf, "ALLSD & ALLSE vs Displacement", axes[2])
        # for i in range(3):
        #     set_axes_style(axes[i])
        # plt.tight_layout()
        # figure.savefig(dataf.replace(".json", ".png").replace("U3RF3jsons", "U3RF3plots"), dpi=1000)
        # # plt.show()
        # plt.close()
        


    #====plot RF3-GT vs RF3-pred at different U3 for all samples====
    plt.style.use("seaborn-colorblind")

    figure, axes = plt.subplots(2,2, figsize=(12,11))
    ax = plot_scatter_with_trendline2(axes[0,0], ultimate_force1k["GT"], ultimate_force1k["Pred"])
    ax.set_title("$@1000\mu\epsilon$", fontsize=16, **LABELFONT)
    set_axes_style(ax)
    ax = plot_scatter_with_trendline2(axes[0,1], ultimate_force2k["GT"], ultimate_force2k["Pred"])
    ax.set_title("$@2000\mu\epsilon$", fontsize=16, **LABELFONT)
    set_axes_style(ax)
    ax = plot_scatter_with_trendline2(axes[1,0], ultimate_force3k["GT"], ultimate_force3k["Pred"])
    ax.set_title("$@3000\mu\epsilon$", fontsize=16, **LABELFONT)
    set_axes_style(ax)
    ax = plot_scatter_with_trendline2(axes[1,1], ultimate_force5k["GT"], ultimate_force5k["Pred"])
    ax.set_title("$@5000\mu\epsilon$", fontsize=16, **LABELFONT)
    set_axes_style(ax)
    figure.text(0.5, 0.06, 'Reaction Force (N) - GT Model',
                ha='center', fontsize=AXISLABELSIZE, **LABELFONT)
    figure.text(0.06, 0.5, 'Reaction Force (N) - Pred Model', va='center',
                rotation='vertical', fontsize=AXISLABELSIZE, **LABELFONT)
    plt.show()
    figure.savefig(
        r"C:\Users\wangs\My Drive\Dissertation\finite element project\ForcePredVsGT.jpg", dpi=1000)
    print(f"Number of samples {n}")
    
    #====plot compliances====
    gtcompliances, predcompliances = zip(*compliances)
    mape = np.array(list(map(lambda x: abs(x[0]-x[1])/x[0]*100, compliances)))
    mapestd = mape.std()
    print(f"Mean percentage error: {mape.mean():4.2f}% , std={mapestd:4.2f}")
    figure, ax = plt.subplots(1,1, figsize=(6,5))
    ax = plot_scatter_compliances(ax, gtcompliances, predcompliances)
    # ax.set_title("$\epsilon=1000$", fontsize=16)
    plt.tight_layout()
    plt.show()
    figure.savefig(
        r"C:\Users\wangs\My Drive\Dissertation\finite element project\CompliancePredVsGT.jpg", dpi=1000)

    # ====plot compliances vs lesion volume====

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerTuple
    import matplotlib.font_manager as font_manager

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
    plt.xlabel("Lesion Volume (%)",
               fontsize=AXISLABELSIZE, **LABELFONT)
    plt.ylabel("Compliance ($\mu\epsilon/N$)",
             fontsize=AXISLABELSIZE, **LABELFONT)
    _set_label_tick_color(ax, "black")
    line_gt, slope_gt, r2_gt = add_trendline(ax, gt_x, gt_y, "b-", "GT")
    line_pred, slope_pred, r2_pred = add_trendline(
        ax, pred_x, pred_y, "r--", "Pred")
    
    legend_handle_gt = mlines.Line2D([], [], color='b', marker='o', linestyle='-',
                                     markersize=6, label=f"GT: $R^2$ = {r2_gt:3.2f}, $y={slope_gt:3.2f} x$")
    legend_handle_pred = mlines.Line2D([], [], color='r', marker='x', linestyle='--',
                                       markersize=6, label=f"Pred: $R^2$ = {r2_pred:3.2f}, $y={slope_pred:3.2f} x$")
    
    font_props = font_manager.FontProperties(
        size=12, weight='bold', family='Times New Roman')
    plt.legend(handles=[legend_handle_gt,
               legend_handle_pred], loc="upper left", prop=font_props)
    plt.tight_layout()
    plt.show()
    figure.savefig(
        r"C:\Users\wangs\My Drive\Dissertation\finite element project\ComplianceVsLesionVolume.jpg", dpi=1000)

    print(compliances_lesion_v_n1)
    print(compliances_lesion_v)

    # ====plot lesion volume tn-1 vs lesion volume tn====
    fig = plt.figure(figsize=(14, 12.5), facecolor='none')
    fig.patch.set_alpha(0.0)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    ax = plt.subplot2grid((2, 4), (0, 2), colspan=2)
    ax2 = plt.subplot2grid((2, 4), (1, 1), colspan=2)
    ax1.tick_params(axis='x', labelsize=AXISLABELSIZE)
    ax1.tick_params(axis='y', labelsize=AXISLABELSIZE)
    ax2.tick_params(axis='x', labelsize=AXISLABELSIZE)
    ax2.tick_params(axis='y', labelsize=AXISLABELSIZE)
    set_axes_style(ax)
    set_axes_style(ax1)
    set_axes_style(ax2)

    # ==============

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

    ax.set_xlim(left=0, right=80)
    ax.set_ylim(bottom=0, top=1000)
    ax.set_xlabel("Lesion Volume (%)",
                  fontsize=AXISLABELSIZE, **LABELFONT)
    ax.set_ylabel("Compliance ($\mu\epsilon/N$)",
                 fontsize=AXISLABELSIZE, **LABELFONT)

    
    _set_label_tick_color(ax, "black")
    line_gt, slope_gt, r2_gt = add_trendline(ax, gt_x, gt_y, "b-", "GT")
    line_pred, slope_pred, r2_pred = add_trendline(
        ax, pred_x, pred_y, "r--", "Pred")

    legend_handle_gt = mlines.Line2D([], [], color='b', marker='o', linestyle='-',
                                     markersize=6, label="")
    legend_handle_pred = mlines.Line2D([], [], color='r', marker='x', linestyle='--',
                                       markersize=6, label="")

    font_props = font_manager.FontProperties(
        size=12, weight='bold', family='Times New Roman')
    ax.legend(handles=[legend_handle_gt,
                       legend_handle_pred], loc="upper left", prop=font_props)
    
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    x_pos = 0.15
    text1 = f"GT: $R^2$ = {r2_gt:3.2f}, y = {slope_gt:3.2f} x"
    text2 = f"Pred: $R^2$ = {r2_pred:3.2f}, y = {slope_pred:3.2f} x"
    ax.text(x_pos, 0.94, text1, transform=trans,
              fontsize=14, horizontalalignment = "left", **LABELFONT)
    ax.text(x_pos, 0.89, text2, transform=trans,
              fontsize=14, horizontalalignment = "left", **LABELFONT)
    
    ax.tick_params(axis='x', labelsize=AXISLABELSIZE)
    ax.tick_params(axis='y', labelsize=AXISLABELSIZE)
    plt.yticks(fontsize=AXISLABELSIZE)
    plt.xticks(fontsize=AXISLABELSIZE)

    # ==============

    plot_scatter_compliances(ax1, gtcompliances, predcompliances)
    ax1.set_xlim(right=1100)
    ax1.set_ylim(top=1100)
    ax1.tick_params(axis='x', labelsize=AXISLABELSIZE)
    ax1.tick_params(axis='y', labelsize=AXISLABELSIZE)
    plt.yticks(fontsize=AXISLABELSIZE)
    plt.xticks(fontsize=AXISLABELSIZE)

    # ==============

    lesion_gt, lesion_pred, lesion_tm1 = zip(
        *compliances_lesion_v_n1["lesions"])
    lesion_gt = np.array(lesion_gt)
    lesion_pred = np.array(lesion_pred)
    lesion_tm1 = np.array(lesion_tm1)

    n = len(lesion_gt)
    colors = draw_colors_from_colormap(n, "gist_rainbow")

    # for i in range(n):
    #     ax.plot(["$t_n$", "$t_{n+1}$"], [lesion_tm1[i],
    #             lesion_pred[i]], color=colors[i], linestyle="-", lw=1)
    #     ax.plot(["$t_n$", "$t_{n+1}$"], [lesion_tm1[i],
    #             lesion_gt[i]], color=colors[i], linestyle="--", lw=1)

    import seaborn as sns
    dataframe = {"$t_n$": lesion_tm1,
                 "$t_{n+1}$ GT": lesion_gt, "$t_{n+1}$ Pred": lesion_pred}
    dataframe = pd.DataFrame(dataframe)
    dataframe.to_csv(r"C:\Users\wangs\My Drive\Dissertation\finite element project\lesions.csv")
    dataframe = dataframe.melt(
        var_name="group", value_name="Lesion Volume (%)")
    sns.boxplot(data=dataframe, ax=ax2, x="group", y="Lesion Volume (%)", palette=sns.color_palette("Paired", 9)[2::-1])
    sns.stripplot(data=dataframe, ax=ax2, x="group",
                  y="Lesion Volume (%)", color="black", size=4)
    # ax.set_xlabel("")
    ax2.set_ylim(bottom=0, top=80)
    ax2.set_xlabel("",
                   fontsize=AXISLABELSIZE, **LABELFONT)
    ax2.set_ylabel("Lesion Volume (%)", fontsize=AXISLABELSIZE, **LABELFONT)
    _set_label_tick_color(ax2, "black")
    ax2.tick_params(axis='x', labelsize=AXISLABELSIZE)
    ax2.tick_params(axis='y', labelsize=AXISLABELSIZE)
    plt.yticks(fontsize=AXISLABELSIZE)
    plt.xticks(fontsize=AXISLABELSIZE)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95,
                        bottom=0.05, hspace=0.3, wspace=0.6)
    # fig.text(0.03, 0.95, 'A',
    #          ha='center', fontsize=20, **LABELFONT)
    # fig.text(0.5, 0.95, 'B',
    #          ha='center', fontsize=20, **LABELFONT)
    # fig.text(0.27, 0.45, 'C',
    #          ha='center', fontsize=20, **LABELFONT)
    plt.show()
    fig.savefig(
        r"C:\Users\wangs\My Drive\Dissertation\finite element project\Figure4Compiled.jpg", dpi=1000)


    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Fit ANOVA model
    dataframe.columns = ["group", "values"]
    model = ols('values ~ group', data=dataframe).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print ANOVA results
    print("ANOVA results:")
    print(anova_table)

    # Perform Tukey's HSD post hoc test
    posthoc = pairwise_tukeyhsd(dataframe['values'], dataframe['group'])

    # Print post hoc test results
    print("\nPost hoc test results:")
    print(posthoc)
    print("")