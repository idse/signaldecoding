#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SI Figure: VIB 2D Accuracy - Continuous Gene Expression Analysis
Reads data from VIB 2D analysis and generates accuracy plots for B50 condition

Based on SI_fig_accuracy_continuous.py
"""

import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

import fns_plotting_scripts as fns_plot


# ============= HELPER FUNCTION (from fns_plotting_scripts) =============
def calc_prediction_hist(tar_test, target_predict, N_bins, g_min, g_max, min_count=5):
    """
    Calculate conditional histogram of predicted vs observed values
    
    Parameters:
    -----------
    tar_test : np.ndarray
        Observed/test target values
    target_predict : np.ndarray
        Predicted target values
    N_bins : int
        Number of bins
    g_min : float
        Minimum value for binning
    g_max : float
        Maximum value for binning
    min_count : int
        Minimum count per bin to compute statistics
        
    Returns:
    --------
    bins : np.ndarray
        Bin centers
    hist : np.ndarray
        2D histogram (normalized)
    hist_cond : np.ndarray
        Conditional histogram (normalized per column)
    hist_av : np.ndarray
        Mean observed value per predicted bin
    hist_var : np.ndarray
        Variance of observed value per predicted bin
    """
    edges = np.linspace(g_min, g_max, N_bins + 1)
    bins = (edges[1:] + edges[:-1]) / 2
    
    # x is predicted, y is observed
    hist, _, _ = np.histogram2d(target_predict, tar_test, edges)
    hist = hist / np.sum(hist)
    
    hist_cond = np.zeros(hist.shape)
    for b in range(N_bins):
        sumcounts = np.sum(hist[b, :])  # normalize each column
        if sumcounts > 0:
            hist_cond[b, :] = hist[b, :] / sumcounts
            
    hist_av = np.zeros(N_bins)
    hist_var = np.zeros(N_bins)
    hist_sq_sum = np.zeros(N_bins)
    count = np.zeros(N_bins)
    N_entries = len(tar_test)
    
    for i in range(N_entries):
        for b in range(N_bins):
            # condition on predicted data, average observed (test) data
            # this avoids 'errors in variables' issues
            if target_predict[i] >= edges[b] and target_predict[i] < edges[b + 1]:
                hist_av[b] += tar_test[i]
                hist_sq_sum[b] += tar_test[i] ** 2
                count[b] += 1
                
    for b in range(N_bins):
        if count[b] > min_count:
            hist_av[b] /= count[b]
            hist_var[b] = hist_sq_sum[b] / count[b] - hist_av[b] ** 2
        else:
            hist_av[b] = np.nan
            hist_var[b] = np.nan
    
    return bins, hist, hist_cond, hist_av, hist_var


# ============= CONFIGURATION =============
directory = 'data_expt_20_scaled_norm_bgsub'
cond = 'B50'

# VIB analysis settings
SPLIT_TYPE = 'random'  # Must match what was used in analysis script
subdirectory_VIB = directory + '/analysis_VIB_2D_split_' + SPLIT_TYPE

# Output directory for figures
subdirectory_plot_fig = directory + '/SI_fig_accuracy_continuous_VIB2D'
if not os.path.exists(subdirectory_plot_fig):
    os.makedirs(subdirectory_plot_fig)

print("="*60)
print("VIB 2D ACCURACY ANALYSIS - CONTINUOUS GENE EXPRESSION")
print("="*60)
print(f"Condition: {cond}")
print(f"VIB data directory: {subdirectory_VIB}")
print(f"Output directory: {subdirectory_plot_fig}")

# ============= LOAD VIB DATA =============
print("\nLoading VIB 2D data...")

vib_data_path = os.path.join(subdirectory_VIB, f'latent_2D_data_{cond}.pickle')
with open(vib_data_path, 'rb') as f:
    vib_data = pickle.load(f)

# Extract relevant data
Y_full = vib_data['Y_full']           # Observed gene expression (test targets)
Y_pred = vib_data['Y_pred']           # Predicted gene expression
metricdist_full = vib_data['metricdist_full']  # Radial position
markers_clean = vib_data['markers_clean']       # Cell fates
OUTPUT_NAMES = vib_data['OUTPUT_NAMES']         # Gene names
latent_2D = vib_data['latent_2D']               # 2D latent space

N_genes = Y_full.shape[1]
N_samples = Y_full.shape[0]

print(f"  Samples: {N_samples}")
print(f"  Genes: {N_genes}")
print(f"  Latent shape: {latent_2D.shape}")

# For compatibility with plotting functions, use Y_full as tar_test and Y_pred as target_predict
tar_test = Y_full
target_predict = Y_pred

# Load original data for r_max
subdirectory_data = directory + '/data'
with open(subdirectory_data + '/data_' + cond + ".pickle", 'rb') as f:
    data = pickle.load(f)
r_max = data.r_max

# ============= PLOTTING SETUP =============
file_suffix = '.png'
file_suffix_pdf = '.pdf'
plt.close('all')

fs = 11
fs2 = 9
params = {
    'font.size': fs,
    'xtick.labelsize': fs2,
    'ytick.labelsize': fs2,
}
plt.rcParams.update(params)

thresh = 1
min_val = 0
max_val = 6

# ============= FIGURE 1: GENE-BY-GENE OVERVIEW =============
print("\nGenerating gene-by-gene overview figure...")

margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 1 - 0.2
hspace = 0.001
wspace = 0.001

lim = 360
ms = 0.15

H, W = 9, N_genes
fig_size = [13, 8]
fs2 = 6
params = {
    'figure.figsize': fig_size,
    'xtick.labelsize': fs2,
    'ytick.labelsize': fs2,
}
plt.rcParams.update(params)

ticks_on = 0

fig = plt.figure()
chrt = 0
fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1 - margin_right, 
                    top=margin_top, wspace=wspace, hspace=hspace)

# Row 1: Density scatter plots
corrs = np.zeros((N_genes, 2))
for kg in range(0, N_genes):
    chrt += 1
    plt.subplot(H, W, chrt)
    plt.title(OUTPUT_NAMES[kg], fontsize=10)
    
    N_max = target_predict.shape[0]
    N_plot = min(4000, N_max)
    indices = np.random.randint(0, N_max, N_plot)
    fns_plot.scatter_density(target_predict[indices, kg], tar_test[indices, kg])
    
    corrs[kg, 0], _ = fns_plot.calc_pearsoncorr(target_predict[:, kg], tar_test[:, kg])
    corrs[kg, 1], _ = fns_plot.calc_spearmancorr(target_predict[:, kg], tar_test[:, kg])
    
    max_val_kg = np.nanpercentile(tar_test[indices, kg], 99.9)
    
    plt.plot([0, max_val_kg], [0, max_val_kg], '-', color='grey', lw=0.5)
    
    plt.xlim([min_val, max_val_kg])
    plt.ylim([min_val, max_val_kg])
    
    if ticks_on:
        plt.xticks([min_val, max_val_kg])
        plt.yticks([min_val, max_val_kg])
    else:
        plt.xticks([])
        plt.yticks([])

# Row 2: Conditional histograms
N_bins = 30
for kg in range(0, N_genes):
    chrt += 1
    plt.subplot(H, W, chrt)
    
    max_val_kg = np.nanpercentile(tar_test[:, kg], 99.9)
    
    bins, hist, hist_cond, hist_av, hist_var = calc_prediction_hist(
        tar_test[:, kg], target_predict[:, kg], N_bins, min_val, max_val_kg
    )
    
    plt.imshow(np.rot90(hist_cond, k=1), extent=[min_val, max_val_kg, min_val, max_val_kg],
               cmap='YlGnBu', vmin=0, vmax=0.4)
    
    plt.plot(bins, hist_av, '-k', lw=1.5)
    plt.plot(bins, hist_av - np.sqrt(hist_var), ':k', lw=1)
    plt.plot(bins, hist_av + np.sqrt(hist_var), ':k', lw=1)
    
    plt.plot([0, max_val_kg], [0, max_val_kg], '-', color='grey', lw=0.5)
    
    plt.xlim([min_val, max_val_kg])
    plt.ylim([min_val, max_val_kg])
    
    if ticks_on:
        plt.xticks([min_val, max_val_kg])
        plt.yticks([min_val, max_val_kg])
    else:
        plt.xticks([])
        plt.yticks([])

# Row 3: Radial profile comparison
# For this, we need to pick a representative colony. Use all data reshaped.
# Since data is already flattened, we'll use the full dataset
N_bins_x = 20
radial_av_all = np.zeros((N_genes, N_bins_x, 2))

for kg in range(0, N_genes):
    chrt += 1
    plt.subplot(H, W, chrt)
    
    # Reshape for calc_profile_meanvar: expects (N_sys, N_cells, N_var)
    bins_x, mean_x_expt, var_x, P_x = fns_plot.calc_profile_meanvar(
        tar_test[:, kg][np.newaxis, :, np.newaxis],
        metricdist_full[np.newaxis, :, np.newaxis],
        N_bins_x, r_max
    )
    radial_av_all[kg, :, 0] = mean_x_expt[:, 0]
    
    bins_x, mean_x_sim, var_x, P_x = fns_plot.calc_profile_meanvar(
        target_predict[:, kg][np.newaxis, :, np.newaxis],
        metricdist_full[np.newaxis, :, np.newaxis],
        N_bins_x, r_max
    )
    radial_av_all[kg, :, 1] = mean_x_sim[:, 0]
    
    max_val_rad = 1.1 * np.nanmax(mean_x_expt)
    
    plt.plot([0, max_val_rad], [0, max_val_rad], '-', color='grey', lw=0.5)
    plt.scatter(mean_x_sim, mean_x_expt, color='k', s=10)
    
    plt.xlim([min_val, max_val_rad])
    plt.ylim([min_val, max_val_rad])
    
    if ticks_on:
        plt.xticks([min_val, max_val_rad])
        plt.yticks([min_val, max_val_rad])
    else:
        plt.xticks([])
        plt.yticks([])

plt.tight_layout()
plt.savefig(subdirectory_plot_fig + "/" + 'SI_fig_genes_cont_overview_VIB2D_' + cond + file_suffix, dpi=150)
print(f"  Saved: SI_fig_genes_cont_overview_VIB2D_{cond}{file_suffix}")


# ============= FIGURE 2: SUMMARY PLOTS =============
print("\nGenerating summary figure...")

fs2 = 11
fig_size = [9, 3]
params = {
    'figure.figsize': fig_size,
    'font.size': fs,
    'xtick.labelsize': fs2,
    'ytick.labelsize': fs2,
}
plt.rcParams.update(params)

H, W = 1, 3
ticks_on = 1
max_val = 6

plt.figure()

# Panel 1: Radial profile scatter (all genes)
plt.subplot(H, W, 1)
for kg in range(0, N_genes):
    plt.scatter(radial_av_all[kg, :, 1], radial_av_all[kg, :, 0], s=10,
                color=fns_plot.return_colmaps('genes', N_var=N_genes)[kg])

plt.plot([0, 1e5], [0, 1e5], '-', color='grey', lw=0.5)

plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])

if ticks_on:
    plt.xticks([min_val, max_val])
    plt.yticks([min_val, max_val])
else:
    plt.xticks([])
    plt.yticks([])

plt.xlabel('pred. radial profile (a.u.)')
plt.ylabel('exp. radial profile (a.u.)')

# Panel 2: Conditional binned averages (all genes)
plt.subplot(H, W, 2)

# Analyze with same binning
N_bins = 30
hist_av_all = np.zeros((N_genes, N_bins))
hist_var_all = np.zeros((N_genes, N_bins))
hist_cond_av_all = np.zeros((N_genes, N_bins, N_bins))

max_val = 6
for kg in range(0, N_genes):
    bins, hist, hist_cond, hist_av, hist_var = calc_prediction_hist(
        tar_test[:, kg], target_predict[:, kg], N_bins, min_val, max_val
    )
    
    hist_av_all[kg, :] = hist_av
    hist_var_all[kg, :] = hist_var
    hist_cond_av_all[kg, :, :] = hist_cond

for kg in range(0, N_genes):
    plt.plot(bins, hist_av_all[kg, :], '-',
             color=fns_plot.return_colmaps('genes', N_var=N_genes)[kg])

plt.plot([0, 1e5], [0, 1e5], '-', color='grey', lw=0.5)

plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])

if ticks_on:
    plt.xticks([min_val, max_val])
    plt.yticks([min_val, max_val])
else:
    plt.xticks([])
    plt.yticks([])

plt.xlabel('pred. SC values (a.u.)')
plt.ylabel('exp. SC values (a.u.)')

# Panel 3: Averaged conditional histogram
plt.subplot(H, W, 3)

plt.imshow(np.rot90(np.nanmean(hist_cond_av_all, axis=0), k=1),
           extent=[min_val, max_val, min_val, max_val], cmap='YlGnBu', vmin=0, vmax=0.3)

hist_av_all_av = np.nanmean(hist_av_all, axis=0)
hist_var_all_av = np.nanmean(hist_var_all, axis=0)

plt.plot(bins, hist_av_all_av, '-k')
plt.plot(bins, hist_av_all_av - np.sqrt(hist_var_all_av), ':k')
plt.plot(bins, hist_av_all_av + np.sqrt(hist_var_all_av), ':k')

plt.plot([0, 1e5], [0, 1e5], '-', color='grey', lw=0.5)

plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])

if ticks_on:
    plt.xticks([min_val, max_val])
    plt.yticks([min_val, max_val])
else:
    plt.xticks([])
    plt.yticks([])

plt.xlabel('pred. SC values (a.u.)')
plt.ylabel('exp. SC values (a.u.)')

plt.tight_layout()
plt.savefig(subdirectory_plot_fig + "/" + 'SI_fig_av_sg_VIB2D_' + cond + file_suffix_pdf)
print(f"  Saved: SI_fig_av_sg_VIB2D_{cond}{file_suffix_pdf}")



plt.show()