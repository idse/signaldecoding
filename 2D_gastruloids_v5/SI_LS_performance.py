#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIB Classification Performance Visualization
Shows correct/incorrect predictions in latent space for each gene
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import dill as pickle
import fns_plotting_scripts as fns_plot

# Close all existing figures
plt.close('all')

# ============= CONFIGURATION =============
mode_expt = '20'
directory = 'data_expt_20_scaled_norm_bgsub'
cond = 'B50'

# Genes to plot - either specify list or use all genes
PLOT_GENES = ['ISL1', 'TFAP2C', 'SOX17', 'TBXT', 'TBX6', 'NANOG', 'SOX2']
# PLOT_GENES = None  # Set to None to plot all genes

THRESHOLD = 1.0  # Threshold for binary classification
N_BINS = 50  # Number of bins for error probability heatmap

# Input/Output directories
subdirectory_data = directory + '/data'
subdirectory_plot = directory + '/analysis_regression_sg_multi_vib'
subdirectory_plot_data = subdirectory_plot + '/data'
OUTPUT_DIR = subdirectory_plot + '/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print(f"CLASSIFICATION PERFORMANCE ANALYSIS - Condition: {cond}")
print("="*60)

# ============= LOAD ORIGINAL DATA =============
print("\nLoading original data...")
f = open(subdirectory_data + '/data_' + cond + ".pickle", 'rb')
data = pickle.load(f)
f.close()

OUTPUT_NAMES = data.gene_names

# Set PLOT_GENES to all genes if not specified
if PLOT_GENES is None:
    PLOT_GENES = list(OUTPUT_NAMES)

# Get indices of genes to plot
PLOT_GENE_INDICES = [list(OUTPUT_NAMES).index(gene) for gene in PLOT_GENES if gene in OUTPUT_NAMES]
print(f"\nGenes to plot: {PLOT_GENES}")
print(f"Gene indices: {PLOT_GENE_INDICES}")
print(f"Classification threshold: {THRESHOLD}")

# ============= LOAD VIB RESULTS =============
print("\nLoading VIB results...")
f = open(subdirectory_plot_data + '/data_regression_VIB_' + cond + ".pickle", 'rb')
(vib, feat_train, feat_test, tar_train, tar_test, 
 metricdist_train, metricdist_test, markers_train, markers_test, 
 target_predict_train, target_predict, 
 scaler_X, scaler_Y) = pickle.load(f)
f.close()

# Combine train and test data
X_raw = np.vstack([feat_train, feat_test])
Y_raw = np.vstack([tar_train, tar_test])
Y_pred_raw = np.vstack([target_predict_train, target_predict])

print(f"Total samples: {X_raw.shape[0]}")

# ============= GET LATENT CODES =============
print("Computing latent codes...")
X_scaled = scaler_X.transform(X_raw)
X_full = torch.FloatTensor(X_scaled)

vib.eval()
with torch.no_grad():
    latent_mu, _ = vib.encode(X_full)
    latent_codes = latent_mu.numpy()

# Calculate axis limits
x_min, x_max = np.percentile(latent_codes[:, 0], [0.1, 99.9])
y_min, y_max = np.percentile(latent_codes[:, 1], [0.1, 99.9])

x_range = x_max - x_min
y_range = y_max - y_min
x_min_expanded = x_min - 0.25 * x_range
x_max_expanded = x_max + 0.25 * x_range
y_min_expanded = y_min - 0.25 * y_range
y_max_expanded = y_max + 0.25 * y_range

# ============= FIGURE 1: CLASSIFICATION PERFORMANCE (SCATTER) =============
print("\nGenerating classification performance scatter plot...")

Y_true_binary = (Y_raw > THRESHOLD).astype(int)
Y_pred_binary = (Y_pred_raw > THRESHOLD).astype(int)

n_panels = len(PLOT_GENE_INDICES)
fig_width = 2 * n_panels
fig_height = 2

fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
if n_panels == 1:
    axes = [axes]

for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
    ax = axes[panel_idx]
    
    y_true = Y_true_binary[:, target_idx]
    y_pred = Y_pred_binary[:, target_idx]
    
    correct = (y_true == y_pred)
    incorrect = (y_true != y_pred)
    
    # Plot correct (green) first with low alpha, then incorrect (red) on top
    if np.sum(correct) > 0:
        ax.scatter(latent_codes[correct, 0], latent_codes[correct, 1],
                  c='green', s=10, alpha=0.3, edgecolors='none')
    
    if np.sum(incorrect) > 0:
        ax.scatter(latent_codes[incorrect, 0], latent_codes[incorrect, 1],
                  c='red', s=10, alpha=0.6, edgecolors='none')
    
    # Calculate accuracy
    accuracy = np.sum(correct) / len(correct)
    
    ax.set_title(f'{OUTPUT_NAMES[target_idx]}\nAcc: {accuracy:.2f}', 
                fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_classification_performance.png'), 
            dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved: latent_space_classification_performance.png")

# ============= FIGURE 2: ERROR PROBABILITY HEATMAP =============
print("\nGenerating error probability heatmap...")

def compute_error_probability(latent_codes, y_true, y_pred, x_edges, y_edges, min_samples=5):
    """Compute probability of incorrect prediction in each bin"""
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1
    error_prob = np.full((n_bins_y, n_bins_x), np.nan)
    
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            mask_x = (latent_codes[:, 0] >= x_edges[i]) & (latent_codes[:, 0] < x_edges[i+1])
            mask_y = (latent_codes[:, 1] >= y_edges[j]) & (latent_codes[:, 1] < y_edges[j+1])
            mask = mask_x & mask_y
            
            if np.sum(mask) >= min_samples:
                bin_true = y_true[mask]
                bin_pred = y_pred[mask]
                incorrect = (bin_true != bin_pred)
                error_prob[j, i] = np.mean(incorrect)
    
    return error_prob

# Create bins
x_edges = np.linspace(x_min_expanded, x_max_expanded, N_BINS + 1)
y_edges = np.linspace(y_min_expanded, y_max_expanded, N_BINS + 1)

n_panels = len(PLOT_GENE_INDICES)
fig_width = 2 * n_panels
fig_height = 2

fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
if n_panels == 1:
    axes = [axes]

for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
    ax = axes[panel_idx]
    
    y_true = Y_true_binary[:, target_idx]
    y_pred = Y_pred_binary[:, target_idx]
    
    # Compute error probability
    error_prob = compute_error_probability(latent_codes, y_true, y_pred, x_edges, y_edges)
    
    # Plot heatmap (green = all correct, red = all wrong)
    # Use reversed RdYlGn colormap so green = low error (good), red = high error (bad)
    im = ax.imshow(error_prob, origin='lower',
                   extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                   cmap='RdYlGn_r', interpolation='nearest',
                   vmin=0, vmax=1, alpha=0.9)
    
    ax.set_title(OUTPUT_NAMES[target_idx], fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_error_probability.png'), 
            dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved: latent_space_error_probability.png")

print(f"\nResults saved to: {OUTPUT_DIR}\n")