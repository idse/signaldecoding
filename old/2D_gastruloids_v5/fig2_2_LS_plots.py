#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIB Visualization Script
Generates plots for single condition B50 using saved VIB results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
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
# PLOT_GENES = None  # Set to None to plot all genes (will be set to data.gene_names)

# Input/Output directories
subdirectory_data = directory + '/data'
subdirectory_plot = directory + '/analysis_regression_sg_multi_vib'
subdirectory_plot_data = subdirectory_plot + '/data'
OUTPUT_DIR = subdirectory_plot + '/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print(f"VIB VISUALIZATION - Condition: {cond}")
print("="*60)

# ============= LOAD ORIGINAL DATA =============
print("\nLoading original data...")
f = open(subdirectory_data + '/data_' + cond + ".pickle", 'rb')
data = pickle.load(f)
f.close()

INPUT_NAMES = data.signal_names
OUTPUT_NAMES = data.gene_names

# Set PLOT_GENES to all genes if not specified
if PLOT_GENES is None:
    PLOT_GENES = list(OUTPUT_NAMES)

# Get indices of genes to plot
PLOT_GENE_INDICES = [list(OUTPUT_NAMES).index(gene) for gene in PLOT_GENES if gene in OUTPUT_NAMES]
print(f"\nGenes to plot: {PLOT_GENES}")
print(f"Gene indices: {PLOT_GENE_INDICES}")

# ============= LOAD VIB RESULTS =============
print("\nLoading VIB results...")
f = open(subdirectory_plot_data + '/data_regression_VIB_' + cond + ".pickle", 'rb')
(vib, feat_train, feat_test, tar_train, tar_test, 
 metricdist_train, metricdist_test, markers_train, markers_test, 
 target_predict_train, target_predict, 
 scaler_X, scaler_Y) = pickle.load(f)
f.close()

# Combine train and test data for full visualization
X_raw = np.vstack([feat_train, feat_test])
Y_raw = np.vstack([tar_train, tar_test])
Y_pred_raw = np.vstack([target_predict_train, target_predict])
marker_labels = np.concatenate([markers_train, markers_test])

N_DIM_INPUT = X_raw.shape[1]
N_DIM_OUTPUT = Y_raw.shape[1]

print(f"Total samples: {X_raw.shape[0]}")
print(f"Input dimensions: {N_DIM_INPUT}")
print(f"Output dimensions: {N_DIM_OUTPUT}")

# Get cell fate markers
markers_clean, MARKER_NAMES = fns_plot.return_fates(Y_raw, OUTPUT_NAMES, thresh=1)
markers_pred, _ = fns_plot.return_fates(Y_pred_raw, OUTPUT_NAMES, thresh=1)

# ============= GET LATENT CODES =============
print("\nComputing latent codes...")
X_scaled = scaler_X.transform(X_raw)
X_full = torch.FloatTensor(X_scaled)

vib.eval()
with torch.no_grad():
    latent_mu, _ = vib.encode(X_full)
    latent_codes = latent_mu.numpy()

print(f"Latent codes shape: {latent_codes.shape}")

# Calculate axis limits based on percentiles
x_min, x_max = np.percentile(latent_codes[:, 0], [0.1, 99.9])
y_min, y_max = np.percentile(latent_codes[:, 1], [0.1, 99.9])

# Expand axis limits by 1.5x
x_range = x_max - x_min
y_range = y_max - y_min
x_min_expanded = x_min - 0.25 * x_range
x_max_expanded = x_max + 0.25 * x_range
y_min_expanded = y_min - 0.25 * y_range
y_max_expanded = y_max + 0.25 * y_range

print(f"Latent space limits (0.1-99.9 percentile):")
print(f"  Dimension 1: [{x_min:.3f}, {x_max:.3f}]")
print(f"  Dimension 2: [{y_min:.3f}, {y_max:.3f}]")

# Define colors and styles for markers
cmap = mpl.cm.get_cmap('tab10')
marker_colors = cmap(list(np.linspace(0, 1, 10)))
marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

unique_markers = np.unique(markers_clean)
has_valid_markers = not (len(unique_markers) == 1 and np.isnan(unique_markers[0]))

if has_valid_markers:
    unique_markers = unique_markers[~np.isnan(unique_markers)]

# ============= VISUALIZATIONS =============
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# ============= FIGURE 1: CELL FATE MARKERS (GROUND TRUTH) =============
print("\n1. Cell fate markers (ground truth)...")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

if has_valid_markers:
    for idx, marker_val in enumerate(unique_markers):
        mask = markers_clean == marker_val
        ax.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                   c=[marker_colors[idx % 10]], marker=marker_styles[idx % len(marker_styles)],
                   s=10, alpha=0.7, label=MARKER_NAMES[int(marker_val)], 
                   edgecolors='black', linewidth=0.3)
else:
    ax.scatter(latent_codes[:, 0], latent_codes[:, 1], 
               c='blue', marker='o', s=10, alpha=0.7, 
               edgecolors='black', linewidth=0.3)

ax.set_xlabel('Latent Dimension 1', fontsize=12)
ax.set_ylabel('Latent Dimension 2', fontsize=12)
ax.set_xlim([x_min_expanded, x_max_expanded])
ax.set_ylim([y_min_expanded, y_max_expanded])
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
if has_valid_markers:
    ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_cell_fates_true.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_cell_fates_true.png")

# ============= FIGURE 2: PREDICTED CELL FATE MARKERS =============
print("\n2. Cell fate markers (predicted)...")

unique_markers_pred = np.unique(markers_pred)
has_valid_markers_pred = not (len(unique_markers_pred) == 1 and np.isnan(unique_markers_pred[0]))

if has_valid_markers_pred:
    unique_markers_pred = unique_markers_pred[~np.isnan(unique_markers_pred)]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

if has_valid_markers_pred:
    for idx, marker_val in enumerate(unique_markers_pred):
        mask = markers_pred == marker_val
        if int(marker_val) < len(MARKER_NAMES):
            label = MARKER_NAMES[int(marker_val)]
        else:
            label = f"Fate {int(marker_val)}"
        
        ax.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                   c=[marker_colors[idx % 10]], marker=marker_styles[idx % len(marker_styles)],
                   s=10, alpha=0.7, label=label, 
                   edgecolors='black', linewidth=0.3)
else:
    ax.scatter(latent_codes[:, 0], latent_codes[:, 1], 
               c='blue', marker='o', s=10, alpha=0.7, 
               edgecolors='black', linewidth=0.3)

ax.set_xlabel('Latent Dimension 1', fontsize=12)
ax.set_ylabel('Latent Dimension 2', fontsize=12)
ax.set_xlim([x_min_expanded, x_max_expanded])
ax.set_ylim([y_min_expanded, y_max_expanded])
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
if has_valid_markers_pred:
    ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_cell_fates_predicted.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_cell_fates_predicted.png")

# ============= FIGURE 3: INPUT FEATURES (X) =============
print("\n3. Input features colored on latent space...")

n_panels = N_DIM_INPUT
fig_width = 2 * n_panels
fig_height = 2

fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
if n_panels == 1:
    axes = [axes]

for feature_idx in range(N_DIM_INPUT):
    ax = axes[feature_idx]
    feature_values = X_raw[:, feature_idx]
    
    # Sort by value so highest values appear on top
    sort_idx = np.argsort(feature_values)
    vmax = np.percentile(feature_values, 99)
    
    ax.scatter(latent_codes[sort_idx, 0], latent_codes[sort_idx, 1],
               c=feature_values[sort_idx], cmap='YlGnBu',
               vmin=0, vmax=vmax, s=5, alpha=0.8, edgecolors='none')
    
    ax.set_title(INPUT_NAMES[feature_idx], fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_inputs.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_inputs.png")

# ============= FIGURE 4: TARGET VALUES (Y TRUE) =============
print("\n4. True target values colored on latent space...")

n_panels = len(PLOT_GENE_INDICES)
fig_width = 2 * n_panels
fig_height = 2

fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
if n_panels == 1:
    axes = [axes]

for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
    ax = axes[panel_idx]
    target_values = Y_raw[:, target_idx]
    
    sort_idx = np.argsort(target_values)
    vmax = np.percentile(target_values, 99)
    
    ax.scatter(latent_codes[sort_idx, 0], latent_codes[sort_idx, 1],
               c=target_values[sort_idx], cmap='YlGnBu',
               vmin=0, vmax=vmax, s=5, alpha=0.8, edgecolors='none')
    
    ax.set_title(OUTPUT_NAMES[target_idx], fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_targets_true.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_targets_true.png")

# ============= FIGURE 5: PREDICTED TARGET VALUES (Y PRED) =============
print("\n5. Predicted target values colored on latent space...")

fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
if n_panels == 1:
    axes = [axes]

for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
    ax = axes[panel_idx]
    pred_values = Y_pred_raw[:, target_idx]
    
    sort_idx = np.argsort(pred_values)
    vmax = np.percentile(pred_values, 99)
    
    ax.scatter(latent_codes[sort_idx, 0], latent_codes[sort_idx, 1],
               c=pred_values[sort_idx], cmap='YlGnBu',
               vmin=0, vmax=vmax, s=5, alpha=0.8, edgecolors='none')
    
    ax.set_title(OUTPUT_NAMES[target_idx], fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_targets_predicted.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_targets_predicted.png")

# ============= FIGURE 6: PHASE DIAGRAMS =============
print("\n6. Cell fate phase diagrams...")

# Create high-resolution 2D grid
n_bins = 100
x_edges = np.linspace(x_min_expanded, x_max_expanded, n_bins + 1)
y_edges = np.linspace(y_min_expanded, y_max_expanded, n_bins + 1)

def compute_marker_phase_diagram(latent_codes, markers, x_edges, y_edges):
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1
    phase_diagram = np.full((n_bins_y, n_bins_x), np.nan)
    
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            mask_x = (latent_codes[:, 0] >= x_edges[i]) & (latent_codes[:, 0] < x_edges[i+1])
            mask_y = (latent_codes[:, 1] >= y_edges[j]) & (latent_codes[:, 1] < y_edges[j+1])
            mask = mask_x & mask_y
            
            if np.sum(mask) > 0:
                bin_markers = markers[mask]
                bin_markers_valid = bin_markers[~np.isnan(bin_markers)]
                
                if len(bin_markers_valid) > 0:
                    unique, counts = np.unique(bin_markers_valid, return_counts=True)
                    most_frequent = unique[np.argmax(counts)]
                    phase_diagram[j, i] = most_frequent
    
    return phase_diagram

phase_diagram_true = compute_marker_phase_diagram(latent_codes, markers_clean, x_edges, y_edges)
phase_diagram_pred = compute_marker_phase_diagram(latent_codes, markers_pred, x_edges, y_edges)

# Get unique markers across both
all_unique_markers = np.unique(np.concatenate([
    markers_clean[~np.isnan(markers_clean)],
    markers_pred[~np.isnan(markers_pred)]
]))

n_unique = len(all_unique_markers)
marker_colormap = ListedColormap(marker_colors[:n_unique])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: True markers
ax1 = axes[0]
im1 = ax1.imshow(phase_diagram_true, origin='lower', 
                 extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                 cmap=marker_colormap, interpolation='nearest',
                 vmin=all_unique_markers[0], vmax=all_unique_markers[-1])
ax1.set_xlabel('Latent Dimension 1', fontsize=12)
ax1.set_ylabel('Latent Dimension 2', fontsize=12)
ax1.set_aspect('equal')

# Panel 2: Predicted markers
ax2 = axes[1]
im2 = ax2.imshow(phase_diagram_pred, origin='lower',
                 extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                 cmap=marker_colormap, interpolation='nearest',
                 vmin=all_unique_markers[0], vmax=all_unique_markers[-1])
ax2.set_xlabel('Latent Dimension 1', fontsize=12)
ax2.set_ylabel('Latent Dimension 2', fontsize=12)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'marker_phase_diagrams.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: marker_phase_diagrams.png")

# ============= FIGURE 7: EXTRAPOLATED PHASE DIAGRAM =============
print("\n7. Extrapolated phase diagram...")

# Create KNN classifier for predicted markers
valid_mask_pred = ~np.isnan(markers_pred)
latent_valid_pred = latent_codes[valid_mask_pred]
markers_valid_pred = markers_pred[valid_mask_pred].astype(int)

if len(markers_valid_pred) > 0:
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(latent_valid_pred, markers_valid_pred)
    
    # Create dense grid
    n_grid = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min_expanded, x_max_expanded, n_grid),
        np.linspace(y_min_expanded, y_max_expanded, n_grid)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    predicted_markers_grid = knn.predict(grid_points)
    predicted_markers_grid = predicted_markers_grid.reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Pure extrapolation
    ax1 = axes[0]
    im1 = ax1.imshow(predicted_markers_grid, origin='lower',
                     extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                     cmap=marker_colormap, interpolation='bilinear',
                     vmin=all_unique_markers[0], vmax=all_unique_markers[-1],
                     alpha=1.0)
    
    ax1.set_xlabel('Latent Dimension 1', fontsize=13)
    ax1.set_ylabel('Latent Dimension 2', fontsize=13)
    ax1.set_xlim([x_min_expanded, x_max_expanded])
    ax1.set_ylim([y_min_expanded, y_max_expanded])
    ax1.set_aspect('equal')
    
    # Panel 2: Extrapolation with true marker overlay
    ax2 = axes[1]
    im2 = ax2.imshow(predicted_markers_grid, origin='lower',
                     extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                     cmap=marker_colormap, interpolation='bilinear',
                     vmin=all_unique_markers[0], vmax=all_unique_markers[-1],
                     alpha=0.3)
    
    if has_valid_markers:
        unique_markers_clean = np.unique(markers_clean)
        unique_markers_clean = unique_markers_clean[~np.isnan(unique_markers_clean)]
        
        for idx, marker_val in enumerate(unique_markers_clean):
            mask = markers_clean == marker_val
            if np.sum(mask) > 0:
                ax2.scatter(latent_codes[mask, 0], latent_codes[mask, 1],
                           c=[marker_colors[idx % 10]], 
                           marker=marker_styles[idx % len(marker_styles)],
                           s=5, alpha=0.8, 
                           label=MARKER_NAMES[int(marker_val)],
                           edgecolors='black', linewidth=0.2)
    
    ax2.set_xlabel('Latent Dimension 1', fontsize=13)
    ax2.set_ylabel('Latent Dimension 2', fontsize=13)
    ax2.set_xlim([x_min_expanded, x_max_expanded])
    ax2.set_ylim([y_min_expanded, y_max_expanded])
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'marker_phase_diagram_extrapolated.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: marker_phase_diagram_extrapolated.png")
else:
    print("Warning: No valid predicted markers for extrapolation.")

# ============= SUMMARY =============
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total samples: {len(markers_clean)}")
print(f"Unique cell fates: {len(unique_markers) if has_valid_markers else 'N/A'}")
if has_valid_markers:
    print(f"Cell fate names: {[MARKER_NAMES[int(m)] for m in unique_markers]}")

print("\n✓ VISUALIZATION COMPLETE")
print(f"All figures saved to: {OUTPUT_DIR}\n")