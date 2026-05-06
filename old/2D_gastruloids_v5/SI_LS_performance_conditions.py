#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIB Error Probability Analysis Across Conditions
Trains VIB models in three different ways and visualizes error probabilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import dill as pickle
from sklearn.preprocessing import StandardScaler

import fns_plotting_scripts as fns_plot
import fns_NN

# Close all existing figures
plt.close('all')

# ============= CONFIGURATION =============
mode_expt = '20'
directory = 'data_expt_20_scaled_norm_bgsub'
subdirectory_data = directory + '/data'

# Conditions to analyze
conditions = ['B10', 'B50', 'B200']

# Genes to plot
PLOT_GENES = ['ISL1', 'TFAP2C', 'SOX17', 'TBXT', 'TBX6', 'NANOG', 'SOX2']

THRESHOLD = 1.0  # Threshold for binary classification
N_BINS = 50  # Number of bins for error probability heatmap
MIN_SAMPLES = 5  # Minimum samples per bin for analysis
TRAIN_SIZE = 3  # Number of colonies for training

# VIB model parameters
LATENT_DIM = 2
HIDDEN_DIM = 128
N_LAYERS = 2
EPOCHS = 800
LEARNING_RATE = 1e-3
BETA = 0.01

# Output directory
OUTPUT_DIR = directory + '/analysis_error_probability_multi_condition'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("VIB ERROR PROBABILITY ANALYSIS")
print("="*60)
print(f"Conditions: {conditions}")
print(f"Genes: {PLOT_GENES}")

# ============= LOAD ALL DATA =============
print("\nLoading data from all conditions...")

all_data = {}
for cond in conditions:
    print(f"  Loading {cond}...")
    f = open(subdirectory_data + '/data_' + cond + ".pickle", 'rb')
    data = pickle.load(f)
    f.close()
    
    feature = data.signals[:, :, :]
    target = data.genes[:, :, :]
    
    # Split by colonies
    (feat_train, feat_test, tar_train, tar_test,
     metricdist_train, metricdist_test, markers_train, markers_test) = fns_NN.test_train_split_colonies(
        data, feature, target, train_size=TRAIN_SIZE
    )
    
    # Combine for full dataset
    X_full = np.vstack([feat_train, feat_test])
    Y_full = np.vstack([tar_train, tar_test])
    
    all_data[cond] = {
        'X_train': feat_train,
        'Y_train': tar_train,
        'X_test': feat_test,
        'Y_test': tar_test,
        'X_full': X_full,
        'Y_full': Y_full,
        'signal_names': data.signal_names,
        'gene_names': data.gene_names
    }

OUTPUT_NAMES = all_data[conditions[0]]['gene_names']
N_DIM_INPUT = all_data[conditions[0]]['X_full'].shape[1]
N_DIM_OUTPUT = all_data[conditions[0]]['Y_full'].shape[1]

# Get indices of genes to plot
PLOT_GENE_INDICES = [list(OUTPUT_NAMES).index(gene) for gene in PLOT_GENES if gene in OUTPUT_NAMES]

print(f"Input dimensions: {N_DIM_INPUT}")
print(f"Output dimensions: {N_DIM_OUTPUT}")
print(f"Gene indices to plot: {PLOT_GENE_INDICES}")

# ============= HELPER FUNCTIONS =============

def train_vib_model(X_train, Y_train, verbose=False):
    """Train a VIB model"""
    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    X_train_torch = torch.FloatTensor(X_train_scaled)
    Y_train_torch = torch.FloatTensor(Y_train_scaled)
    
    # Create and train model
    vib = fns_NN.FlexibleVIB(
        input_dim=N_DIM_INPUT,
        output_dim=N_DIM_OUTPUT,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        encoder_type='nonlinear',
        decoder_type='nonlinear'
    )
    
    _ = fns_NN.train_model(
        vib, X_train_torch, Y_train_torch,
        is_vae=False,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        beta=BETA,
        verbose=verbose
    )
    
    return vib, scaler_X, scaler_Y


def get_predictions_and_latent(vib, scaler_X, scaler_Y, X_data, Y_data):
    """Get predictions and latent codes for data"""
    X_scaled = scaler_X.transform(X_data)
    X_torch = torch.FloatTensor(X_scaled)
    
    vib.eval()
    with torch.no_grad():
        latent_mu, _ = vib.encode(X_torch)
        latent_codes = latent_mu.numpy()
        Y_pred_scaled = vib(X_torch)[0].numpy()
    
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    
    return Y_pred, latent_codes


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


def compute_occupancy_mask(latent_codes, x_edges, y_edges, min_samples=5):
    """Compute which bins contain data (for background) - uses same min_samples as error computation"""
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1
    occupancy = np.zeros((n_bins_y, n_bins_x))
    
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            mask_x = (latent_codes[:, 0] >= x_edges[i]) & (latent_codes[:, 0] < x_edges[i+1])
            mask_y = (latent_codes[:, 1] >= y_edges[j]) & (latent_codes[:, 1] < y_edges[j+1])
            mask = mask_x & mask_y
            
            if np.sum(mask) >= min_samples:
                occupancy[j, i] = 1
    
    return occupancy


def get_latent_limits(all_latent_codes):
    """Get consistent axis limits across all conditions"""
    all_x = np.concatenate([lc[:, 0] for lc in all_latent_codes])
    all_y = np.concatenate([lc[:, 1] for lc in all_latent_codes])
    
    x_min, x_max = np.percentile(all_x, [0.1, 99.9])
    y_min, y_max = np.percentile(all_y, [0.1, 99.9])
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min_exp = x_min - 0.25 * x_range
    x_max_exp = x_max + 0.25 * x_range
    y_min_exp = y_min - 0.25 * y_range
    y_max_exp = y_max + 0.25 * y_range
    
    return x_min_exp, x_max_exp, y_min_exp, y_max_exp


# ============= SCENARIO 1: TRAIN ON EACH CONDITION =============
print("\n" + "="*60)
print("SCENARIO 1: Train on each condition separately")
print("="*60)

scenario1_results = {}

for cond in conditions:
    print(f"\nTraining on {cond}...")
    
    X_train = all_data[cond]['X_train']
    Y_train = all_data[cond]['Y_train']
    X_full = all_data[cond]['X_full']
    Y_full = all_data[cond]['Y_full']
    
    # Train model
    vib, scaler_X, scaler_Y = train_vib_model(X_train, Y_train, verbose=True)
    
    # Get predictions on full dataset
    Y_pred, latent_codes = get_predictions_and_latent(vib, scaler_X, scaler_Y, X_full, Y_full)
    
    scenario1_results[cond] = {
        'Y_true': Y_full,
        'Y_pred': Y_pred,
        'latent_codes': latent_codes
    }
    
    print(f"  {cond}: {len(latent_codes)} samples")

# Get consistent axis limits
all_latent_scenario1 = [scenario1_results[c]['latent_codes'] for c in conditions]
x_min, x_max, y_min, y_max = get_latent_limits(all_latent_scenario1)
x_edges = np.linspace(x_min, x_max, N_BINS + 1)
y_edges = np.linspace(y_min, y_max, N_BINS + 1)

# Plot
print("\nGenerating Scenario 1 plots...")
n_panels = len(PLOT_GENE_INDICES)
fig_width = 2 * n_panels
fig_height = 2 * len(conditions)

fig, axes = plt.subplots(len(conditions), n_panels, figsize=(fig_width, fig_height))
if len(conditions) == 1:
    axes = axes.reshape(1, -1)

for row_idx, cond in enumerate(conditions):
    Y_true = scenario1_results[cond]['Y_true']
    Y_pred = scenario1_results[cond]['Y_pred']
    latent_codes = scenario1_results[cond]['latent_codes']
    
    Y_true_binary = (Y_true > THRESHOLD).astype(int)
    Y_pred_binary = (Y_pred > THRESHOLD).astype(int)
    
    for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
        ax = axes[row_idx, panel_idx]
        
        y_true = Y_true_binary[:, target_idx]
        y_pred = Y_pred_binary[:, target_idx]
        
        error_prob = compute_error_probability(latent_codes, y_true, y_pred, x_edges, y_edges, min_samples=MIN_SAMPLES)
        
        im = ax.imshow(error_prob, origin='lower',
                       extent=[x_min, x_max, y_min, y_max],
                       cmap='RdYlGn_r', interpolation='nearest',
                       vmin=0, vmax=1, alpha=0.9)
        
        if row_idx == 0:
            ax.set_title(OUTPUT_NAMES[target_idx], fontsize=11, fontweight='bold')
        
        if panel_idx == 0:
            ax.set_ylabel(cond, fontsize=11, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig(os.path.join(OUTPUT_DIR, 'error_probability_scenario1_train_each.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: error_probability_scenario1_train_each.png")

# ============= SCENARIO 2: TRAIN ON B50 ONLY =============
print("\n" + "="*60)
print("SCENARIO 2: Train on B50, predict all conditions")
print("="*60)

print("\nTraining on B50...")
X_train_B50 = all_data['B50']['X_train']
Y_train_B50 = all_data['B50']['Y_train']

vib_B50, scaler_X_B50, scaler_Y_B50 = train_vib_model(X_train_B50, Y_train_B50, verbose=True)

scenario2_results = {}

for cond in conditions:
    print(f"\nPredicting on {cond}...")
    X_full = all_data[cond]['X_full']
    Y_full = all_data[cond]['Y_full']
    
    Y_pred, latent_codes = get_predictions_and_latent(vib_B50, scaler_X_B50, scaler_Y_B50, X_full, Y_full)
    
    scenario2_results[cond] = {
        'Y_true': Y_full,
        'Y_pred': Y_pred,
        'latent_codes': latent_codes
    }
    
    print(f"  {cond}: {len(latent_codes)} samples")

# Get consistent axis limits (SAME for all conditions)
all_latent_scenario2 = [scenario2_results[c]['latent_codes'] for c in conditions]
x_min_s2, x_max_s2, y_min_s2, y_max_s2 = get_latent_limits(all_latent_scenario2)
x_edges_s2 = np.linspace(x_min_s2, x_max_s2, N_BINS + 1)
y_edges_s2 = np.linspace(y_min_s2, y_max_s2, N_BINS + 1)

# Compute B50 occupancy mask for background (using same min_samples threshold)
latent_B50_background = scenario2_results['B50']['latent_codes']
occupancy_B50 = compute_occupancy_mask(latent_B50_background, x_edges_s2, y_edges_s2, min_samples=MIN_SAMPLES)

# Plot
print("\nGenerating Scenario 2 plots...")
fig, axes = plt.subplots(len(conditions), n_panels, figsize=(fig_width, fig_height))
if len(conditions) == 1:
    axes = axes.reshape(1, -1)

for row_idx, cond in enumerate(conditions):
    Y_true = scenario2_results[cond]['Y_true']
    Y_pred = scenario2_results[cond]['Y_pred']
    latent_codes = scenario2_results[cond]['latent_codes']
    
    Y_true_binary = (Y_true > THRESHOLD).astype(int)
    Y_pred_binary = (Y_pred > THRESHOLD).astype(int)
    
    for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
        ax = axes[row_idx, panel_idx]
        
        # Plot B50 occupancy as grey background
        background = np.where(occupancy_B50 > 0, 0.5, np.nan)  # Grey where data exists
        ax.imshow(background, origin='lower',
                 extent=[x_min_s2, x_max_s2, y_min_s2, y_max_s2],
                 cmap='gray', interpolation='nearest',
                 vmin=0, vmax=1, alpha=0.3, zorder=0)
        
        y_true = Y_true_binary[:, target_idx]
        y_pred = Y_pred_binary[:, target_idx]
        
        error_prob = compute_error_probability(latent_codes, y_true, y_pred, x_edges_s2, y_edges_s2, min_samples=MIN_SAMPLES)
        
        im = ax.imshow(error_prob, origin='lower',
                       extent=[x_min_s2, x_max_s2, y_min_s2, y_max_s2],
                       cmap='RdYlGn_r', interpolation='nearest',
                       vmin=0, vmax=1, alpha=0.9, zorder=1)
        
        if row_idx == 0:
            ax.set_title(OUTPUT_NAMES[target_idx], fontsize=11, fontweight='bold')
        
        if panel_idx == 0:
            ax.set_ylabel(cond, fontsize=11, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([x_min_s2, x_max_s2])
        ax.set_ylim([y_min_s2, y_max_s2])
        ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig(os.path.join(OUTPUT_DIR, 'error_probability_scenario2_train_B50.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: error_probability_scenario2_train_B50.png")

# ============= SCENARIO 3: TRAIN ON ALL CONDITIONS =============
print("\n" + "="*60)
print("SCENARIO 3: Train on all conditions combined")
print("="*60)

print("\nCombining training data from all conditions...")
X_train_all = np.vstack([all_data[c]['X_train'] for c in conditions])
Y_train_all = np.vstack([all_data[c]['Y_train'] for c in conditions])

print(f"Combined training samples: {X_train_all.shape[0]}")
print("Training model...")

vib_all, scaler_X_all, scaler_Y_all = train_vib_model(X_train_all, Y_train_all, verbose=True)

scenario3_results = {}

for cond in conditions:
    print(f"\nPredicting on {cond}...")
    X_full = all_data[cond]['X_full']
    Y_full = all_data[cond]['Y_full']
    
    Y_pred, latent_codes = get_predictions_and_latent(vib_all, scaler_X_all, scaler_Y_all, X_full, Y_full)
    
    scenario3_results[cond] = {
        'Y_true': Y_full,
        'Y_pred': Y_pred,
        'latent_codes': latent_codes
    }
    
    print(f"  {cond}: {len(latent_codes)} samples")

# Get consistent axis limits (SAME for all conditions)
all_latent_scenario3 = [scenario3_results[c]['latent_codes'] for c in conditions]
x_min_s3, x_max_s3, y_min_s3, y_max_s3 = get_latent_limits(all_latent_scenario3)
x_edges_s3 = np.linspace(x_min_s3, x_max_s3, N_BINS + 1)
y_edges_s3 = np.linspace(y_min_s3, y_max_s3, N_BINS + 1)

# Compute combined occupancy mask for background (using same min_samples threshold)
latent_all_background = np.vstack([scenario3_results[c]['latent_codes'] for c in conditions])
occupancy_all = compute_occupancy_mask(latent_all_background, x_edges_s3, y_edges_s3, min_samples=MIN_SAMPLES)

# Plot
print("\nGenerating Scenario 3 plots...")
fig, axes = plt.subplots(len(conditions), n_panels, figsize=(fig_width, fig_height))
if len(conditions) == 1:
    axes = axes.reshape(1, -1)

for row_idx, cond in enumerate(conditions):
    Y_true = scenario3_results[cond]['Y_true']
    Y_pred = scenario3_results[cond]['Y_pred']
    latent_codes = scenario3_results[cond]['latent_codes']
    
    Y_true_binary = (Y_true > THRESHOLD).astype(int)
    Y_pred_binary = (Y_pred > THRESHOLD).astype(int)
    
    for panel_idx, target_idx in enumerate(PLOT_GENE_INDICES):
        ax = axes[row_idx, panel_idx]
        
        # Plot combined occupancy as grey background
        background = np.where(occupancy_all > 0, 0.5, np.nan)  # Grey where data exists
        ax.imshow(background, origin='lower',
                 extent=[x_min_s3, x_max_s3, y_min_s3, y_max_s3],
                 cmap='gray', interpolation='nearest',
                 vmin=0, vmax=1, alpha=0.3, zorder=0)
        
        y_true = Y_true_binary[:, target_idx]
        y_pred = Y_pred_binary[:, target_idx]
        
        error_prob = compute_error_probability(latent_codes, y_true, y_pred, x_edges_s3, y_edges_s3, min_samples=MIN_SAMPLES)
        
        im = ax.imshow(error_prob, origin='lower',
                       extent=[x_min_s3, x_max_s3, y_min_s3, y_max_s3],
                       cmap='RdYlGn_r', interpolation='nearest',
                       vmin=0, vmax=1, alpha=0.9, zorder=1)
        
        if row_idx == 0:
            ax.set_title(OUTPUT_NAMES[target_idx], fontsize=11, fontweight='bold')
        
        if panel_idx == 0:
            ax.set_ylabel(cond, fontsize=11, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([x_min_s3, x_max_s3])
        ax.set_ylim([y_min_s3, y_max_s3])
        ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig(os.path.join(OUTPUT_DIR, 'error_probability_scenario3_train_all.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: error_probability_scenario3_train_all.png")

# ============= SUMMARY =============
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nThree scenarios analyzed:")
print("  1. Train on each condition separately")
print("  2. Train on B50 only, predict all conditions")
print("     (light grey background shows B50 sampled bins)")
print("  3. Train on all conditions combined")
print("     (light grey background shows all conditions sampled bins)")
print(f"\nAll figures saved to: {OUTPUT_DIR}\n")