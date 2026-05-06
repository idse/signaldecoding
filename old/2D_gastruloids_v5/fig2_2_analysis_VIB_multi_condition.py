#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIB Multi-Condition Analysis
Analyzes multiple experimental conditions in a shared 2D latent space
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
import os
import dill as pickle
import fns_plotting_scripts as fns_plot
import fns_NN

# Close all existing figures
plt.close('all')

# ============= CONFIGURATION =============
LATENT_DIM = 2
EPOCHS = 800
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
BETA = 0.01
N_LAYERS = 2

# Train/Test split
TRAIN_SIZE = 3  # Number of colonies to use for training (rest for testing)

# VIB Training Mode
MODE_TRAIN = True  # Set to False to load pre-trained model and results

# VIB Training Strategy
# Options: 'all_conditions' or 'single_condition'
TRAINING_STRATEGY = 'single_condition'  # 'all_conditions' or 'single_condition'
TRAINING_CONDITION = 'B50'  # Only used if TRAINING_STRATEGY = 'single_condition'

# Data paths
directory = 'data_expt_20_scaled_norm_bgsub'
subdirectory_data = directory + '/data'
OUTPUT_DIR = directory + '/fig2_2_VIB_multi_condition'
VIB_DATA_DIR = OUTPUT_DIR + '/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIB_DATA_DIR, exist_ok=True)

# Conditions to analyze
conditions = ['B10', 'B50', 'B200']

print("="*60)
print("LOADING DATA FROM MULTIPLE CONDITIONS")
print("="*60)

# ============= LOAD DATA =============

all_data = {}
for cond in conditions:
    print(f"\nLoading condition: {cond}")
    f = open(subdirectory_data + '/data_' + cond + ".pickle", 'rb')
    data = pickle.load(f)
    f.close()
    
    # Use colony-based train/test split
    feature = data.signals
    target = data.genes
    
    (feat_train, feat_test, tar_train, tar_test,
     metricdist_train, metricdist_test, markers_train, markers_test) = fns_NN.test_train_split_colonies(
        data, feature, target, train_size=TRAIN_SIZE
    )
    
    # Combine train and test for visualization (we'll track which is which)
    feature_clean = np.vstack([feat_train, feat_test])
    target_clean = np.vstack([tar_train, tar_test])
    markers_clean = np.concatenate([markers_train, markers_test])
    
    # Create split labels
    split_labels = np.array(['train'] * len(feat_train) + ['test'] * len(feat_test))
    
    markers_clean, fate_names = fns_plot.return_fates(target_clean, data.gene_names, thresh=1)
    
    all_data[cond] = {
        'X': feature_clean,
        'Y': target_clean,
        'markers': markers_clean,
        'split_labels': split_labels,
        'X_train': feat_train,
        'Y_train': tar_train,
        'X_test': feat_test,
        'Y_test': tar_test,
        'signal_names': data.signal_names,
        'gene_names': data.gene_names,
        'fate_names': fate_names
    }
    print(f"  {cond}: {len(feat_train)} train, {len(feat_test)} test samples")

# Use the first condition to get dimension names
INPUT_NAMES = all_data[conditions[0]]['signal_names']
OUTPUT_NAMES = all_data[conditions[0]]['gene_names']
MARKER_NAMES = all_data[conditions[0]]['fate_names']

N_DIM_INPUT = all_data[conditions[0]]['X'].shape[1]
N_DIM_OUTPUT = all_data[conditions[0]]['Y'].shape[1]

# ============= COMBINE DATA =============

print("\n" + "="*60)
print("COMBINING DATA FROM ALL CONDITIONS")
print("="*60)

# Concatenate all conditions
X_list = []
Y_list = []
marker_list = []
condition_labels = []  # Track which condition each sample comes from
split_labels_list = []  # Track train/test split

for cond in conditions:
    X_list.append(all_data[cond]['X'])
    Y_list.append(all_data[cond]['Y'])
    marker_list.append(all_data[cond]['markers'])
    split_labels_list.append(all_data[cond]['split_labels'])
    condition_labels.extend([cond] * all_data[cond]['X'].shape[0])

X_raw = np.vstack(X_list)
Y_raw = np.vstack(Y_list)
marker_labels = np.concatenate(marker_list)
condition_labels = np.array(condition_labels)
split_labels = np.concatenate(split_labels_list)

print(f"\nCombined dataset:")
print(f"  Total samples: {X_raw.shape[0]}")
print(f"  Train samples: {np.sum(split_labels == 'train')}")
print(f"  Test samples: {np.sum(split_labels == 'test')}")
print(f"  Input dimensions: {N_DIM_INPUT}")
print(f"  Output dimensions: {N_DIM_OUTPUT}")
print(f"  Conditions: {conditions}")

# ============= CELL FATE PROPORTION ANALYSIS =============

print("\n" + "="*60)
print("ANALYZING CELL FATE PROPORTIONS")
print("="*60)

# Define colors for markers using tab10
cmap = mpl.cm.get_cmap('tab10')
marker_colors = cmap(list(np.linspace(0, 1, 10)))

# Check if we have valid markers
unique_markers_all = np.unique(marker_labels)
has_valid_markers = not (len(unique_markers_all) == 1 and np.isnan(unique_markers_all[0]))

if has_valid_markers:
    # Remove NaN from unique_markers if present
    unique_markers_all = unique_markers_all[~np.isnan(unique_markers_all)]
    
    # Calculate proportions for each condition
    proportions_dict = {}
    
    for cond in conditions:
        cond_mask = condition_labels == cond
        cond_markers = marker_labels[cond_mask]
        
        # Count each fate
        fate_counts = {}
        for marker_val in unique_markers_all:
            count = np.sum(cond_markers == marker_val)
            fate_counts[int(marker_val)] = count
        
        # Convert to proportions
        total = len(cond_markers)
        proportions = {fate: count/total for fate, count in fate_counts.items()}
        proportions_dict[cond] = proportions
        
        print(f"\n{cond}:")
        for fate_idx in sorted(fate_counts.keys()):
            fate_name = MARKER_NAMES[fate_idx]
            count = fate_counts[fate_idx]
            pct = proportions[fate_idx] * 100
            print(f"  {fate_name}: {count} cells ({pct:.1f}%)")
    
    # Create stacked bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    
    # Prepare data for stacked bars
    n_conditions = len(conditions)
    bar_width = 0.6
    x_pos = np.arange(n_conditions)
    
    # Create stacked bars
    bottom = np.zeros(n_conditions)
    
    for idx, marker_val in enumerate(sorted(unique_markers_all)):
        marker_idx = int(marker_val)
        heights = [proportions_dict[cond].get(marker_idx, 0) for cond in conditions]
        
        ax_bar.bar(x_pos, heights, bar_width, 
                  bottom=bottom,
                  label=MARKER_NAMES[marker_idx],
                  color=marker_colors[idx % 10],
                  edgecolor='black',
                  linewidth=0.5)
        
        # Add percentage labels on bars (only if > 5%)
        for i, (height, cond) in enumerate(zip(heights, conditions)):
            if height > 0.05:  # Only show if > 5%
                label_y = bottom[i] + height/2
                ax_bar.text(i, label_y, f'{height*100:.0f}%', 
                          ha='center', va='center', 
                          fontsize=9, fontweight='bold',
                          color='white' if height > 0.15 else 'black')
        
        bottom += heights
    
    # Formatting
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(conditions, fontsize=12)
    ax_bar.set_ylabel('Proportion of Cells', fontsize=12)
    ax_bar.set_xlabel('Condition', fontsize=12)
    ax_bar.set_title('Cell Fate Proportions Across Conditions', 
                     fontsize=14, fontweight='bold', pad=15)
    ax_bar.set_ylim([0, 1])
    ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                 fontsize=10, framealpha=0.9)
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cell_fate_proportions_stacked.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: cell_fate_proportions_stacked.png")
    
else:
    print("Warning: All markers are NaN. Skipping cell fate proportion analysis.")

# ============= REMOVE JUNK DATA =============

print("\n" + "="*60)
print("REMOVING JUNK DATA (MARKER = 6)")
print("="*60)

# Check if marker 6 exists in the data
if 6 in marker_labels:
    # Find indices where marker is NOT 6
    keep_mask = marker_labels != 6
    
    # Count how many samples will be removed
    n_removed = np.sum(~keep_mask)
    n_total = len(marker_labels)
    
    print(f"Removing {n_removed} samples with marker = 6 ({n_removed/n_total*100:.1f}% of data)")
    
    # Filter all arrays
    X_raw = X_raw[keep_mask]
    Y_raw = Y_raw[keep_mask]
    marker_labels = marker_labels[keep_mask]
    condition_labels = condition_labels[keep_mask]
    split_labels = split_labels[keep_mask]
    
    print(f"Remaining samples: {len(marker_labels)}")
    
    # Print breakdown by condition
    print("\nRemaining samples per condition:")
    for cond in conditions:
        n_samples = np.sum(condition_labels == cond)
        print(f"  {cond}: {n_samples} samples ({n_samples/len(condition_labels)*100:.1f}%)")
else:
    print("No samples with marker = 6 found. No data removed.")

# ============= PREPARE DATA FOR TRAINING =============

print("\n" + "="*60)
print("PREPARING DATA FOR VIB TRAINING")
print("="*60)

if TRAINING_STRATEGY == 'single_condition':
    print(f"Strategy: Training VIB on {TRAINING_CONDITION} only")
    print(f"Other conditions will be projected onto this latent space")
    
    # Get training condition data (only train split)
    train_mask = (condition_labels == TRAINING_CONDITION) & (split_labels == 'train')
    test_mask = (condition_labels == TRAINING_CONDITION) & (split_labels == 'test')
    
    X_train_raw = X_raw[train_mask]
    Y_train_raw = Y_raw[train_mask]
    X_test_raw = X_raw[test_mask]
    Y_test_raw = Y_raw[test_mask]
    
else:  # all_conditions
    print("Strategy: Training VIB on all conditions combined")
    
    # Use train split from all conditions
    train_mask = split_labels == 'train'
    test_mask = split_labels == 'test'
    
    X_train_raw = X_raw[train_mask]
    Y_train_raw = Y_raw[train_mask]
    X_test_raw = X_raw[test_mask]
    Y_test_raw = Y_raw[test_mask]

# Standardize
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)
X_train = torch.FloatTensor(X_train_scaled)
Y_train = torch.FloatTensor(Y_train_scaled)

X_test_scaled = scaler_X.transform(X_test_raw)
Y_test_scaled = scaler_Y.transform(Y_test_raw)
X_test = torch.FloatTensor(X_test_scaled)
Y_test = torch.FloatTensor(Y_test_scaled)

# Full dataset for visualization
X_scaled = scaler_X.transform(X_raw)
Y_scaled = scaler_Y.transform(Y_raw)
X_full = torch.FloatTensor(X_scaled)
Y_full = torch.FloatTensor(Y_scaled)

print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Test samples: {X_test_scaled.shape[0]}")
print(f"Total samples for visualization: {X_scaled.shape[0]}")

# ============= TRAIN VIB =============

print("\n" + "="*60)
print("VIB MODEL TRAINING/LOADING")
print("="*60)

if MODE_TRAIN:
    print("Mode: TRAINING")
    print("Training 2D VIB model...")
    
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
        vib, X_train, Y_train, 
        is_vae=False, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        beta=BETA,
        verbose=True,
        print_every=200
    )
    
    # Evaluate on test set
    test_loss = fns_NN.evaluate_model(vib, X_test, Y_test, is_vae=False)
    print(f"\nFinal test loss: {test_loss:.4f}")
    
    # Get latent codes and predictions for full dataset
    vib.eval()
    with torch.no_grad():
        latent_mu, _ = vib.encode(X_full)
        latent_codes = latent_mu.numpy()
        Y_pred_full = vib.decode(latent_mu).numpy()
    
    # Save model and results
    print(f"\nSaving model and results to {VIB_DATA_DIR}...")
    torch.save(vib.state_dict(), os.path.join(VIB_DATA_DIR, 'vib_model.pth'))
    
    # Save other necessary data
    save_data = {
        'latent_codes': latent_codes,
        'Y_pred_full': Y_pred_full,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y,
        'X_raw': X_raw,
        'Y_raw': Y_raw,
        'marker_labels': marker_labels,
        'condition_labels': condition_labels,
        'split_labels': split_labels,
        'INPUT_NAMES': INPUT_NAMES,
        'OUTPUT_NAMES': OUTPUT_NAMES,
        'MARKER_NAMES': MARKER_NAMES,
        'N_DIM_INPUT': N_DIM_INPUT,
        'N_DIM_OUTPUT': N_DIM_OUTPUT,
        'TRAINING_STRATEGY': TRAINING_STRATEGY,
        'TRAINING_CONDITION': TRAINING_CONDITION,
        'test_loss': test_loss
    }
    
    with open(os.path.join(VIB_DATA_DIR, 'vib_results.pickle'), 'wb') as f:
        pickle.dump(save_data, f)
    
    print("✓ Model and results saved")
    
else:
    print("Mode: LOADING")
    print(f"Loading pre-trained model and results from {VIB_DATA_DIR}...")
    
    # Load saved data
    with open(os.path.join(VIB_DATA_DIR, 'vib_results.pickle'), 'rb') as f:
        save_data = pickle.load(f)
    
    latent_codes = save_data['latent_codes']
    Y_pred_full = save_data['Y_pred_full']
    scaler_X = save_data['scaler_X']
    scaler_Y = save_data['scaler_Y']
    X_raw = save_data['X_raw']
    Y_raw = save_data['Y_raw']
    marker_labels = save_data['marker_labels']
    condition_labels = save_data['condition_labels']
    split_labels = save_data['split_labels']
    INPUT_NAMES = save_data['INPUT_NAMES']
    OUTPUT_NAMES = save_data['OUTPUT_NAMES']
    MARKER_NAMES = save_data['MARKER_NAMES']
    N_DIM_INPUT = save_data['N_DIM_INPUT']
    N_DIM_OUTPUT = save_data['N_DIM_OUTPUT']
    
    # Load model (optional, only if needed for further predictions)
    vib = fns_NN.FlexibleVIB(
        input_dim=N_DIM_INPUT, 
        output_dim=N_DIM_OUTPUT, 
        latent_dim=LATENT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        n_layers=N_LAYERS,
        encoder_type='nonlinear',
        decoder_type='nonlinear'
    )
    vib.load_state_dict(torch.load(os.path.join(VIB_DATA_DIR, 'vib_model.pth'), 
                                   weights_only=True))
    vib.eval()
    
    print("✓ Model and results loaded")
    if 'test_loss' in save_data:
        print(f"Test loss: {save_data['test_loss']:.4f}")

print(f"Latent codes shape: {latent_codes.shape}")

# Inverse transform predictions to original scale (needed for all visualizations)
Y_pred_original = scaler_Y.inverse_transform(Y_pred_full)

# Calculate axis limits based on percentiles
x_min, x_max = np.percentile(latent_codes[:, 0], [0.1, 99.9])
y_min, y_max = np.percentile(latent_codes[:, 1], [0.1, 99.9])

# Expand axis limits by 1.5x for all plots
x_range = x_max - x_min
y_range = y_max - y_min
x_min_expanded = x_min - 0.25 * x_range
x_max_expanded = x_max + 0.25 * x_range
y_min_expanded = y_min - 0.25 * y_range
y_max_expanded = y_max + 0.25 * y_range

print(f"Latent space limits (0.1-99.9 percentile):")
print(f"  Dimension 1: [{x_min:.3f}, {x_max:.3f}]")
print(f"  Dimension 2: [{y_min:.3f}, {y_max:.3f}]")
print(f"Expanded limits (1.5x):")
print(f"  Dimension 1: [{x_min_expanded:.3f}, {x_max_expanded:.3f}]")
print(f"  Dimension 2: [{y_min_expanded:.3f}, {y_max_expanded:.3f}]")

# Define colors and styles for markers
marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

unique_markers = np.unique(marker_labels)

# ============= VISUALIZATIONS =============

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# ============= FIGURE 1: CELL FATE MARKERS =============
fig = plt.figure(figsize=(16, 14))

# Check if markers are all NaN
has_valid_markers = not (len(unique_markers) == 1 and np.isnan(unique_markers[0]))

if has_valid_markers:
    # Remove NaN from unique_markers if present
    unique_markers = unique_markers[~np.isnan(unique_markers)]
    n_markers = len(unique_markers)
else:
    n_markers = 0
    print("Warning: All markers are NaN. Plotting without marker coloring.")

# SUBPLOT 1: ALL DATA
ax1 = plt.subplot(2, 2, 1)

if has_valid_markers:
    for idx, marker_val in enumerate(unique_markers):
        mask = marker_labels == marker_val
        ax1.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                   c=[marker_colors[idx % 10]], marker=marker_styles[idx % len(marker_styles)],
                   s=30, alpha=0.7, label=MARKER_NAMES[int(marker_val)], 
                   edgecolors='black', linewidth=0.3)
    title_suffix = '(Colored by Cell Fate)'
else:
    ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], 
               c='blue', marker='o', s=30, alpha=0.7, 
               edgecolors='black', linewidth=0.3)
    title_suffix = '(No Marker Data Available)'

ax1.set_xlabel('Latent Dimension 1', fontsize=12)
ax1.set_ylabel('Latent Dimension 2', fontsize=12)
ax1.set_title(f'All Conditions Combined\n{title_suffix}', fontsize=13, fontweight='bold')
ax1.set_xlim([x_min_expanded, x_max_expanded])
ax1.set_ylim([y_min_expanded, y_max_expanded])
ax1.grid(True, alpha=0.3)
if has_valid_markers:
    ax1.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

# SUBPLOTS 2-4: INDIVIDUAL CONDITIONS
for plot_idx, cond in enumerate(conditions):
    ax = plt.subplot(2, 2, plot_idx + 2)
    
    # Plot all data in grey background
    ax.scatter(latent_codes[:, 0], latent_codes[:, 1], 
              c='lightgray', s=15, alpha=0.3, edgecolors='none')
    
    # Overlay specific condition data
    cond_mask = condition_labels == cond
    
    if has_valid_markers:
        for idx, marker_val in enumerate(unique_markers):
            mask = cond_mask & (marker_labels == marker_val)
            if np.sum(mask) > 0:
                ax.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                          c=[marker_colors[idx % 10]], marker=marker_styles[idx % len(marker_styles)],
                          s=35, alpha=0.8, label=MARKER_NAMES[int(marker_val)], 
                          edgecolors='black', linewidth=0.4)
        title_suffix = '(Colored by Cell Fate, Grey = All Data)'
    else:
        ax.scatter(latent_codes[cond_mask, 0], latent_codes[cond_mask, 1], 
                  c='blue', marker='o', s=35, alpha=0.8, 
                  label=cond, edgecolors='black', linewidth=0.4)
        title_suffix = '(Blue = This Condition, Grey = All Data)'
    
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title(f'Condition: {cond}\n{title_suffix}', 
                fontsize=13, fontweight='bold')
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

plt.suptitle('Multi-Condition Latent Space Analysis - Cell Fate Markers (Ground Truth)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_cell_fates.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_cell_fates.png")

# ============= FIGURE: PREDICTED CELL FATE MARKERS =============

print("\n" + "="*60)
print("CALCULATING PREDICTED CELL FATE MARKERS")
print("="*60)

# Calculate predicted markers using return_fates function
marker_labels_pred, fate_names_pred = fns_plot.return_fates(Y_pred_original, OUTPUT_NAMES, thresh=1)

print(f"Predicted fate names: {fate_names_pred}")
print(f"Unique predicted markers: {np.unique(marker_labels_pred)}")

# Create figure
fig = plt.figure(figsize=(16, 14))

unique_markers_pred = np.unique(marker_labels_pred)

# Check if predicted markers are all NaN
has_valid_markers_pred = not (len(unique_markers_pred) == 1 and np.isnan(unique_markers_pred[0]))

if has_valid_markers_pred:
    # Remove NaN from unique_markers_pred if present
    unique_markers_pred = unique_markers_pred[~np.isnan(unique_markers_pred)]
    n_markers_pred = len(unique_markers_pred)
else:
    n_markers_pred = 0
    print("Warning: All predicted markers are NaN.")

# SUBPLOT 1: ALL DATA
ax1 = plt.subplot(2, 2, 1)

if has_valid_markers_pred:
    for idx, marker_val in enumerate(unique_markers_pred):
        mask = marker_labels_pred == marker_val
        # Use MARKER_NAMES if the index exists, otherwise use fate_names_pred
        if int(marker_val) < len(MARKER_NAMES):
            label = MARKER_NAMES[int(marker_val)]
        else:
            label = fate_names_pred[int(marker_val)] if int(marker_val) < len(fate_names_pred) else f"Fate {int(marker_val)}"
        
        ax1.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                   c=[marker_colors[idx % 10]], marker=marker_styles[idx % len(marker_styles)],
                   s=30, alpha=0.7, label=label, 
                   edgecolors='black', linewidth=0.3)
    title_suffix = '(Colored by Predicted Cell Fate)'
else:
    ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], 
               c='blue', marker='o', s=30, alpha=0.7, 
               edgecolors='black', linewidth=0.3)
    title_suffix = '(No Predicted Marker Data)'

ax1.set_xlabel('Latent Dimension 1', fontsize=12)
ax1.set_ylabel('Latent Dimension 2', fontsize=12)
ax1.set_title(f'All Conditions Combined\n{title_suffix}', fontsize=13, fontweight='bold')
ax1.set_xlim([x_min_expanded, x_max_expanded])
ax1.set_ylim([y_min_expanded, y_max_expanded])
ax1.grid(True, alpha=0.3)
if has_valid_markers_pred:
    ax1.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

# SUBPLOTS 2-4: INDIVIDUAL CONDITIONS
for plot_idx, cond in enumerate(conditions):
    ax = plt.subplot(2, 2, plot_idx + 2)
    
    # Plot all data in grey background
    ax.scatter(latent_codes[:, 0], latent_codes[:, 1], 
              c='lightgray', s=15, alpha=0.3, edgecolors='none')
    
    # Overlay specific condition data
    cond_mask = condition_labels == cond
    
    if has_valid_markers_pred:
        for idx, marker_val in enumerate(unique_markers_pred):
            mask = cond_mask & (marker_labels_pred == marker_val)
            if np.sum(mask) > 0:
                # Use MARKER_NAMES if the index exists, otherwise use fate_names_pred
                if int(marker_val) < len(MARKER_NAMES):
                    label = MARKER_NAMES[int(marker_val)]
                else:
                    label = fate_names_pred[int(marker_val)] if int(marker_val) < len(fate_names_pred) else f"Fate {int(marker_val)}"
                
                ax.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                          c=[marker_colors[idx % 10]], marker=marker_styles[idx % len(marker_styles)],
                          s=35, alpha=0.8, label=label, 
                          edgecolors='black', linewidth=0.4)
        title_suffix = '(Colored by Predicted Cell Fate, Grey = All Data)'
    else:
        ax.scatter(latent_codes[cond_mask, 0], latent_codes[cond_mask, 1], 
                  c='blue', marker='o', s=35, alpha=0.8, 
                  label=cond, edgecolors='black', linewidth=0.4)
        title_suffix = '(Blue = This Condition, Grey = All Data)'
    
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title(f'Condition: {cond}\n{title_suffix}', 
                fontsize=13, fontweight='bold')
    ax.set_xlim([x_min_expanded, x_max_expanded])
    ax.set_ylim([y_min_expanded, y_max_expanded])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

plt.suptitle('Multi-Condition Latent Space Analysis - Predicted Cell Fate Markers', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space_predicted_cell_fates.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: latent_space_predicted_cell_fates.png")

# ============= FIGURE: PHASE DIAGRAM OF MARKER IDENTITY =============

print("\n" + "="*60)
print("GENERATING MARKER PHASE DIAGRAMS")
print("="*60)

# Create high-resolution 2D grid
n_bins = 100
x_edges = np.linspace(x_min_expanded, x_max_expanded, n_bins + 1)
y_edges = np.linspace(y_min_expanded, y_max_expanded, n_bins + 1)

# Function to compute most frequent marker in each bin
def compute_marker_phase_diagram(latent_codes, markers, x_edges, y_edges):
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1
    phase_diagram = np.full((n_bins_y, n_bins_x), np.nan)
    
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            # Find points in this bin
            mask_x = (latent_codes[:, 0] >= x_edges[i]) & (latent_codes[:, 0] < x_edges[i+1])
            mask_y = (latent_codes[:, 1] >= y_edges[j]) & (latent_codes[:, 1] < y_edges[j+1])
            mask = mask_x & mask_y
            
            if np.sum(mask) > 0:
                # Get markers in this bin
                bin_markers = markers[mask]
                # Remove NaN values
                bin_markers_valid = bin_markers[~np.isnan(bin_markers)]
                
                if len(bin_markers_valid) > 0:
                    # Find most frequent marker
                    unique, counts = np.unique(bin_markers_valid, return_counts=True)
                    most_frequent = unique[np.argmax(counts)]
                    phase_diagram[j, i] = most_frequent
    
    return phase_diagram

# Compute phase diagrams for true and predicted markers
phase_diagram_true = compute_marker_phase_diagram(latent_codes, marker_labels, x_edges, y_edges)
phase_diagram_pred = compute_marker_phase_diagram(latent_codes, marker_labels_pred, x_edges, y_edges)

# Create figure with 2 panels
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Get unique markers across both true and predicted
all_unique_markers = np.unique(np.concatenate([
    marker_labels[~np.isnan(marker_labels)],
    marker_labels_pred[~np.isnan(marker_labels_pred)]
]))

# Create colormap for markers
n_unique = len(all_unique_markers)
from matplotlib.colors import ListedColormap
marker_colormap = ListedColormap(marker_colors[:n_unique])

# Panel 1: True markers
ax1 = axes[0]
im1 = ax1.imshow(phase_diagram_true, origin='lower', 
                 extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                 cmap=marker_colormap, interpolation='nearest',
                 vmin=all_unique_markers[0], vmax=all_unique_markers[-1])
ax1.set_xlabel('Latent Dimension 1', fontsize=12)
ax1.set_ylabel('Latent Dimension 2', fontsize=12)
ax1.set_title('True Cell Fate Phase Diagram', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')

# Add colorbar for true markers
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Cell Fate', fontsize=11)
# Set colorbar ticks to marker values
cbar1.set_ticks(all_unique_markers)
if has_valid_markers:
    cbar1.set_ticklabels([MARKER_NAMES[int(m)] if int(m) < len(MARKER_NAMES) else f'Fate {int(m)}' 
                          for m in all_unique_markers])

# Panel 2: Predicted markers
ax2 = axes[1]
im2 = ax2.imshow(phase_diagram_pred, origin='lower',
                 extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                 cmap=marker_colormap, interpolation='nearest',
                 vmin=all_unique_markers[0], vmax=all_unique_markers[-1])
ax2.set_xlabel('Latent Dimension 1', fontsize=12)
ax2.set_ylabel('Latent Dimension 2', fontsize=12)
ax2.set_title('Predicted Cell Fate Phase Diagram', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')

# Add colorbar for predicted markers
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Cell Fate', fontsize=11)
cbar2.set_ticks(all_unique_markers)
if has_valid_markers:
    cbar2.set_ticklabels([MARKER_NAMES[int(m)] if int(m) < len(MARKER_NAMES) else f'Fate {int(m)}' 
                          for m in all_unique_markers])

plt.suptitle('Cell Fate Phase Diagrams - Most Frequent Marker per Bin', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'marker_phase_diagrams.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: marker_phase_diagrams.png")

# ============= FIGURE: EXTRAPOLATED PHASE DIAGRAM =============

print("\n" + "="*60)
print("GENERATING EXTRAPOLATED MARKER PHASE DIAGRAM")
print("="*60)

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier for predicted markers
# Remove NaN values for training
valid_mask_pred = ~np.isnan(marker_labels_pred)
latent_valid_pred = latent_codes[valid_mask_pred]
markers_valid_pred = marker_labels_pred[valid_mask_pred].astype(int)

if len(markers_valid_pred) > 0:
    # Train KNN classifier (using k=15 for smooth boundaries)
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(latent_valid_pred, markers_valid_pred)
    
    # Create dense grid for extrapolation
    n_grid = 200  # Higher resolution for smooth visualization
    xx, yy = np.meshgrid(
        np.linspace(x_min_expanded, x_max_expanded, n_grid),
        np.linspace(y_min_expanded, y_max_expanded, n_grid)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict marker identity for all grid points
    predicted_markers_grid = knn.predict(grid_points)
    predicted_markers_grid = predicted_markers_grid.reshape(xx.shape)
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Pure extrapolation without overlay
    ax1 = axes[0]
    im1 = ax1.imshow(predicted_markers_grid, origin='lower',
                     extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                     cmap=marker_colormap, interpolation='bilinear',
                     vmin=all_unique_markers[0], vmax=all_unique_markers[-1],
                     alpha=1.0)
    
    ax1.set_xlabel('Latent Dimension 1', fontsize=13)
    ax1.set_ylabel('Latent Dimension 2', fontsize=13)
    ax1.set_title('Extrapolated Phase Diagram\n(KNN Prediction)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim([x_min_expanded, x_max_expanded])
    ax1.set_ylim([y_min_expanded, y_max_expanded])
    ax1.set_aspect('equal')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Predicted Cell Fate', fontsize=11)
    cbar1.set_ticks(all_unique_markers)
    if has_valid_markers_pred:
        cbar1.set_ticklabels([MARKER_NAMES[int(m)] if int(m) < len(MARKER_NAMES) else f'Fate {int(m)}' 
                             for m in all_unique_markers])
    
    # Panel 2: Extrapolation with true marker overlay
    ax2 = axes[1]
    im2 = ax2.imshow(predicted_markers_grid, origin='lower',
                     extent=[x_min_expanded, x_max_expanded, y_min_expanded, y_max_expanded],
                     cmap=marker_colormap, interpolation='bilinear',
                     vmin=all_unique_markers[0], vmax=all_unique_markers[-1],
                     alpha=0.3)
    
    # Overlay actual data points colored by TRUE cell fate (not predicted)
    if has_valid_markers:
        unique_markers = np.unique(marker_labels)
        unique_markers = unique_markers[~np.isnan(unique_markers)]
        
        for idx, marker_val in enumerate(unique_markers):
            mask = marker_labels == marker_val
            if np.sum(mask) > 0:
                ax2.scatter(latent_codes[mask, 0], latent_codes[mask, 1],
                           c=[marker_colors[idx % 10]], 
                           marker=marker_styles[idx % len(marker_styles)],
                           s=8, alpha=0.8, 
                           label=MARKER_NAMES[int(marker_val)],
                           edgecolors='black', linewidth=0.2)
    
    ax2.set_xlabel('Latent Dimension 1', fontsize=13)
    ax2.set_ylabel('Latent Dimension 2', fontsize=13)
    ax2.set_title('Extrapolation + True Cell Fates\n(Ground Truth Overlay)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim([x_min_expanded, x_max_expanded])
    ax2.set_ylim([y_min_expanded, y_max_expanded])
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    
    plt.suptitle('Extrapolated Cell Fate Phase Diagram', 
                 fontsize=16, fontweight='bold')
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

print(f"\nTraining Strategy: {TRAINING_STRATEGY}")
if TRAINING_STRATEGY == 'single_condition':
    print(f"Training Condition: {TRAINING_CONDITION}")

for cond in conditions:
    n_samples = np.sum(condition_labels == cond)
    print(f"{cond}: {n_samples} samples ({n_samples/len(condition_labels)*100:.1f}%)")

print(f"\nTotal samples: {len(condition_labels)}")
print(f"Unique cell fates: {n_markers if has_valid_markers else 'N/A (all NaN)'}")
if has_valid_markers:
    print(f"Cell fate names: {[MARKER_NAMES[int(m)] for m in unique_markers]}")
else:
    print(f"Cell fate names: Not available (markers are NaN)")

print("\n✓ ANALYSIS COMPLETE")
print(f"Results saved to: {OUTPUT_DIR}\n")