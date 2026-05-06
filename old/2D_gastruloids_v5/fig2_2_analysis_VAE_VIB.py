#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE vs VIB Comparison Analysis
Compare reconstruction losses across different latent dimensions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pickle
import dill
import fns_NN

# ============= ANALYSIS OPTIONS =============
RUN_ANALYSIS = False  # Set to False to skip training and only plot existing results

# ============= CONFIGURATION =============
EPOCHS = 800
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
BETA = 0.01
TRAIN_SIZE = 3  # Number of colonies to use for training
N_RUNS = 10  # Number of training runs to average over

ENCODER_TYPE = 'nonlinear'
DECODER_TYPE = 'nonlinear'

# ============= LOAD EXPERIMENTAL DATA =============
directory = 'data_expt_20_scaled_norm_bgsub'
cond = 'B50'
subdirectory_data = directory + '/data'
OUTPUT_DIR = directory + '/fig2_2_VAE_vs_VIB'

print("Loading experimental data...")
f = open(subdirectory_data + '/data_' + cond + ".pickle", 'rb')
data = dill.load(f)
f.close()

feature = data.signals
target = data.genes

# Split into train/test by colonies
print(f"\nSplitting data: {TRAIN_SIZE} colonies for training, {feature.shape[0] - TRAIN_SIZE} for testing")
(feat_train, feat_test, tar_train, tar_test, 
 metricdist_train, metricdist_test, markers_train, markers_test) = fns_NN.test_train_split_colonies(
    data, feature, target, train_size=TRAIN_SIZE
)

print(f"Training set: {feat_train.shape[0]} cells")
print(f"Test set: {feat_test.shape[0]} cells")

N_DIM_INPUT = feat_train.shape[1]
N_DIM_OUTPUT = tar_train.shape[1]

print(f"Input dimensions: {N_DIM_INPUT}")
print(f"Output dimensions: {N_DIM_OUTPUT}")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
data_dir = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)

# ============= DATA PREPROCESSING =============
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit scalers on training data only
X_train_scaled = scaler_X.fit_transform(feat_train)
Y_train_scaled = scaler_Y.fit_transform(tar_train)

# Transform test data using training scalers
X_test_scaled = scaler_X.transform(feat_test)
Y_test_scaled = scaler_Y.transform(tar_test)

# Convert to torch tensors
X_train = torch.FloatTensor(X_train_scaled)
Y_train = torch.FloatTensor(Y_train_scaled)
X_test = torch.FloatTensor(X_test_scaled)
Y_test = torch.FloatTensor(Y_test_scaled)

print(f"\nTrain - Input X: {X_train.shape}, Output Y: {Y_train.shape}")
print(f"Test - Input X: {X_test.shape}, Output Y: {Y_test.shape}")

# ============= RUN ANALYSIS =============
max_latent_dim = min(N_DIM_INPUT, 7)
results_file = os.path.join(data_dir, 'vae_vib_results.pkl')

if RUN_ANALYSIS:
    print("\n" + "="*60)
    print(f"RUNNING VAE vs VIB ANALYSIS ({N_RUNS} runs per model)")
    print("="*60)
    
    # Initialize storage for all runs
    vib_train_losses_all_runs = []
    vib_test_losses_all_runs = []
    vae_train_losses_all_runs = []
    vae_test_losses_all_runs = []
    
    # Run multiple training runs
    for run in range(N_RUNS):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{N_RUNS}")
        print('='*60)
        
        # Train VIB models
        print("\nTraining VIB models...")
        vib_train_losses = []
        vib_test_losses = []
        
        for latent_dim in range(1, max_latent_dim + 1):
            print(f"\nVIB with latent_dim={latent_dim}")
            vib = fns_NN.FlexibleVIB(
                input_dim=N_DIM_INPUT, 
                output_dim=N_DIM_OUTPUT, 
                latent_dim=latent_dim, 
                hidden_dim=HIDDEN_DIM, 
                n_layers=2,
                encoder_type=ENCODER_TYPE, 
                decoder_type=DECODER_TYPE
            )
            _ = fns_NN.train_model(
                vib, X_train, Y_train, 
                is_vae=False, 
                epochs=EPOCHS, 
                lr=LEARNING_RATE, 
                beta=BETA,
                verbose=(run == 0)  # Only verbose for first run
            )
            
            train_loss = fns_NN.evaluate_model(vib, X_train, Y_train, is_vae=False)
            test_loss = fns_NN.evaluate_model(vib, X_test, Y_test, is_vae=False)
            
            vib_train_losses.append(train_loss)
            vib_test_losses.append(test_loss)
            
            if run == 0:
                print(f"  Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}")
        
        vib_train_losses_all_runs.append(vib_train_losses)
        vib_test_losses_all_runs.append(vib_test_losses)
        
        # Train VAE models
        print("\n\nTraining VAE models...")
        vae_train_losses = []
        vae_test_losses = []
        
        for latent_dim in range(1, max_latent_dim + 1):
            print(f"\nVAE with latent_dim={latent_dim}")
            vae = fns_NN.FlexibleVAE(
                input_dim=N_DIM_INPUT, 
                latent_dim=latent_dim, 
                hidden_dim=HIDDEN_DIM, 
                n_layers=2,
                encoder_type=ENCODER_TYPE, 
                decoder_type=DECODER_TYPE
            )
            _ = fns_NN.train_model(
                vae, X_train, Y_train, 
                is_vae=True, 
                epochs=EPOCHS, 
                lr=LEARNING_RATE, 
                beta=BETA,
                verbose=(run == 0)  # Only verbose for first run
            )
            
            train_loss = fns_NN.evaluate_model(vae, X_train, is_vae=True)
            test_loss = fns_NN.evaluate_model(vae, X_test, is_vae=True)
            
            vae_train_losses.append(train_loss)
            vae_test_losses.append(test_loss)
            
            if run == 0:
                print(f"  Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}")
        
        vae_train_losses_all_runs.append(vae_train_losses)
        vae_test_losses_all_runs.append(vae_test_losses)
    
    # Convert to numpy arrays and compute statistics
    vib_train_losses_all_runs = np.array(vib_train_losses_all_runs)  # Shape: (N_RUNS, max_latent_dim)
    vib_test_losses_all_runs = np.array(vib_test_losses_all_runs)
    vae_train_losses_all_runs = np.array(vae_train_losses_all_runs)
    vae_test_losses_all_runs = np.array(vae_test_losses_all_runs)
    
    # Compute mean and std
    vae_train_losses_mean = np.mean(vae_train_losses_all_runs, axis=0).tolist()
    vae_train_losses_std = np.std(vae_train_losses_all_runs, axis=0).tolist()
    vae_test_losses_mean = np.mean(vae_test_losses_all_runs, axis=0).tolist()
    vae_test_losses_std = np.std(vae_test_losses_all_runs, axis=0).tolist()
    
    vib_train_losses_mean = np.mean(vib_train_losses_all_runs, axis=0).tolist()
    vib_train_losses_std = np.std(vib_train_losses_all_runs, axis=0).tolist()
    vib_test_losses_mean = np.mean(vib_test_losses_all_runs, axis=0).tolist()
    vib_test_losses_std = np.std(vib_test_losses_all_runs, axis=0).tolist()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"VAE Test Losses (mean ± std):")
    for i, (m, s) in enumerate(zip(vae_test_losses_mean, vae_test_losses_std)):
        print(f"  Latent dim {i+1}: {m:.6f} ± {s:.6f}")
    print(f"\nVIB Test Losses (mean ± std):")
    for i, (m, s) in enumerate(zip(vib_test_losses_mean, vib_test_losses_std)):
        print(f"  Latent dim {i+1}: {m:.6f} ± {s:.6f}")
    
    # Save results
    results = {
        'vae_train_losses_mean': vae_train_losses_mean,
        'vae_train_losses_std': vae_train_losses_std,
        'vae_test_losses_mean': vae_test_losses_mean,
        'vae_test_losses_std': vae_test_losses_std,
        'vib_train_losses_mean': vib_train_losses_mean,
        'vib_train_losses_std': vib_train_losses_std,
        'vib_test_losses_mean': vib_test_losses_mean,
        'vib_test_losses_std': vib_test_losses_std,
        'vae_train_losses_all_runs': vae_train_losses_all_runs.tolist(),
        'vae_test_losses_all_runs': vae_test_losses_all_runs.tolist(),
        'vib_train_losses_all_runs': vib_train_losses_all_runs.tolist(),
        'vib_test_losses_all_runs': vib_test_losses_all_runs.tolist(),
        'latent_dims': list(range(1, max_latent_dim + 1)),
        'n_dim_input': N_DIM_INPUT,
        'n_dim_output': N_DIM_OUTPUT,
        'train_size': TRAIN_SIZE,
        'n_train_cells': X_train.shape[0],
        'n_test_cells': X_test.shape[0],
        'n_runs': N_RUNS,
        'config': {
            'epochs': EPOCHS,
            'lr': LEARNING_RATE,
            'hidden_dim': HIDDEN_DIM,
            'beta': BETA,
            'encoder_type': ENCODER_TYPE,
            'decoder_type': DECODER_TYPE
        }
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to: {results_file}")

else:
    print("\n" + "="*60)
    print("LOADING EXISTING RESULTS")
    print("="*60)
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}\nSet RUN_ANALYSIS=True to generate results.")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    vae_train_losses_mean = results['vae_train_losses_mean']
    vae_train_losses_std = results['vae_train_losses_std']
    vae_test_losses_mean = results['vae_test_losses_mean']
    vae_test_losses_std = results['vae_test_losses_std']
    vib_train_losses_mean = results['vib_train_losses_mean']
    vib_train_losses_std = results['vib_train_losses_std']
    vib_test_losses_mean = results['vib_test_losses_mean']
    vib_test_losses_std = results['vib_test_losses_std']
    latent_dims_list = results['latent_dims']
    N_DIM_INPUT = results['n_dim_input']
    N_DIM_OUTPUT = results['n_dim_output']
    N_RUNS = results['n_runs']
    
    print(f"✓ Loaded results from: {results_file}")
    print(f"  Input dim: {N_DIM_INPUT}, Output dim: {N_DIM_OUTPUT}")
    print(f"  Latent dims tested: {latent_dims_list}")
    print(f"  Training cells: {results['n_train_cells']}, Test cells: {results['n_test_cells']}")
    print(f"  Number of runs averaged: {N_RUNS}")

# ============= NORMALIZE LOSSES PER DIMENSION =============
# Normalize by number of output dimensions for fair comparison
vae_train_losses_norm = [loss / N_DIM_INPUT for loss in vae_train_losses_mean]
vae_train_losses_norm_std = [std / N_DIM_INPUT for std in vae_train_losses_std]
vae_test_losses_norm = [loss / N_DIM_INPUT for loss in vae_test_losses_mean]
vae_test_losses_norm_std = [std / N_DIM_INPUT for std in vae_test_losses_std]

vib_train_losses_norm = [loss / N_DIM_OUTPUT for loss in vib_train_losses_mean]
vib_train_losses_norm_std = [std / N_DIM_OUTPUT for std in vib_train_losses_std]
vib_test_losses_norm = [loss / N_DIM_OUTPUT for loss in vib_test_losses_mean]
vib_test_losses_norm_std = [std / N_DIM_OUTPUT for std in vib_test_losses_std]

print(f"\nNormalized losses (per dimension):")
print(f"  VAE losses divided by {N_DIM_INPUT} (input dimensions)")
print(f"  VIB losses divided by {N_DIM_OUTPUT} (output dimensions)")

# ============= GENERATE COMPARISON PLOTS =============
print("\n\nGenerating VAE vs VIB comparison plots...")

latent_dims_list = results['latent_dims'] if not RUN_ANALYSIS else list(range(1, max_latent_dim + 1))

# Convert to numpy arrays for plotting
vae_train_losses_norm = np.array(vae_train_losses_norm)
vae_train_losses_norm_std = np.array(vae_train_losses_norm_std)
vae_test_losses_norm = np.array(vae_test_losses_norm)
vae_test_losses_norm_std = np.array(vae_test_losses_norm_std)
vib_train_losses_norm = np.array(vib_train_losses_norm)
vib_train_losses_norm_std = np.array(vib_train_losses_norm_std)
vib_test_losses_norm = np.array(vib_test_losses_norm)
vib_test_losses_norm_std = np.array(vib_test_losses_norm_std)

# Plot 1: Train and Test losses separately (normalized with error bars)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Training losses
ax = axes[0]
ax.errorbar(latent_dims_list, vae_train_losses_norm, yerr=vae_train_losses_norm_std,
           fmt='bs-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
           label=f'VAE: X({N_DIM_INPUT}D) → Z → X({N_DIM_INPUT}D)')
ax.errorbar(latent_dims_list, vib_train_losses_norm, yerr=vib_train_losses_norm_std,
           fmt='ro-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
           label=f'VIB: X({N_DIM_INPUT}D) → Z → Y({N_DIM_OUTPUT}D)')
ax.set_xlabel('Latent Dimension', fontsize=13)
ax.set_ylabel('MSE per Dimension', fontsize=13)
ax.set_title(f'Training Set Performance (N={N_RUNS} runs)', fontsize=15, fontweight='bold')
ax.set_xticks(latent_dims_list)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

# Test losses
ax = axes[1]
ax.errorbar(latent_dims_list, vae_test_losses_norm, yerr=vae_test_losses_norm_std,
           fmt='bs-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
           label=f'VAE: X({N_DIM_INPUT}D) → Z → X({N_DIM_INPUT}D)')
ax.errorbar(latent_dims_list, vib_test_losses_norm, yerr=vib_test_losses_norm_std,
           fmt='ro-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
           label=f'VIB: X({N_DIM_INPUT}D) → Z → Y({N_DIM_OUTPUT}D)')
ax.set_xlabel('Latent Dimension', fontsize=13)
ax.set_ylabel('MSE per Dimension', fontsize=13)
ax.set_title(f'Test Set Performance (N={N_RUNS} runs)', fontsize=15, fontweight='bold')
ax.set_xticks(latent_dims_list)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, '07a_vae_vib_train_test_comparison.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")
plt.show()

# Plot 2: Combined plot with interpretation (normalized with error bars)
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

ax.errorbar(latent_dims_list, vae_test_losses_norm, yerr=vae_test_losses_norm_std,
           fmt='bs-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
           label=f'VAE (Test): X({N_DIM_INPUT}D) → Z → X({N_DIM_INPUT}D)')
ax.errorbar(latent_dims_list, vib_test_losses_norm, yerr=vib_test_losses_norm_std,
           fmt='ro-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
           label=f'VIB (Test): X({N_DIM_INPUT}D) → Z → Y({N_DIM_OUTPUT}D)')

ax.set_xlabel('Latent Dimension', fontsize=13)
ax.set_ylabel('MSE per Dimension', fontsize=13)
ax.set_title(f'Reconstruction Loss Comparison: VAE vs VIB (Test Set, N={N_RUNS} runs)', 
            fontsize=15, fontweight='bold')
ax.set_xticks(latent_dims_list)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')

interpretation_text = (
    "Interpretation:\n"
    "• VIB elbow → Intrinsic dimensionality of X→Y mapping\n"
    "• VAE elbow → Intrinsic dimensionality of X data (including noise)\n"
    "• VIB elbow < VAE elbow → Inputs contain predictively irrelevant variation\n"
    f"Note: Losses normalized per dimension (VAE÷{N_DIM_INPUT}, VIB÷{N_DIM_OUTPUT})\n"
    f"Error bars show ±1 std over {N_RUNS} training runs"
)
ax.text(0.98, 0.97, interpretation_text, transform=ax.transAxes, 
       fontsize=10, verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, '07b_vae_vib_comparison.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")
plt.show()

# Plot 3: Polished version for publication (baseline-subtracted and normalized)
# For polished plot: subtract baseline (error at 7D) and normalize by error at 1D
# Use mean values only (no error bars for polished version)

# Baseline is the last value (latent_dim = 7)
vae_baseline = vae_test_losses_norm[-1]
vib_baseline = vib_test_losses_norm[-1]

# Subtract baseline
vae_relative = vae_test_losses_norm - vae_baseline
vib_relative = vib_test_losses_norm - vib_baseline

# Normalize by error at 1D (first value after baseline subtraction)
vae_normalized_polished = vae_relative / vae_relative[0]
vib_normalized_polished = vib_relative / vib_relative[0]

fig, ax = plt.subplots(1, 1, figsize=(3, 3))

ax.plot(latent_dims_list, vae_normalized_polished, 'bs-', linewidth=2, markersize=6,
       label='signal compression')
ax.plot(latent_dims_list, vib_normalized_polished, 'ro-', linewidth=2, markersize=6, 
       label='signal-to-fate compression')

ax.set_xlabel('Latent Dimension')
ax.set_ylabel('Normalized Error')

# Remove all grid lines
ax.grid(False)

# Remove tick marks but keep labels
ax.tick_params(axis='both', which='both', length=0)

# Set x-axis to show only the latent dimensions tested
ax.set_xticks(latent_dims_list)

# Clean legend - move to top outside plot (further up)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=1)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, '07c_vae_vib_polished.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Polished plot saved to: {output_file}")
plt.show()

print("\nAnalysis complete!")