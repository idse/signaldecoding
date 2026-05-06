#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:22:52 2023

@author: D.Brueckner

VIB regression analysis with colony-based train/test split
Uses VIB from fns_NN.py instead of sklearn MLP
"""

import torch
import numpy as np
import os
import dill as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import fns_plotting_scripts as fns_plot
import fns_NN

# ============= CONFIGURATION =============
mode_single = 1   # Single train/test split
mode_colony = 1   # Process individual colonies
mode_av = 1       # Average over multiple runs
N_run = 10        # Number of runs for averaging

train_size = 3    # Number of colonies for training

# Experiment selection
mode_expt = '20'
if mode_expt == '20':
    directory = 'data_expt_20_scaled_norm_bgsub' 
    conditions = ['B50']
elif mode_expt == 'X7':
    directory = 'data_expt_X7'
    conditions = ['B50', 'B50T30']
elif mode_expt == 'X10':
    directory = 'data_expt_X10'
    conditions = ['B50', 'B50_MEKi']

print(f"Directory: {directory}")

subdirectory_data = directory + '/data'

# VIB model parameters (for signal input)
LATENT_DIM = 2
HIDDEN_DIM = 128
N_LAYERS = 2
EPOCHS = 800
LEARNING_RATE = 1e-3
BETA = 0.01

# Output directory
subdirectory_plot = directory + '/analysis_regression_sg_multi_vib'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)

print(f"Output directory: {subdirectory_plot}")

# ============= PROCESS CONDITIONS =============

for cond in conditions:
    print(f"\n{'='*60}")
    print(f"Processing condition: {cond}")
    print('='*60)
    
    # Load data
    f = open(subdirectory_data + '/data_' + cond + ".pickle", 'rb')
    data = pickle.load(f)
    f.close()
    
    # Use signals as input
    feature = data.signals[:, :, :]
    target = data.genes[:, :, :]

    # Ensure 3D arrays
    if len(feature.shape) == 2:
        feature = feature[:, :, np.newaxis]
    if len(target.shape) == 2:
        target = target[:, :, np.newaxis]
    
    N_DIM_INPUT = feature.shape[2]
    N_DIM_OUTPUT = target.shape[2]
    
    print(f"Input dimensions: {N_DIM_INPUT}")
    print(f"Output dimensions: {N_DIM_OUTPUT}")
    
    # ============= SINGLE TRAIN/TEST SPLIT =============
    
    if mode_single:
        print("\nRunning single train/test split...")
        
        # Split data by colonies
        (feat_train, feat_test, tar_train, tar_test, 
         metricdist_train, metricdist_test, markers_train, markers_test) = fns_NN.test_train_split_colonies(
            data, feature, target, train_size=train_size
        )
        
        print(f"Train samples: {feat_train.shape[0]}")
        print(f"Test samples: {feat_test.shape[0]}")
        
        # Standardize using StandardScaler
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        feat_train_z = scaler_X.fit_transform(feat_train)
        tar_train_z = scaler_Y.fit_transform(tar_train)
        feat_test_z = scaler_X.transform(feat_test)
        tar_test_z = scaler_Y.transform(tar_test)
        
        # Convert to torch tensors
        X_train = torch.FloatTensor(feat_train_z)
        Y_train = torch.FloatTensor(tar_train_z)
        X_test = torch.FloatTensor(feat_test_z)
        Y_test = torch.FloatTensor(tar_test_z)
        
        # Create and train VIB model
        vib = fns_NN.FlexibleVIB(
            input_dim=N_DIM_INPUT,
            output_dim=N_DIM_OUTPUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LAYERS,
            encoder_type='nonlinear',
            decoder_type='nonlinear'
        )
        
        print("Training VIB model...")
        _ = fns_NN.train_model(
            vib, X_train, Y_train,
            is_vae=False,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            beta=BETA,
            verbose=True,
            print_every=200
        )
        
        # Get predictions
        vib.eval()
        with torch.no_grad():
            target_predict_train_z = vib(X_train)[0].numpy()
            target_predict_z = vib(X_test)[0].numpy()
        
        # Inverse transform to original scale
        target_predict_train = scaler_Y.inverse_transform(target_predict_train_z)
        target_predict = scaler_Y.inverse_transform(target_predict_z)
        
        print(f"Train loss: {fns_NN.evaluate_model(vib, X_train, Y_train, is_vae=False):.4f}")
        print(f"Test loss: {fns_NN.evaluate_model(vib, X_test, Y_test, is_vae=False):.4f}")
        
        # Save results (keep same structure as original)
        f = open(subdirectory_plot_data + '/data_regression_VIB_' + cond + ".pickle", 'wb')
        pickle.dump((vib, feat_train, feat_test, tar_train, tar_test, 
                    metricdist_train, metricdist_test, markers_train, markers_test, 
                    target_predict_train, target_predict, 
                    scaler_X, scaler_Y), f)
        f.close()
        print(f"✓ Saved single split results")
    
    # ============= PROCESS INDIVIDUAL COLONIES =============
    
    if mode_colony:
        print("\nProcessing individual colonies...")
        
        for it in range(data.N_sys):
            indices = ~np.isnan(data.X[it, :, 0].ravel())
            xx_colony = data.X[it, indices, 0]
            yy_colony = data.X[it, indices, 1]
            rr_colony = data.metricdist[it, indices]
            feature_colony = feature[it, indices, :]
            target_colony = target[it, indices, :]
            markers_colony = data.markers[it, indices]
            
            # Standardize using training scalers
            feature_colony_z = scaler_X.transform(feature_colony)
            
            # Convert to torch
            X_colony = torch.FloatTensor(feature_colony_z)
            
            # Predict
            vib.eval()
            with torch.no_grad():
                target_predict_colony_z = vib(X_colony)[0].numpy()
            
            # Inverse transform
            target_predict_colony = scaler_Y.inverse_transform(target_predict_colony_z)
            
            f = open(subdirectory_plot_data + '/data_colony_it' + str(it) + '_VIB_' + cond + ".pickle", 'wb')
            pickle.dump((xx_colony, yy_colony, rr_colony, markers_colony, 
                        target_colony, target_predict_colony), f)
            f.close()
        
        print(f"✓ Saved {data.N_sys} colony results")
    
    # ============= AVERAGE OVER MULTIPLE RUNS =============
    
    if mode_av:
        print(f"\nAveraging over {N_run} runs...")
        
        thresh = 1
        ratios = np.zeros((N_run, data.N_genes, 5))
        MIs = np.zeros((N_run, data.N_genes))
        
        # Split data once (same for all runs, but different random initializations)
        (feat_train, feat_test, tar_train, tar_test, 
         metricdist_train, metricdist_test, markers_train, markers_test) = fns_NN.test_train_split_colonies(
            data, feature, target, train_size=train_size
        )
        
        for it in range(N_run):
            if np.mod(it, 10) == 0:
                print(f"  Run {it}/{N_run}")
            
            # Standardize
            scaler_X_run = StandardScaler()
            scaler_Y_run = StandardScaler()
            
            feat_train_z = scaler_X_run.fit_transform(feat_train)
            tar_train_z = scaler_Y_run.fit_transform(tar_train)
            feat_test_z = scaler_X_run.transform(feat_test)
            
            # Convert to torch
            X_train_run = torch.FloatTensor(feat_train_z)
            Y_train_run = torch.FloatTensor(tar_train_z)
            X_test_run = torch.FloatTensor(feat_test_z)
            
            # Create new VIB model (fresh random initialization each run)
            vib_run = fns_NN.FlexibleVIB(
                input_dim=N_DIM_INPUT,
                output_dim=N_DIM_OUTPUT,
                latent_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
                n_layers=N_LAYERS,
                encoder_type='nonlinear',
                decoder_type='nonlinear'
            )
            
            # Train
            _ = fns_NN.train_model(
                vib_run, X_train_run, Y_train_run,
                is_vae=False,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                beta=BETA,
                verbose=False
            )
            
            # Predict
            vib_run.eval()
            with torch.no_grad():
                target_predict_z = vib_run(X_test_run)[0].numpy()
            
            # Inverse transform
            target_predict = scaler_Y_run.inverse_transform(target_predict_z)
            
            # Save model for this run
            f = open(subdirectory_plot_data + '/data_regression_VIB_' + cond + '_run' + str(it) + ".pickle", 'wb')
            pickle.dump((vib_run, scaler_X_run, scaler_Y_run), f)
            f.close()
            
            # Calculate metrics for each gene
            for kg in range(data.N_genes):
                pos_expt = tar_test[:, kg] >= thresh
                pos_sim = target_predict[:, kg] >= thresh
                
                conf_matrix_vals = confusion_matrix(pos_expt, pos_sim).ravel()
                
                if len(conf_matrix_vals) > 1:
                    TN, FP, FN, TP = conf_matrix_vals
                    
                    # Specificity (True Negative Rate)
                    ratios[it, kg, 0] = fns_plot.diff_zero(TN, (TN + FP))
                    # Sensitivity (True Positive Rate / Recall)
                    ratios[it, kg, 1] = fns_plot.diff_zero(TP, (TP + FN))
                    
                    # Negative Predictive Value
                    ratios[it, kg, 2] = fns_plot.diff_zero(TN, (TN + FN))
                    # Precision (Positive Predictive Value)
                    ratios[it, kg, 3] = fns_plot.diff_zero(TP, (TP + FP))
                    
                    # Accuracy
                    ratios[it, kg, 4] = fns_plot.diff_zero((TP + TN), (TN + FP + FN + TP))
                
                # Mutual Information
                MIs[it, kg] = fns_plot.calc_MI_sklearn(tar_test[:, kg], target_predict[:, kg])
        
        # Save averaged results
        f = open(subdirectory_plot_data + '/data_regression_av_VIB_' + cond + ".pickle", 'wb')
        pickle.dump((ratios), f)
        f.close()
        
        f = open(subdirectory_plot_data + '/data_regression_MI_av_VIB_' + cond + ".pickle", 'wb')
        pickle.dump((MIs), f)
        f.close()
        
        print(f"✓ Saved averaged results over {N_run} runs")
        
        # Print summary statistics
        print("\nSummary statistics (mean ± std across runs):")
        print(f"  Specificity: {np.mean(ratios[:, :, 0]):.3f} ± {np.std(ratios[:, :, 0]):.3f}")
        print(f"  Sensitivity: {np.mean(ratios[:, :, 1]):.3f} ± {np.std(ratios[:, :, 1]):.3f}")
        print(f"  Precision:   {np.mean(ratios[:, :, 3]):.3f} ± {np.std(ratios[:, :, 3]):.3f}")
        print(f"  Accuracy:    {np.mean(ratios[:, :, 4]):.3f} ± {np.std(ratios[:, :, 4]):.3f}")
        print(f"  MI:          {np.mean(MIs):.3f} ± {np.std(MIs):.3f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)