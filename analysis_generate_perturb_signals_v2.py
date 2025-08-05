#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:22:52 2023

@author: D.Brueckner


split test/train by colonies


"""

import numpy as np
import os
import dill as pickle

import fns_plotting_scripts as fns_plot


mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    cond = 'B50'
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)


subdirectory_data = directory + '/data'
subdirectory_plot = directory + '/analysis_perturb_signals'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
    

import fns_plotting_scripts as fns_plot
import matplotlib.pyplot as plt
file_suffix = '.png'
plt.close('all')
fs=11
fs2=9
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)


f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
data = pickle.load(f)
f.close()

feature = data.signals
target = data.genes

feature_clean,target_clean,metricdist_clean,markers_clean = fns_plot.clean_data_full(data,feature,target)
feature_clean_z,mean_feat_train,stdev_feat_train = fns_plot.do_zscore(feature_clean)

it = 0
indices = ~np.isnan(data.X[it,:,0].ravel())
xx_colony = data.X[it,indices,0]
yy_colony = data.X[it,indices,1]
rr_colony = data.metricdist[it,indices]
feature_colony = data.signals[it,indices,:]
feature_colony_z,_,_ = fns_plot.do_zscore(feature_colony,mean_feat_train,stdev_feat_train)

target_colony = data.genes[it,indices,:]

feature_perturb = np.zeros(feature_clean.shape)
feature_colony_perturb = np.zeros(feature_colony.shape)

N_bins_x = 20
k_nn = 5

mode_knn = 0

for mode_perturb in [0]:
    
    if mode_perturb==0: #yap act
        ks_pert = 1
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(feature_colony[:,ks_pert][np.newaxis,:,np.newaxis],rr_colony[:][np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        
        ind_sig_pert = np.argmax(mean_x)
        sig_max = mean_x[ind_sig_pert]
        sig_min = sig_max - np.sqrt(var_x[ind_sig_pert,0])/4
        
        indices_cond = np.where(feature_colony[:,ks_pert] > sig_min)[0]
        sig_perturb = sig_max
    if mode_perturb==1: #erk inh
        ks_pert = 5
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(feature_colony[:,ks_pert][np.newaxis,:,np.newaxis],rr_colony[:][np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        ind_sig_pert = np.argmin(mean_x)
        sig_min = mean_x[ind_sig_pert]
        sig_max = sig_min + np.sqrt(var_x[ind_sig_pert,0])/2
        
        indices_cond = np.where(feature_colony[:,ks_pert] < sig_max)[0]
        sig_perturb = sig_min
    
    
    feature_colony_cond = feature_colony[indices_cond,:]
    feature_colony_z_cond = feature_colony_z[indices_cond,:]
    
    from scipy.spatial.distance import cdist
    #for distance use z-valued features
    if mode_knn==0:
        distances = cdist(feature_colony_z[:,:],feature_colony_z_cond[:,:])
        suffix = '_' + 'all'
    elif mode_knn==1:
        distances = cdist(feature_colony_z[:,0][:,np.newaxis],feature_colony_z_cond[:,0][:,np.newaxis])
        suffix = '_' + 'smad'
    elif mode_knn==2:
        distances = cdist(feature_colony_z[:,:],feature_colony_z_cond[:,:])
        suffix = '_' + 'allinc'
        
    N_cells = feature_colony_z.shape[0]
    indices_knn_cond_all = []
    for j in range(N_cells):
        indices_knn_cond = np.argsort(distances[j,:])[:k_nn]
        indices_knn_cond_all.append(indices_knn_cond)
        mean_knn_signal = np.mean(feature_colony_cond[indices_knn_cond,:],axis=0)
        feature_colony_perturb[j,:] = mean_knn_signal
    
    #feature_perturb[:,ks_pert] = sig_max
    if not mode_knn==2:
        feature_colony_perturb[:,ks_pert] = sig_max
    
    
    
    lim = 360
    ms = 0.15
    
    min_val = 0
    max_val = 3
    max_val_vmax = 3
    
    H,W=1,data.N_signals
    fig_size = [12,2.5]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    
    j = 100
    
    chrt=0
    for ks in range(0,data.N_signals):
        
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.scatter(rr_colony,feature_colony[:,ks],ms,color='lightgrey')
        plt.scatter(rr_colony[j],feature_colony[j,ks],10,color='b')
        
        plt.scatter(rr_colony[indices_knn_cond_all[j]],feature_colony_cond[indices_knn_cond_all[j],ks],5,color='r')
        
        plt.title(data.signal_names[ks],fontsize=10)
        
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout()
    plt.savefig(subdirectory_plot + "/" + 'signals_pert_scatter_rad_perturb' + str(mode_perturb) + suffix + '_' + cond + file_suffix)
    
    
    
    
    #"""
    
    
    fig = plt.figure()
    chrt=0
    for ks in range(0,data.N_signals):
        
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.scatter(rr_colony,feature_colony[:,ks],ms,color='lightgrey')
        plt.scatter(rr_colony[indices_cond],feature_colony[indices_cond,ks],ms,color='r')
        
        plt.title(data.signal_names[ks],fontsize=10)
        
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout()
    plt.savefig(subdirectory_plot + "/" + 'signals_unpert_mark_scatter_rad_perturb' + str(mode_perturb) + suffix + '_' + cond + file_suffix)
    
    
    fig = plt.figure()
    chrt=0
    for ks in range(0,data.N_signals):
        
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.scatter(rr_colony,feature_colony_perturb[:,ks],ms,color='lightgrey')
        plt.scatter(rr_colony[indices_cond],feature_colony_perturb[indices_cond,ks],ms,color='r')
        
        plt.title(data.signal_names[ks],fontsize=10)
        
        plt.xticks([])
        #plt.yticks([])
        
    plt.tight_layout()
    plt.savefig(subdirectory_plot + "/" + 'signals_pert_scatter_rad_perturb' + str(mode_perturb) + suffix + '_' + cond + file_suffix)
    
    
    
    H,W=2,data.N_signals
    fig_size = [12,3.5]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    chrt=0
    for ks in range(0,data.N_signals):
    
        chrt+=1
        plt.subplot(H,W,chrt)
        order = np.argsort(feature_colony[:,ks])
        max_val_vmax = np.nanpercentile(feature_colony[:,ks],99)
        min_val = np.nanpercentile(feature_colony[:,ks],0.1)
        plt.scatter(xx_colony[order],yy_colony[order],ms,c=feature_colony[order,ks],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
        
        plt.title(data.signal_names[ks],fontsize=10)
        plt.axis('off')
        
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
        
        plt.subplot(H,W,W+chrt)
        order = np.argsort(feature_colony_perturb[:,ks])
        plt.scatter(xx_colony[order],yy_colony[order],ms,c=feature_colony_perturb[order,ks],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
        
        plt.axis('off')
        
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
    
    
    plt.tight_layout()
    plt.savefig(subdirectory_plot + "/" + 'signals_control_perturb' + str(mode_perturb) + suffix + '_' + cond + file_suffix)
        
    
    fig_size = [12,2.5]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    H,W=1,data.N_signals
    fig = plt.figure()
    chrt=0
    N_bins_x = 20
    max_val = 6
    for ks in range(0,data.N_signals):
        chrt+=1
        plt.subplot(H,W,chrt)
    
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(feature_colony[:,ks][np.newaxis,:,np.newaxis],rr_colony[:][np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='b')
        #plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color='b',alpha=0.2)
        
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(feature_colony_perturb[:,ks][np.newaxis,:,np.newaxis],rr_colony[:][np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='r')
        
        #plt.ylim([min_val,max_val_vmax])
        plt.title(data.signal_names[ks],fontsize=10)
        plt.xticks([])
        plt.yticks([])
        
        
    plt.tight_layout()
    plt.savefig(subdirectory_plot + "/" + 'signal_profile_control_perturb' + str(mode_perturb) + suffix + '_' + cond + file_suffix)

    

    mode_z = 1
    mode_input = 's'
    subdirectory_reg = directory + '/analysis_regression_' + mode_input + 'g_multi' + '_z' + str(mode_z) + '_LS_2x10_2_2x10_v3_test'

    subdirectory_reg_data = subdirectory_reg + '/data'
    mode_reg = 'MLP'
    kernel = 'relu'
    f = open(subdirectory_reg_data + '/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test, target_predict_train, target_predict,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
    f.close()
    
    
    
    feature_colony_z,_,_ = fns_plot.do_zscore(feature_colony_perturb,mean_feat_train,stdev_feat_train)
    target_predict_colony = fns_plot.undo_zscore(reg.predict(feature_colony_z),mean_tar_train,stdev_tar_train)
    
    if mode_perturb == 1:
        indices_genes = [0,4,5,9,11,13,14]
    else:
        indices_genes = [0,4,5,13,9,14]
    N_genes_plot = len(indices_genes)
    H,W=2,N_genes_plot
    fig_size = [12,3.5]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    chrt=0
    for kg in indices_genes:
    
        chrt+=1
        plt.subplot(H,W,chrt)
        order = np.argsort(target_colony[:,kg])
        max_val_vmax = np.nanpercentile(target_colony[:,kg],99)
        min_val = np.nanpercentile(target_colony[:,kg],0.1)
        plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_colony[order,kg],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
        
        plt.title(data.gene_names[kg],fontsize=10)
        plt.axis('off')
        
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
        
        plt.subplot(H,W,W+chrt)
        order = np.argsort(target_predict_colony[:,kg])
        plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_predict_colony[order,kg],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
        
        plt.axis('off')
        
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
    
    
    plt.tight_layout()
    plt.savefig(subdirectory_plot + "/" + 'genes_predict_perturb' + str(mode_perturb) + suffix + '_' + cond + file_suffix)
    #"""
    
    
    
    
