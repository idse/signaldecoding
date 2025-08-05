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

mode_z = 1
mode_input = 's'
mode_classbal = 0


#standard
if mode_classbal:
    suffix_classbal = '_classbal'
else:
    suffix_classbal = ''
    
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_LS_2x10_1_2x10'
subdirectory_plot_data = subdirectory_plot + '/data'
subdirectory_plot_data_reg = subdirectory_plot + '/data_many_reg'


subdirectory_plot_fig = subdirectory_plot + '/LS_test_1d'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)


thresh = 1

mode_reg = 'MLP'
kernel = 'relu'
bottleneck_layer = 3

import fns_plotting_scripts as fns_plot
import matplotlib.pyplot as plt
file_suffix = '.png'
file_suffix_pdf = '.pdf'
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

if mode_input == 's':
    feature = data.signals
elif mode_input == 'r':
    feature = data.metricdist
elif mode_input == 'n':
    feature = data.nuc_feat

target = data.genes[:,:,:]

train_size = 3

if mode_classbal==1:
    feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_classbal(data,feature,target,train_size=train_size)
elif mode_classbal==2:
    feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_classbal_resample(data,feature,target,train_size=train_size)
else:
    feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_v2(data,feature,target,train_size=train_size)


f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios,ratios_conf,accuracy,corrs,variances) = pickle.load(f)
f.close()

plt.figure()

N_run = 20
for it in range(N_run):
    f = open(subdirectory_plot_data_reg + '/data_regression_' + mode_reg + kernel + '_' + cond + '_run' + str(it) + ".pickle",'rb')
    (reg,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
    f.close()
    
    if mode_z:
        feat_train_z,mean_feat_train,stdev_feat_train = fns_plot.do_zscore(feat_train)
        tar_train_z,mean_tar_train,stdev_tar_train = fns_plot.do_zscore(tar_train)
        feat_test_z,_,_ = fns_plot.do_zscore(feat_test,mean_feat_train,stdev_feat_train)
        
        target_predict = fns_plot.undo_zscore(reg.predict(feat_test_z),mean_tar_train,stdev_tar_train)
    else:
        target_predict = reg.predict(feat_test)

     
    thresh = 1
    #markers_predict = fns_plot.return_fates(target_predict[:,:],data.gene_names,thresh)
    
    activations = fns_plot.get_activations(reg,feat_test_z)
    LS = activations[bottleneck_layer]
    bottleneck_dim = LS.shape[1]
    
    
    fig_size = [10,10]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    H,W = 5,5
    
    x_max = np.max(LS)
    
    
    plt.figure()
    chrt=0
    
    plt.suptitle('mean precision = ' + str(np.round(np.nanmean(ratios_conf[it,:,1]),3)))
    
    chrt+=1
    plt.subplot(H,W,chrt)
    plt.title('radius',fontsize=10)
    plt.scatter(LS,metricdist_test,s=5)
    
    plt.xlim([0,x_max])
    
    for ks in range(0,data.N_signals):
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.title(data.signal_names[ks],fontsize=10)
        
        plt.scatter(LS,feat_test[:,ks],s=5)
        
        plt.yticks([])
        plt.xticks([])
        
        plt.xlim([0,x_max])

    plt.tight_layout()
    
    for kg in range(0,data.N_genes):
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.title(data.gene_names[kg],fontsize=10)
        
        plt.scatter(LS,tar_test[:,kg],s=5)
        plt.plot([0,x_max],[1,1],'--k')
        
        plt.yticks([])
        plt.xticks([])
        
        plt.xlim([0,x_max])

    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'LS_1d_' + mode_reg + kernel + '_' + cond + '_run' + str(it) + file_suffix)
    
    
    
    #pos_expt = tar_test[:,kg]>=thresh
    #pos_sim = target_predict[:,kg]>=thresh
    
    """
    plt.figure()
    chrt=0
    
    chrt+=1
    plt.subplot(H,W,chrt)
    plt.title('radius',fontsize=10)
    plt.scatter(LS,metricdist_test,s=5)
    
    plt.xlim([0,x_max])
    
    for ks in range(0,data.N_signals):
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.title(data.signal_names[ks],fontsize=10)
        
        plt.scatter(LS,feat_test[:,ks],s=5)
        
        plt.yticks([])
        plt.xticks([])
        
        plt.xlim([0,x_max])

    plt.tight_layout()
    
    for kg in range(0,data.N_genes):
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.title(data.gene_names[kg],fontsize=10)
        
        plt.scatter(LS,tar_test[:,kg],s=5)
        plt.plot([0,x_max],[1,1],'--k')
        
        plt.yticks([])
        plt.xticks([])
        
        plt.xlim([0,x_max])

    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'LS_1d_' + mode_reg + kernel + '_' + cond + file_suffix)
    """
    
    
    
    
    
