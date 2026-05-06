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

mode_plot_signals = 1
mode_plot_predict_patterns = 1
mode_plot_fates = 1
mode_accuracy = 1
mode_plot_LS = 1
mode_plot_predict_eval = 1

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
    
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_LS_2x10_2_2x10'

subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/fig2_LS'
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

f = open(subdirectory_plot_data + '/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test, target_predict_train, target_predict,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
f.close()

subdirectory_plot_fig = directory + '/fig2'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)
    

subdirectory_plot_fig = directory + '/fig2'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)
    
    
margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 1-0.2
hspace = 0.001
wspace = 0.001

lim = 360
ms = 0.15


min_val = 0
max_val = 3

N_fates = 6
 
thresh = 1
markers_predict = fns_plot.return_fates(target_predict[:,:],data.gene_names,thresh)

if mode_z:
    feat_train_z,mean_feat_train,stdev_feat_train = fns_plot.do_zscore(feat_train)
    feat_test_z,_,_ = fns_plot.do_zscore(feat_test,mean_feat_train,stdev_feat_train)
    activations = fns_plot.get_activations(reg,feat_test_z)
else:
    activations = fns_plot.get_activations(reg,feat_test)
    

LS = activations[bottleneck_layer]
bottleneck_dim = LS.shape[1]

fig_size = [10,1.5]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
H,W = 1,data.N_signals
plt.figure()
chrt=0
for ks in range(0,data.N_signals):
    chrt+=1
    plt.subplot(H,W,chrt)
    plt.title(data.signal_names[ks],fontsize=10)
    
    order = np.argsort(feat_test[:,ks])
    plt.scatter(LS[order,0],LS[order,1],s=1,c=feat_test[order,ks],cmap='YlGnBu',vmin=min_val,vmax=max_val)
    
    plt.yticks([])
    plt.xticks([])

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'LS_2d_signals_' + mode_reg + kernel + '_' + cond + file_suffix)


fig_size = [4,3]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

for mode_sim in [0,1]:
    plt.figure()
    
    for m in range(N_fates):
        if mode_sim:
            indices = markers_predict==m
        else:
            indices = markers_test==m
        plt.scatter(LS[indices,0],LS[indices,1],color=fns_plot.return_colmaps('fates')[m],s=3)
    
    plt.xlabel('bottleneck feature 1')
    plt.ylabel('bottleneck feature 2')

    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'LS_2d_marker_sim' + str(mode_sim) + '_' + mode_reg + kernel + '_' + cond + file_suffix)


fig_size = [12,2.2]   
params = {
  'figure.figsize': fig_size,
  }
plt.rcParams.update(params)
H,W = 1,N_fates

for mode_sim in [0,1]:
    plt.figure()
    chrt=0
    for m in range(N_fates):
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.title(data.fate_names[m])
        if mode_sim:
            indices = markers_predict==m
        else:
            indices = markers_test==m
        plt.scatter(LS[:,0],LS[:,1],color='lightgrey',s=3)
        plt.scatter(LS[indices,0],LS[indices,1],color=fns_plot.return_colmaps('fates')[m],s=3)
        
        
        plt.tight_layout()
        plt.savefig(subdirectory_plot_fig + "/" + 'LS_2d_marker_individ_sim' + str(mode_sim) + '_colony_' + mode_reg + kernel + '_' + cond + file_suffix)


