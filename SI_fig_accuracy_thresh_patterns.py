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
    #['B10','B50','B200']
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
    
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_3x10'
subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/SI_fig_accuracy_thresh'
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

it = 0
f = open(subdirectory_plot_data + '/data_colony_it' + str(it) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(xx_colony, yy_colony, rr_colony, markers_colony, target_colony, target_predict_colony) = pickle.load(f)
f.close()


margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 1-0.2
hspace = 0.001
wspace = 0.001

lim = 360
ms = 0.15


min_val = 0
max_val = 9
max_val_vmax = 5

H,W=9,data.N_genes  
fig_size = [15,9]
#[17,9*1.3]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

thresholds = np.ones(data.N_genes)

fig = plt.figure()
chrt=0
fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    order = np.argsort(target_predict_colony[:,kg])
    plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_predict_colony[order,kg],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
    
    plt.title(data.gene_names[kg],fontsize=10)
    plt.axis('off')

    plt.xticks([])
    plt.yticks([])
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])

#g expt
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    order = np.argsort(target_colony[:,kg])
    plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_colony[order,kg],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax)#,vmin=data.g_min[kg],vmax=data.g_max[kg])

    
    plt.axis('off')

    plt.xticks([])
    plt.yticks([])
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
#thresh sim
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    order = np.argsort(target_predict_colony[:,kg])
    plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_predict_colony[order,kg]>thresholds[kg],cmap='YlGnBu',vmin=0,vmax=1) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
    
    plt.axis('off')

    plt.xticks([])
    plt.yticks([])
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])


#thresh expt
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    order = np.argsort(target_colony[:,kg])
    plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_colony[order,kg]>thresholds[kg],cmap='YlGnBu',vmin=0,vmax=1)#,vmin=data.g_min[kg],vmax=data.g_max[kg])

    
    plt.axis('off')

    plt.xticks([])
    plt.yticks([])
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
#thresh overlay
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    indices_pos = target_colony[:,kg]>thresholds[kg]
    plt.scatter(xx_colony[indices_pos],yy_colony[indices_pos],ms,color='b',alpha=0.3,zorder=-1)
    
    indices_pos = target_predict_colony[:,kg]>thresholds[kg]
    plt.scatter(xx_colony[indices_pos],yy_colony[indices_pos],ms,color='r',alpha=0.3,zorder=1)
    
    plt.axis('off')

    plt.xticks([])
    plt.yticks([])
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])


#true pos
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    
    pos_expt = target_colony[:,kg]>thresholds[kg]
    pos_sim = target_predict_colony[:,kg]>thresholds[kg]
    
    indices = pos_expt*pos_sim
    
    plt.scatter(xx_colony[:],yy_colony[:],ms,color='lightgrey')
    plt.scatter(xx_colony[indices],yy_colony[indices],ms,color='green')

    plt.axis('off')
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
    plt.xticks([])
    plt.yticks([])

#true neg
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    neg_expt = target_colony[:,kg]<=thresholds[kg]
    neg_sim = target_predict_colony[:,kg]<=thresholds[kg]
    
    indices = neg_expt*neg_sim
    
    plt.scatter(xx_colony[:],yy_colony[:],ms,color='lightgrey')
    plt.scatter(xx_colony[indices],yy_colony[indices],ms,color='green')

    plt.axis('off')
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
    plt.xticks([])
    plt.yticks([])

#false POSITIVES
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    neg_expt = target_colony[:,kg]<=thresholds[kg]
    pos_sim = target_predict_colony[:,kg]>thresholds[kg]
    
    indices = neg_expt*pos_sim
    
    plt.scatter(xx_colony[:],yy_colony[:],ms,color='lightgrey')
    plt.scatter(xx_colony[indices],yy_colony[indices],ms,color='r')

    plt.axis('off')
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
    plt.xticks([])
    plt.yticks([])

#false NEGATIVES
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    pos_expt = target_colony[:,kg]>thresholds[kg]
    neg_sim = target_predict_colony[:,kg]<=thresholds[kg]
    
    indices = pos_expt*neg_sim
    
    plt.scatter(xx_colony[:],yy_colony[:],ms,color='lightgrey')
    plt.scatter(xx_colony[indices],yy_colony[indices],ms,color='r')

    plt.axis('off')
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'SI_fig_genes_thresh_overview' + '_' + cond + file_suffix)
    
