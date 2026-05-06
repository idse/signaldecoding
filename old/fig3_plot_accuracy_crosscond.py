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
    conditions = ['B10','B50','B200']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

subdirectory_plot_fig = directory + '/fig3'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)


mode_z = 1
mode_input = 's'
mode_classbal = 0


#standard
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''


thresh = 1

mode_reg = 'MLP'
kernel = 'relu'
bottleneck_layer = 3

import fns_plotting_scripts as fns_plot
import matplotlib.pyplot as plt
file_suffix = '.pdf'
plt.close('all')
fs=11
fs2=9
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

N_genes = 16
N_cond = 3


accuracy_all = np.zeros((N_cond,N_genes,2))
for it_cond,cond in enumerate(conditions):
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    subdirectory_plot = directory + '/analysis_regression_sg_multi_z1_3x10/analysis_crosscond_train_B50_pred_' + cond
    subdirectory_plot_data = subdirectory_plot + '/data'
    
    f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (ratios,ratios_conf,accuracy) = pickle.load(f)
    f.close()
    
    indices = ~np.any(accuracy[:,:]==0,axis=1)

    accuracy_all[it_cond,:,0] = np.mean(accuracy[indices,:],axis=0)
    accuracy_all[it_cond,:,1] = np.sqrt(np.var(accuracy[indices,:],axis=0))
    


fs2 = 8
fig_size = [3,4]   
params = {
          'figure.figsize': fig_size,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

plt.figure()


for it_cond,cond in enumerate(conditions):
    plt.errorbar(accuracy_all[it_cond,:,0],np.arange(data.N_genes),xerr=accuracy_all[it_cond,:,1],fmt='o',color=fns_plot.return_colmaps('conditions')[it_cond],capsize=3,ms=3,label=cond)

plt.legend(frameon=False)
plt.xlim([0,1])
plt.ylim([-0.5,data.N_genes-0.5])
plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlabel('accuracy')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'fig3_accuracy_crosscond_' + '_' + mode_reg + kernel + '_' + cond + file_suffix)



