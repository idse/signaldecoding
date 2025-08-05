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
    cond = 'B200'
    #['B10','B50','B200']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'


mode_z = 1
mode_classbal = 0


#standard
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''
    
mode_input = 's'
subdirectory_plot_all = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_LS_2x10_2_2x10_v3'

#r to g
mode_input = 'r'
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_LS_2x10_2_2x10_v3'

subdirectory_plot_data_all = subdirectory_plot_all + '/data'
subdirectory_plot_data = subdirectory_plot + '/data'


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


f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
data = pickle.load(f)
f.close()

f = open(subdirectory_plot_data_all + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios,ratios_conf,accuracy,corrs,variances) = pickle.load(f)
f.close()

f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios1,ratios_conf1,accuracy1,corrs1,variances1) = pickle.load(f)
f.close()

fig_size = [3,4]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

plt.figure()

indices = ~np.any(ratios[:,:,1]==0,axis=1)
plt.errorbar(np.mean(ratios[indices,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,1],axis=0)),fmt='o',color='k',capsize=3,ms=3)

#plt.errorbar(np.mean(ratios_random[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_random[:,:,1],axis=0)),fmt='x',color='b',capsize=3,ms=3)
plt.errorbar(np.mean(ratios1[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios1[:,:,1],axis=0)),fmt='s',color='r',capsize=3,ms=3)


plt.xlim([0,1])
plt.ylim([-0.5,data.N_genes-0.5])
plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlabel('sensitivity')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(subdirectory_plot + "/" + 'single_vs_multi_fig_posneg_' + 'sensitivity' + '_' + mode_reg + kernel + '_' + cond + file_suffix)



plt.figure()

indices = ~np.any(ratios[:,:,1]==0,axis=1)
plt.errorbar(np.mean(ratios[indices,:,0],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,0],axis=0)),fmt='o',color='k',capsize=3,ms=3)

#plt.errorbar(np.mean(ratios_random[:,:,0],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_random[:,:,0],axis=0)),fmt='x',color='b',capsize=3,ms=3)
plt.errorbar(np.mean(ratios1[:,:,0],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios1[:,:,0],axis=0)),fmt='s',color='r',capsize=3,ms=3)


plt.xlim([0,1])
plt.ylim([-0.5,data.N_genes-0.5])
plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlabel('specificity')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(subdirectory_plot + "/" + 'single_vs_multi_fig_posneg_' + 'specificity' + '_' + mode_reg + kernel + '_' + cond + file_suffix)


plt.figure()

indices = ~np.any(ratios_conf[:,:,1]==0,axis=1)
plt.errorbar(np.mean(ratios_conf[indices,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_conf[indices,:,1],axis=0)),fmt='o',color='k',capsize=3,ms=3)

#plt.errorbar(np.mean(ratios_conf_random[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_conf_random[:,:,1],axis=0)),fmt='x',color='b',capsize=3,ms=3)
plt.errorbar(np.mean(ratios_conf1[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_conf1[:,:,1],axis=0)),fmt='s',color='r',capsize=3,ms=3)


plt.xlim([0,1])
plt.ylim([-0.5,data.N_genes-0.5])
plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlabel('precision')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(subdirectory_plot + "/" + 'single_vs_multi_fig_posneg_' + 'precision' + '_' + mode_reg + kernel + '_' + cond + file_suffix)


plt.figure()

indices = ~np.any(ratios_conf[:,:,1]==0,axis=1)
plt.errorbar(np.mean(ratios_conf[indices,:,0],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_conf[indices,:,0],axis=0)),fmt='o',color='k',capsize=3,ms=3)

#plt.errorbar(np.mean(ratios_conf_random[:,:,0],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_conf_random[:,:,0],axis=0)),fmt='x',color='b',capsize=3,ms=3)
plt.errorbar(np.mean(ratios_conf1[:,:,0],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_conf1[:,:,0],axis=0)),fmt='s',color='r',capsize=3,ms=3)


plt.xlim([0,1])
plt.ylim([-0.5,data.N_genes-0.5])
plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlabel('negative predictive value')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(subdirectory_plot + "/" + 'single_vs_multi_fig_posneg_' + 'negpredval' + '_' + mode_reg + kernel + '_' + cond + file_suffix)



plt.figure()

indices = ~np.any(accuracy[:,:]==0,axis=1)

plt.errorbar(np.mean(accuracy[indices,:],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(accuracy[indices,:],axis=0)),fmt='o',color='k',capsize=3,ms=3,label='')

#plt.errorbar(np.mean(accuracy_random[:,:],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(accuracy_random[:,:],axis=0)),fmt='x',color='b',capsize=3,ms=3)
plt.errorbar(np.mean(accuracy1[:,:],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(accuracy1[:,:],axis=0)),fmt='s',color='r',capsize=3,ms=3)


plt.xlim([0,1])
plt.ylim([-0.5,data.N_genes-0.5])
plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlabel('accuracy')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(subdirectory_plot + "/" + 'single_vs_multi_fig_posneg_' + 'accuracy' + '_' + mode_reg + kernel + '_' + cond + file_suffix)

