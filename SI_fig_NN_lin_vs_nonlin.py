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
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''
    
subdirectory_plot_all = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_3x10_v2'
subdirectory_plot = directory + '/analysis_' + 'regression' + '_sg_single_z' + str(mode_z) + '_SVR'

subdirectory_plot_data_all = subdirectory_plot_all + '/data'
subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/SI_fig_NN'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)

thresh = 1

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

mode_reg = 'MLP'
kernel = 'relu'
f = open(subdirectory_plot_data_all + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios) = pickle.load(f)
f.close()

mode_reg = 'linear'
kernel = ''
f = open(subdirectory_plot_data_all + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios1) = pickle.load(f)
f.close()

#"""
mode_reg = 'SVR'
kernel = 'rbf'
f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios2) = pickle.load(f)
f.close()
#"""

fig_size = [3,4]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

names = ['specificity','sensitivity','negative predictive value','precision','accuracy']

for mode in [0,1,2,3,4]:
    name = names[mode]

    plt.figure()
    
    indices = ~np.any(ratios[:,:,mode]==0,axis=1)
    plt.errorbar(np.mean(ratios[indices,:,mode],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,mode],axis=0)),fmt='o',color='k',capsize=3,ms=3)
    plt.errorbar(np.mean(ratios1[:,:,mode],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios1[:,:,mode],axis=0)),fmt='s',color='r',capsize=3,ms=3)
    #plt.errorbar(np.mean(ratios2[:,:,mode],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios1[:,:,mode],axis=0)),fmt='s',color='g',capsize=3,ms=3)


    plt.xlim([0,1])
    plt.ylim([-0.5,data.N_genes-0.5])
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    
    plt.xlabel(name)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'lin_vs_nonlin_fig_posneg_' + name + '_' + cond + file_suffix)

