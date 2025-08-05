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
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''
    
subdirectory_plot = directory + '/analysis_regression_sweep_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z)

subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/SI_fig_NN'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)

thresh = 1

mode_reg = 'MLP'
kernel = 'relu'
bottleneck_layer = 3

import fns_plotting_scripts as fns_plot
import matplotlib.pyplot as plt
file_suffix = '.pdf'
plt.close('all')
fs=13
fs2=11
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)


f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
data = pickle.load(f)
f.close()

f = open(subdirectory_plot_data + '/data_regression_sweep_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(ratios,ratios_conf,accuracy,MIs) = pickle.load(f)
f.close()

fig_size = [5,3]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

accuracy_mean = np.mean(accuracy,axis=2)
precision_mean = np.mean(ratios_conf[:,:,:,:,1],axis=2)

N_layers_all = [1,2,3,4,5]
N_nodes_all = [1,2,4,6,8,10,15,20]
N_N_layers = len(N_layers_all)
N_N_nodes = len(N_nodes_all)

plt.figure()
plt.imshow(np.mean(accuracy_mean,axis=-1),vmin=0.8,vmax=0.95,cmap='bwr')
plt.colorbar(ticks=[0.8,0.95])
plt.yticks(np.arange(N_N_layers),labels=N_layers_all)
plt.xticks(np.arange(N_N_nodes),labels=N_nodes_all)

plt.xlabel('nodes per layer')
plt.ylabel('number of layers')

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'sweep_' + 'accuracy' + '_' + mode_reg + kernel + '_' + cond + file_suffix)


plt.figure()
plt.imshow(np.mean(precision_mean,axis=-1),vmin=0.6,vmax=0.8,cmap='bwr')
plt.colorbar(ticks=[0.6,0.8])
plt.yticks(np.arange(N_N_layers),labels=N_layers_all)
plt.xticks(np.arange(N_N_nodes),labels=N_nodes_all)

plt.xlabel('nodes per layer')
plt.ylabel('number of layers')

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'sweep_' + 'precision' + '_' + mode_reg + kernel + '_' + cond + file_suffix)

