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

subdirectory_plot = directory + '/analysis_regression_MI_test'
subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/SI_fig_NN'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)

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

f = open(subdirectory_plot_data + '/data_regression_MItest_single_' + cond + ".pickle",'rb')
(MI_single) = pickle.load(f)
f.close()

f = open(subdirectory_plot_data + '/data_regression_MItest_multi_' + cond + ".pickle",'rb')
(MI_multi) = pickle.load(f)
f.close()

fig_size = [3.2,3]   
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

plt.figure()
for ks in range(data.N_signals):
    plt.scatter(MI_single[:,ks,0,:],MI_single[:,ks,1,:],color=fns_plot.return_colmaps('signals')[ks])
plt.plot([0,1e3],[0,1e3],'-k',lw=1)

plt.xlim([0,1])
plt.ylim([0,1])

plt.xlabel('MI test (bits)')
plt.ylabel('MI train (bits)')

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'MI_single_test_train' + '_' + cond + file_suffix)


plt.figure()
for ks in range(data.N_signals):
    plt.scatter(MI_single[:,ks,0,:],MI_single[:,ks,2,:],color=fns_plot.return_colmaps('signals')[ks])
plt.plot([0,1e3],[0,1e3],'-k',lw=1)

plt.xlim([0,1])
plt.ylim([0,1])

plt.xlabel('MI test (bits)')
plt.ylabel('MI knn (bits)')

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'MI_single_test_knn' + '_' + cond + file_suffix)


plt.figure()
plt.scatter(MI_multi[:,0,:],MI_multi[:,1,:])
plt.plot([0,1e3],[0,1e3],'-k',lw=1)

plt.xlim([0,1.5])
plt.ylim([0,1.5])
plt.xlabel('MI test (bits)')
plt.ylabel('MI train (bits)')

plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'MI_multi_test_train' + '_' + cond + file_suffix)