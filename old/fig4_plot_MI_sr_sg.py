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
    conditions = ['B10','B50','B200']
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
    
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_3x10_v2'

subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/fig4'
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
fs=11
fs2=9
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

for cond in conditions:
    print(cond)
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    fs2 = 8
    fig_size = [3,4]   
    params = {
              'figure.figsize': fig_size,
              'xtick.labelsize': fs2,
              'ytick.labelsize': fs2,
              }
    plt.rcParams.update(params)
    
    
    
    f = open(subdirectory_plot_data + '/data_regression_MI_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (MIs) = pickle.load(f)
    f.close()
    
    plt.figure()
    
    indices = ~np.any(MIs[:,:]==0,axis=1)
    
    plt.errorbar(np.mean(MIs[indices,:],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(MIs[indices,:],axis=0)),fmt='o',color='k',capsize=3,ms=8,label='')
    
    MI_gr = np.zeros(data.N_genes)
    for kg in range(0,data.N_genes):
        MI_gr[kg] = fns_plot.calc_MI_sklearn(data.genes[:,:,kg],data.metricdist[:,:])
    
    plt.scatter(MI_gr,np.arange(data.N_genes),edgecolors='k',color='w')
    
    #plt.plot(PI_all_g*np.ones(2),[-0.5,data.N_genes-0.5],':r')
    #plt.plot(PI_all_s*np.ones(2),[-0.5,data.N_genes-0.5],'-k')
    
    plt.xlim([0,1.5])
    plt.ylim([-0.5,data.N_genes-0.5])
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    #plt.xticks([0,0.2,0.4,0.6,0.8,1])
    
    plt.xlabel('MI (bits)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'SI_MI_sr_vs_sg' + '_' + mode_reg + kernel + '_' + cond + file_suffix)
    
    
    
    plt.figure()
    
    indices = ~np.any(MIs[:,:]==0,axis=1)
    
    
    #plt.errorbar(np.mean(MIs[indices,:],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(MIs[indices,:],axis=0)),fmt='o',color='k',capsize=3,ms=8,label='')
    
    MI_gr = np.zeros(data.N_genes)
    for kg in range(0,data.N_genes):
        MI_gr[kg] = fns_plot.calc_MI_sklearn(data.genes[:,:,kg],data.metricdist[:,:])
    
    plt.scatter(MI_gr/np.mean(MIs[indices,:],axis=0),np.arange(data.N_genes),color='k')
    
    #plt.plot(PI_all_g*np.ones(2),[-0.5,data.N_genes-0.5],':r')
    plt.plot(np.ones(2),[-0.5,data.N_genes-0.5],'--k')
    
    plt.xlim([0,1.5])
    plt.ylim([-0.5,data.N_genes-0.5])
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    #plt.xticks([0,0.2,0.4,0.6,0.8,1])
    
    plt.xlabel(r'$I(r,g)/I(\{s\},g)$')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'SI_MI_sr_vs_sg_ratio' + '_' + mode_reg + kernel + '_' + cond + file_suffix)
    
