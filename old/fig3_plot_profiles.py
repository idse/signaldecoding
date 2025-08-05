#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:22:52 2023

@author: D.Brueckner

1D formatting and new opt script: no repetition of signals
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import dill as pickle
import fns_plotting_scripts as fns_plot

from sklearn.feature_selection import mutual_info_regression

mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    conditions = ['B10','B50','B200']
else:
    directory = 'data_sim'
    conditions = ['low','medium','high']


subdirectory_data = directory + '/data'

subdirectory_plot_fig = directory + '/fig3'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)

file_suffix = '.pdf'
plt.close("all")

N_cond = len(conditions)

cond = 'B50'
f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
data = pickle.load(f)
f.close()
s_lim = np.zeros((data.N_signals,2))
for k in range(0,data.N_signals):

    data_here = data.signals[:,:,k]
    s_lim[k,0] = np.nanpercentile(data_here.ravel(),0.5)
    s_lim[k,1] = np.nanpercentile(data_here.ravel(),99.5)

fs = 12
fs2 = 10
 
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

fig_size = [9,3]     
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
H,W = 1,3
chrt=0
plt.figure()
for it_cond,cond in enumerate(conditions):
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    chrt+=1
    plt.subplot(H,W,chrt)
    
    colors = fns_plot.return_colmaps('signals')

    N_bins_x = 20
    for k in range(0,data.N_signals):
    
        data_here = data.signals[:,:,k]
        
        s_min = s_lim[k,0]
        s_max = s_lim[k,1]
        
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar((data_here[:,:,np.newaxis]-s_min)/s_max,data.metricdist,N_bins_x,data.r_max)
        bins_x[-1] = 350
        
        plt.plot(bins_x,mean_x,color=colors[k],lw=2)
        #plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color=colors[k],alpha=0.2)
        
        plt.ylim([0,1])
        plt.yticks([])
        plt.xlabel(r'edge distance ($\mu$m)')
        plt.xlim([0,360])
        if not it_cond>0:
            plt.ylabel('intensity (a.u.)')
        
plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'profiles_' + cond + file_suffix)
    


fig_size = [4,3]     
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

plt.figure()
for it_cond,cond in enumerate(conditions):
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    N_bins_x = 20
    k = 0
    
    data_here = data.signals[:,:,k]
    
    s_min = s_lim[k,0]
    s_max = s_lim[k,1]
    
    bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(data_here[:,:,np.newaxis],data.metricdist,N_bins_x,data.r_max)
    bins_x[-1] = 350
    
    plt.plot(bins_x,mean_x,lw=2)
    #plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color=colors[k],alpha=0.2)
    
 
    plt.yticks([])
    plt.xlabel(r'edge distance ($\mu$m)')
    plt.xlim([0,360])
    if not it_cond>0:
        plt.ylabel('intensity (a.u.)')
        
plt.tight_layout()

#plt.savefig(subdirectory_plot_fig + "/" + 'profiles_' + cond + file_suffix)