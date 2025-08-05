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
    conditions = ['B50']#['B10','B50','B200']
else:
    directory = 'data_sim'
    conditions = ['low','medium','high']


mode_reg = 'MLP'
kernel = 'relu'


subdirectory_plot = directory + '/analysis_regression_sg_pairs_z1_3x10'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
    
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)
    
subdirectory_data = directory + '/data'

file_suffix = '.pdf'
plt.close("all")



N_cond = len(conditions)

fs = 8
fs2 = 5
 
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

N_split = 5
N_run = 5

for it_cond,cond in enumerate(conditions):
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()

    for kg in range(data.N_genes):
        
        print(kg)
        
        MI_all = np.zeros((data.N_signals,data.N_signals,3))
        
        target = data.genes[:,:,kg]

        for ks1 in range(0,data.N_signals):
            print(ks1)
            for ks2 in range(0,data.N_signals):

                if ks1>=ks2:

                    MI = np.zeros(3)
                    
                    for mode in range(3):
                        if mode == 0:
                            feature = data.signals[:,:,ks1][:,:,np.newaxis]
                        elif mode == 1:
                            feature = data.signals[:,:,ks2][:,:,np.newaxis]
                        elif mode == 2:
                            feature = data.signals[:,:,[ks1,ks2]]
                        
                        MI_splits = np.zeros(N_split)
                        for it in range(N_split):
                            feature_clean,target_clean,_ = fns_plot.clean_data(feature,target[:,:,np.newaxis])
                            
                            MI_runs = np.zeros(N_run)
                            for it_run in range(N_run):
                                MI_here,_ = fns_plot.calc_MI_regression(feature_clean,target_clean,hidden_layer_sizes=(10,10))
                                MI_runs[it_run] = MI_here
                                
                            MI_splits[it] = np.max(MI_runs)

                        MI[mode] = np.mean(MI_splits)
                        
                    MI_all[ks1,ks2,:] = MI
                    
        f = open(subdirectory_plot_data+'/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
        pickle.dump((MI_all), f)
        f.close()
            