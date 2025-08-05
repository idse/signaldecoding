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
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)

mode_analyze_single = 1
mode_analyze_multi = 1

N_run = 2

f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
data = pickle.load(f)
f.close()

if mode_analyze_single:

    MI_single = np.zeros((data.N_genes,data.N_signals,3,N_run))
    for it in range(N_run):
        for ks in range(data.N_signals):
    
            feature = data.signals[:,:,ks]
            for kg in range(data.N_genes):
                
                target = data.genes[:,:,kg]
                
                feature_clean,target_clean,_ = fns_plot.clean_data(feature[:,:,np.newaxis],target[:,:,np.newaxis])
                
                MI_single[kg,ks,0,it],MI_single[kg,ks,1,it] = fns_plot.calc_MI_regression(feature_clean,target_clean,hidden_layer_sizes=(10,10,10))
                MI_single[kg,ks,2,it] = fns_plot.calc_MI_sklearn(feature_clean,target_clean)
                
    f = open(subdirectory_plot_data+'/data_regression_MItest_single_' + cond + ".pickle",'wb')
    pickle.dump((MI_single), f)
    f.close()
                
            
if mode_analyze_multi:
    
    feature = data.signals[:,:,:]
    
    MI_multi = np.zeros((data.N_genes,2,N_run))
    for it in range(N_run):
        for kg in range(data.N_genes):
            
            target = data.genes[:,:,kg]
            
            feature_clean,target_clean,_ = fns_plot.clean_data(feature,target[:,:,np.newaxis])
            
            MI_multi[kg,0,it],MI_multi[kg,1,it] = fns_plot.calc_MI_regression(feature_clean,target_clean,hidden_layer_sizes=(10,10))
        
    f = open(subdirectory_plot_data+'/data_regression_MItest_multi_' + cond + ".pickle",'wb')
    pickle.dump((MI_multi), f)
    f.close()
    
    
    
    