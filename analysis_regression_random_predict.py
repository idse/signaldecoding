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
    directory = 'data_expt_20_scaled_norm_bgsub' #
    conditions = ['B10','B50','B200']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'
for cond in conditions:
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()

    N_run = 100
    thresh = 1

    ratios_random_prop = np.zeros((N_run,data.N_genes,5))

    experimental_prop = np.zeros((data.N_genes))
    
    from sklearn.metrics import confusion_matrix

    for kg in range(0,data.N_genes):
        feature_clean,target_clean,metricdist_clean,markers_clean = fns_plot.clean_data_full(data,data.signals[:,:,:],data.genes[:,:,kg][:,:,np.newaxis])
        
        pos_expt = target_clean>thresh
        
        fraction_pos = np.sum(pos_expt)/len(pos_expt)
        experimental_prop[kg] = fraction_pos
        
        for it in range(N_run):
            
            pos_random_prop = np.random.choice([0,1],size=pos_expt.shape,p=[1-fraction_pos,fraction_pos]) #expt proportions

            TN, FP, FN, TP = confusion_matrix(pos_expt, pos_random_prop).ravel()

            ratios_random_prop[it,kg,0] = fns_plot.diff_zero(TN,(TN+FP))
            ratios_random_prop[it,kg,1] = fns_plot.diff_zero(TP,(TP+FN))
            
            ratios_random_prop[it,kg,2] = fns_plot.diff_zero(TN,(TN+FN))
            ratios_random_prop[it,kg,3] = fns_plot.diff_zero(TP,(TP+FP))
            
            ratios_random_prop[it,kg,4] = fns_plot.diff_zero((TP+TN),(TN + FP + FN + TP))
        
    f = open(subdirectory_data+'/data_regression_av_random' + '_' + cond + ".pickle",'wb')
    pickle.dump((ratios_random_prop), f)
    f.close()
    
        
        
        