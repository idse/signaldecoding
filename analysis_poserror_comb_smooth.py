#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:22:52 2023

@author: D.Brueckner
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import dill as pickle

import fns_plotting_scripts as fns_plot

from sklearn.feature_selection import mutual_info_regression


mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm'
    conditions = ['B10','B50','B200']
else:
    directory = 'data_sim'
    conditions = ['low','medium','high']

subdirectory_plot = directory + '/analysis_poserror_comb_smooth'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)
subdirectory_data = directory + '/data'
    
file_suffix = '.png'
plt.close("all")

N_bins_x_raw = 20   
N_bins_x = 1000 

N_cond = len(conditions)
PI_all = np.zeros((N_cond,2))
pos_error_final = np.zeros((N_cond,N_bins_x-1,2))
for it_cond,cond in enumerate(conditions):
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()

    for mode_signals in [1]:
        
        if mode_signals == 1:
            suffix = '_signals'
            dataset = data.signals
            N_var = data.N_signals
            names = data.signal_names
        else:
            suffix = '_genes'
            dataset = data.genes
            N_var = data.N_genes
            names = data.gene_names
        
        N_comb_max = 500
        import itertools
        for N_inc in range(1,N_var+1):
            print(N_inc)
            
            combinations_all = list(itertools.combinations(tuple(np.arange(0,N_var)), N_inc))
            N_comb_here = len(combinations_all)
            
            if N_comb_here > N_comb_max:
                #regular indices
                indices = np.arange(0,N_comb_here,np.round(N_comb_here/N_comb_max))
                combinations = [combinations_all[int(i)] for i in indices]
            else:
                combinations = combinations_all
            
            N_comb = len(combinations)

            pos_error_comb_inc = np.zeros((N_bins_x-1,N_comb))
    
            for it_comb,comb in enumerate(combinations):
                
                if np.mod(it_comb,50)==0:
                    print('progress  = ' + str(np.round(it_comb/N_comb*100,4)) + ' %')
                
                indices_comb = np.array(comb)  

                X_here = dataset[:,:,indices_comb]
                bins_x,mean_x,var_x,pos_error,bins_x_sym,C_matrix_x = fns_plot.calc_poserror_smooth(X_here,data.metricdist,N_bins_x_raw,data.r_max,N_bins_x)
                pos_error_comb_inc[:,it_comb] = pos_error
        
        
            f = open(subdirectory_plot_data+'/data_poserror_comb_Ninc' + str(N_inc) + suffix + '_' + cond + ".pickle",'wb')
            pickle.dump((bins_x_sym,pos_error_comb_inc), f)
            f.close()
        
