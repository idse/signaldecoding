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
    directory = 'data_expt_20_bgsub'
    conditions = ['B10','B50','B200']
else:
    directory = 'data_sim'
    conditions = ['low','medium','high']

subdirectory_plot = directory + '/analysis_poserror_all'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)
subdirectory_data = directory + '/data'
    
file_suffix = '.png'
plt.close("all")

N_bins_x = 20

N_cond = len(conditions)

pos_error_final = np.zeros((N_cond,N_bins_x-1,2))
for it_cond,cond in enumerate(conditions):
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()

    for mode_signals in [0,1]:
        
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


        bins_x,mean_x,var_x,pos_error,bins_x_sym,C_matrix_x = fns_plot.calc_profile_poserror(dataset[:,:,:],data.metricdist,N_bins_x,data.r_max)#,overlap=overlap)
        
        pos_error_final[it_cond,:,mode_signals] = pos_error

f = open(subdirectory_plot_data+'/data_poserror_all' + ".pickle",'wb')
pickle.dump((bins_x_sym,pos_error_final), f)
f.close()

    