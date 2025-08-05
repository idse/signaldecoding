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


mode_smad2 = 0

mode_reg = 'MLP'
kernel = 'relu'

subdirectory_plot = directory + '/analysis_regression_sg_comb_z1_3x10'
subdirectory_plot_data = subdirectory_plot + '/data'
subdirectory_data = directory + '/data'

subdirectory_plot_fig = directory + '/fig2'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)

file_suffix = '.pdf'
plt.close("all")



N_cond = len(conditions)

fs = 12
fs2 = 10
 
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 1-0.2
hspace = 0.001
wspace = 0.001


for it_cond,cond in enumerate(conditions):
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()

    
    f = open(subdirectory_plot_data + '/data_regression_MI_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (MI_individ_all,MI_individ_disc_all,MI_comb_all,MI_comb_disc_all) = pickle.load(f)
    f.close()

    N_split = MI_comb_all.shape[2]
    N_run = MI_comb_all.shape[3]
    
    MI_individ = np.mean(np.max(MI_individ_all,axis=-1),axis=-1)
    MI_comb_max = np.max(MI_comb_all,axis=-1)
    MI_comb_av = np.mean(MI_comb_max,axis=-1)
    MI_comb_err = np.sqrt(np.var(MI_comb_max,axis=-1)/N_split)

    N_var = data.N_signals
    colors = fns_plot.return_colmaps('signals')
    
    
    N_inc_all = [x for x in range(1,N_var+1)]
    N_N_inc = len(N_inc_all)

    
    fig_size = [12,2.3]     
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    H,W = 1,6
    
    N_var = data.N_signals
    colors = fns_plot.return_colmaps('signals')

    N_inc_all = [x for x in range(1,N_var+1)]
    N_N_inc = len(N_inc_all)
    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    
    chrt=0
    for kg in [0,6,9,11,13,14]:
        chrt+=1
        plt.subplot(H,W, chrt)
        plt.title(data.gene_names[kg])
        for k in range(0,N_var):
            if k>0:
                plt.barh(y=k+1,width=MI_individ[k,kg],left=MI_comb_av[k,kg]-MI_individ[k,kg],color=colors[k])
            else:
                plt.barh(y=k+1,width=MI_comb_av[k,kg],color=colors[k])
            plt.errorbar(MI_comb_av[k,kg],k+1,xerr=MI_comb_err[k,kg],capsize=3,color='k')
              
        PI_max = MI_comb_av[-1,kg]
        #plt.plot(0.5*PI_max*np.ones(2),[0,max(N_inc_all)+1],':k')

        plt.xlim([0,np.round(PI_max,2)])
        plt.xticks([0,np.round(PI_max,2)],labels=['0',str(np.round(PI_max,2))])
        plt.ylim([0,max(N_inc_all)+1])
        
        if chrt==1:
            plt.yticks(N_inc_all,labels=data.signal_names)
        else:
            plt.yticks([])
        plt.xlabel('MI (bits)')
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI' + '_fates' + cond + file_suffix)
    