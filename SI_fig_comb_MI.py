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


#subdirectory_plot = directory + '/analysis_regression_sg_comb_z1_LS_3x10_v4'
#subdirectory_plot = directory + '/analysis_regression_sg_comb_z1_2x50'
subdirectory_plot = directory + '/analysis_regression_sg_comb_z1_3x10'
subdirectory_plot_data = subdirectory_plot + '/data'
subdirectory_data = directory + '/data'

subdirectory_plot_fig = directory + '/SI_fig_comb_MI'
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

        
    margin_left = 0.001
    margin_right = 0.001
    margin_bottom = 0.001
    margin_top = 1-0.2
    hspace = 0.001
    wspace = 0.001
    
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

    
    
    N_var = data.N_signals
    colors = fns_plot.return_colmaps('signals')
    
    
    H,W=1,data.N_genes  
    fig_size = [17,1.6]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    
    for kg in range(data.N_genes):
        chrt+=1
        plt.subplot(H,W, chrt)
        plt.title(data.gene_names[kg])
        for k in range(0,N_var):
            
            plt.barh(y=N_inc_all[k],width=MI_individ[k,kg],left=MI_comb_av[k,kg]-MI_individ[k,kg],color=colors[k])
            #plt.errorbar(PI_comb_av[k,kg],N_inc_all[k],xerr=PI_comb_err[k,kg],capsize=3,color='k')
                
        PI_max = MI_comb_av[-1,kg]
        
        plt.xlim([0,np.round(PI_max,2)])
        plt.xticks([0,np.round(PI_max,2)],labels=['0',str(np.round(PI_max,2))])
        plt.ylim([0,max(N_inc_all)+1])
        
        #plt.yticks(N_inc_all,labels=data.signal_names)
        
        plt.yticks([])
        #plt.xlabel('MI (bits)')
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI' + '_all' + cond + file_suffix)
    
    
    
    H,W=1,data.N_signals
    fig_size = [15,3.5]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)

    for ks in range(data.N_signals):
        chrt+=1
        plt.subplot(H,W,chrt)
        plt.title(data.signal_names[ks])
        plt.barh(y=np.arange(data.N_genes),width=MI_individ[ks,:]/MI_comb_av[-1,:],color=colors[ks])
        
        if ks==0:
            plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
        else:
            plt.yticks([])
        
        plt.ylim([-1,data.N_genes])
        plt.xlabel('fraction of total')
        plt.xlim([0,1])
        plt.gca().invert_yaxis()
       
    
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI_single_frac' + '_all' + cond + file_suffix)
    
    
    
    
    fig_size = [4,3]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    plt.figure()
    percentage_max = np.zeros(data.N_genes)
    
    frac_all = np.zeros(data.N_genes)
    
    for kg in range(data.N_genes):
        PI_max = MI_comb_av[-1,kg]
        ks_opt = np.argmax(MI_individ[:,kg])
        frac_all[kg] = MI_individ[ks_opt,kg]/PI_max
        plt.barh(y=kg,width=frac_all[kg],color=colors[ks_opt])
        
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    
    plt.xlim([0,1])
        
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI_frac' + '_all' + cond + file_suffix)

        
    fig_size = [3.5,4]     
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    MI_complete = MI_comb_av[-1,:]
    
    plt.figure()
    for kg in range(data.N_genes):
        for ks in range(data.N_signals):
            plt.scatter(MI_individ[ks,kg]/MI_complete[kg],kg,color=colors[ks])
                
    plt.xlim([0,1])
    plt.yticks(np.arange(0,data.N_genes),labels=data.gene_names)
    plt.xlabel('fraction of total')
    
    plt.gca().invert_yaxis()
    
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI_frac_single' + '_' + cond + file_suffix)
    

    plt.figure()
    for kg in range(data.N_genes):
        plt.plot([MI_comb_av[-1,kg],np.sum(MI_individ[:,kg])],kg*np.ones(2),'-',color='grey')
        plt.plot([MI_comb_av[-1,kg]],kg,'o',color='b',ms=5)
        plt.plot([np.sum(MI_individ[:,kg])],kg,'s',color='r',ms=5)
    plt.ylim([-1,data.N_genes])

    
    plt.yticks(np.arange(0,data.N_genes),labels=data.gene_names)
    plt.xlabel('MI (bits)')
    plt.xlim([0,3])
    plt.xticks([0,1,2,3])
    plt.gca().invert_yaxis()
    
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI_all_vs_sum' + '_' + cond + file_suffix)
    
    
    plt.figure()
    sum_frac = 0
    for kg in range(data.N_genes):
        frac = MI_comb_av[-1,kg]/np.sum(MI_individ[:,kg])
        sum_frac += frac
        plt.plot([frac],kg,'o',color='k',ms=5)
    
    frac_av = sum_frac/data.N_genes
    print(frac_av)
    
    plt.plot(frac_av*np.ones(2),[-1,data.N_genes],'--k')
    
    plt.ylim([-1,data.N_genes])

    plt.yticks(np.arange(0,data.N_genes),labels=data.gene_names)
    plt.xlabel(r'$I(\{s\},g_k)/\Sigma_j I(s_j,g_k)$')
    plt.xlim([0,1])
    plt.gca().invert_yaxis()
    
    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'MI_all_vs_sum_frac' + '_' + cond + file_suffix)
    
     
    