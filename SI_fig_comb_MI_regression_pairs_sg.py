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
    directory = 'data_expt_20_scaled_norm_bgsub_withSmad2'
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

subdirectory_plot_fig = directory + '/SI_fig_comb_MI'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)


file_suffix = '.pdf'
plt.close("all")



N_cond = len(conditions)

fs = 11
fs2 = 8
 
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

N_split = 5
N_run = 5


mode_run = 0
mode_plot_bars = 0
mode_plot_cmap = 0
mode_plot_cmap_v2 = 1
mode_plot_scatter = 0
mode_plot_MI_all = 0

for it_cond,cond in enumerate(conditions):
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    if mode_run:
        
        
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
                        
    if mode_plot_bars:
        
        
        fig_size = [9,9]     
        params = {
                  'figure.figsize': fig_size,
                  }
        plt.rcParams.update(params)
        H,W = data.N_signals,data.N_signals
        
        colors = fns_plot.return_colmaps('signals')
        
        for kg in range(data.N_genes):
            
            f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
            MI_all = pickle.load(f)
            f.close()
            
            plt.figure()
            plt.suptitle(data.gene_names[kg],size=12)
            chrt=0
            for ks1 in range(0,data.N_signals):
                for ks2 in range(0,data.N_signals):
                    chrt+=1
                    
                    
                    if ks1>=ks2:
                        
                        plt.subplot(H,W,chrt)
                        plt.title('1:'+data.signal_names[ks1]+', 2:'+data.signal_names[ks2])
                        
                        #for mode in range(2):
                        plt.barh(y=0,width=MI_all[ks1,ks2,0],color=colors[ks1])
                        plt.barh(y=1,width=MI_all[ks1,ks2,1],color=colors[ks2])
                        plt.barh(y=2,width=MI_all[ks1,ks2,2],color='k')
                        plt.barh(y=3,width=MI_all[ks1,ks2,2]-MI_all[ks1,ks2,1],color=colors[ks1])
                        plt.barh(y=4,width=MI_all[ks1,ks2,2]-MI_all[ks1,ks2,0],color=colors[ks2])
                        
                        plt.yticks(np.arange(5),['I1','I2','I12','U1','U2'])
                        plt.xlim([0,0.8])


            plt.tight_layout()
    
            plt.savefig(subdirectory_plot_fig + "/" + 'MI_pairs' + '_kg' + str(kg) + '_' + cond + file_suffix)
        
    
    if mode_plot_cmap:
        
        fig_size = [3,2.5]     
        params = {
                  'figure.figsize': fig_size,
                  }
        plt.rcParams.update(params)
        
        for kg in [0]:#range(data.N_genes):
            
            f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
            MI_all = pickle.load(f)
            f.close()
            
            
            plt.figure()
            plt.title(data.gene_names[kg],size=12)
            
            Synergy = np.zeros((data.N_signals,data.N_signals))
            Synergy[:] = np.nan
            for ks1 in range(0,data.N_signals):
                for ks2 in range(0,data.N_signals):
                    if ks1>=ks2:
                        Synergy[ks1,ks2] = ( (MI_all[ks1,ks2,0] + MI_all[ks1,ks2,1]) - MI_all[ks1,ks2,2] )/MI_all[ks1,ks2,2]
            
            plt.imshow(Synergy,vmin=-1,vmax=1,cmap='coolwarm')
            
            plt.xticks(np.arange(0,data.N_signals),labels=data.signal_names,rotation=45)
            plt.yticks(np.arange(0,data.N_signals),labels=data.signal_names)

            plt.colorbar()
            plt.tight_layout()
    
            plt.savefig(subdirectory_plot_fig + "/" + 'MI_pairs_norm_cmap' + '_kg' + str(kg) + '_' + cond + file_suffix)
            
            
    if mode_plot_cmap_v2:
        
        fig_size = [3,2.5]     
        params = {
                  'figure.figsize': fig_size,
                  }
        plt.rcParams.update(params)
        
        Synergy_sum = np.zeros((data.N_signals,data.N_signals))
        for kg in range(data.N_genes):
            
            f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
            MI_all = pickle.load(f)
            f.close()
            
            Synergy = np.zeros((data.N_signals,data.N_signals))
            Synergy[:] = np.nan
            for ks1 in range(0,data.N_signals):
                for ks2 in range(0,data.N_signals):
                    if ks1>=ks2:
                        Synergy[ks1,ks2] = ( (MI_all[ks1,ks2,0] + MI_all[ks1,ks2,1]) - MI_all[ks1,ks2,2] )/MI_all[ks1,ks2,2]
            
            Synergy_sum += Synergy
            
        Synergy_sum /= data.N_signals
            
        plt.figure()
        
        plt.imshow(Synergy_sum,vmin=-1,vmax=1,cmap='coolwarm')
        
        plt.xticks(np.arange(0,data.N_signals),labels=data.signal_names,rotation=45)
        plt.yticks(np.arange(0,data.N_signals),labels=data.signal_names)

        plt.colorbar()
        plt.tight_layout()

        plt.savefig(subdirectory_plot_fig + "/" + 'MI_synergy' + '_av_' + cond + file_suffix)
            
        
        H,W=1,data.N_genes  
        fig_size = [17,1.6]   
        params = {
                  'figure.figsize': fig_size,
                  }
        plt.rcParams.update(params)
        
        margin_left = 0.001
        margin_right = 0.001
        margin_bottom = 0.001
        margin_top = 1-0.2
        hspace = 0.001
        wspace = 0.001
        
        N_var = data.N_signals
        N_inc_all = [x for x in range(1,N_var+1)]
        N_N_inc = len(N_inc_all)
        
        fig = plt.figure()
        chrt=0
        fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
        
        for kg in range(data.N_genes):
            
            f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
            MI_all = pickle.load(f)
            f.close()
            
            Synergy = np.zeros((data.N_signals,data.N_signals))
            Synergy[:] = np.nan
            for ks1 in range(0,data.N_signals):
                for ks2 in range(0,data.N_signals):
                    if ks1>=ks2:
                        Synergy[ks1,ks2] = ( (MI_all[ks1,ks2,0] + MI_all[ks1,ks2,1]) - MI_all[ks1,ks2,2] )/MI_all[ks1,ks2,2]
            
            chrt+=1
            plt.subplot(H,W, chrt)
            plt.title(data.gene_names[kg])
            for k in range(0,N_var):
                
                plt.imshow(Synergy,vmin=-1,vmax=1,cmap='coolwarm')

            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()

        plt.savefig(subdirectory_plot_fig + "/" + 'MI_synergy' + '_all' + cond + file_suffix)
            
            
    
    if mode_plot_scatter:
        
        fig_size = [3,3]     
        params = {
                  'figure.figsize': fig_size,
                  }
        plt.rcParams.update(params)
        
        plt.figure()
        for kg in range(data.N_genes):
            
            f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
            MI_all = pickle.load(f)
            f.close()

            Synergy = np.zeros((data.N_signals,data.N_signals))
            Synergy[:] = np.nan
            for ks1 in range(0,data.N_signals):
                for ks2 in range(0,data.N_signals):
                    if ks1>=ks2:
                        Synergy[ks1,ks2] = ( (MI_all[ks1,ks2,0] + MI_all[ks1,ks2,1]) - MI_all[ks1,ks2,2] )/MI_all[ks1,ks2,2]
            
            for ks1 in range(0,data.N_signals):
                for ks2 in range(0,data.N_signals):
                    if ks2 == 4 and ks1 == 5:
                        plt.scatter(Synergy[ks1,ks2],kg,s=20,color='b',zorder=2)
                    if ks1>ks2:
                        plt.scatter(Synergy[ks1,ks2],kg,s=10,color='grey')
                    else:
                        plt.scatter(Synergy[ks1,ks2],kg,s=10,color='r')
                        
            """
            xx = Synergy.ravel()
            plt.scatter(xx,kg*np.ones(xx.shape),s=10,color='grey')
            
            ks1 = 4
            ks2 = 5
            plt.scatter(Synergy[ks2,ks1],kg,s=15,color='b')
            """
            plt.xlim([-1,1])
            #plt.xticks(np.arange(0,data.N_signals),labels=data.signal_names,rotation=45)
            plt.yticks(np.arange(0,data.N_genes),labels=data.gene_names)

            
            plt.tight_layout()
    
            plt.savefig(subdirectory_plot_fig + "/" + 'MI_pairs_norm_cmap' + '_kg' + str(kg) + '_' + cond + file_suffix)
    
    
    if mode_plot_MI_all:
        f = open(subdirectory_plot_data + '/data_regression_MI_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
        (MI_individ_all,MI_individ_disc_all,MI_comb_all,MI_comb_disc_all) = pickle.load(f)
        f.close()
        
        N_split = MI_comb_all.shape[2]
        N_run = MI_comb_all.shape[3]
        
        MI_individ = np.mean(np.max(MI_individ_all,axis=-1),axis=-1)
        MI_comb_max = np.max(MI_comb_all,axis=-1)
        MI_comb_av = np.mean(MI_comb_max,axis=-1)
        
        MI_complete = MI_comb_av[-1,:]
        
        fig_size = [3,4]     
        params = {
                  'figure.figsize': fig_size,
                  }
        plt.rcParams.update(params)
        
        plt.figure()
        for kg in range(data.N_genes):
            
            f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
            MI_all = pickle.load(f)
            f.close()
            
            plt.scatter(MI_individ[:,kg]/MI_complete[kg],kg)
                    
            plt.xlim([0,1])
            plt.yticks(np.arange(0,data.N_genes),labels=data.gene_names)

            
            plt.tight_layout()
    
            plt.savefig(subdirectory_plot_fig + "/" + 'MI_pairs_norm_cmap' + '_kg' + str(kg) + '_' + cond + file_suffix)
        
    
    """
    #Test stuff 
    fig_size = [3,3]     
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    plt.figure()
    for kg in range(data.N_genes):
        
        f = open(subdirectory_plot_data + '/data_regression_pairs_MI_kg' + str(kg) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
        MI_all = pickle.load(f)
        f.close()

        Synergy = np.zeros((data.N_signals,data.N_signals))
        Synergy[:] = np.nan
        for ks1 in range(0,data.N_signals):
            for ks2 in range(0,data.N_signals):
                if ks1>=ks2:
                    Synergy[ks1,ks2] = ( (MI_all[ks1,ks2,0] + MI_all[ks1,ks2,1]) - MI_all[ks1,ks2,2] )#/MI_all[ks1,ks2,2]
        
        for ks1 in range(0,data.N_signals):
            ks2 = ks1
            plt.scatter(MI_all[ks1,ks2,0],Synergy[ks1,ks2],s=20,color='b',zorder=2)
            #plt.scatter(Synergy[ks1,ks2],kg,s=20,color='b',zorder=2)
            
            plt.plot([-1,1],[-1,1],'-k')
            
        plt.xlim([-1,1])
        #plt.yticks(np.arange(0,data.N_genes),labels=data.gene_names)

        
        plt.tight_layout()

        #plt.savefig(subdirectory_plot_fig + "/" + 'MI_pairs_norm_cmap' + '_kg' + str(kg) + '_' + cond + file_suffix)
    """
