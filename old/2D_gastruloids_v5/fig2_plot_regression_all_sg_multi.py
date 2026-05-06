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

mode_plot_signals = 0
mode_plot_predict_patterns = 0
mode_plot_fates = 1
mode_accuracy = 0
mode_plot_predict_eval = 0

mode_accuracy_indices = 0

mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    cond = 'B50'
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'
    
mode = 'fig2_standard'
    
if mode == 'fig2_standard':
    subdirectory_plot = directory + '/analysis_regression_sg_multi_vib'
    subdirectory_plot_fig = subdirectory_plot

if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)

subdirectory_plot_data = subdirectory_plot + '/data'

thresh = 1

mode_reg = 'VIB'
kernel = ''

import fns_plotting_scripts as fns_plot
import matplotlib.pyplot as plt
file_suffix = '.png'
file_suffix_pdf = '.pdf'
plt.close('all')
fs=11
fs2=9
params = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)


f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
data = pickle.load(f)
f.close()

f = open(subdirectory_plot_data + '/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(vib, feat_train, feat_test, tar_train, tar_test, 
            metricdist_train, metricdist_test, markers_train, markers_test, 
            target_predict_train, target_predict, 
            scaler_X, scaler_Y) = pickle.load(f)
f.close()

it = 1
f = open(subdirectory_plot_data + '/data_colony_it' + str(it) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(xx_colony, yy_colony, rr_colony, markers_colony, target_colony, target_predict_colony) = pickle.load(f)
f.close()


margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 1-0.2
hspace = 0.001
wspace = 0.001

lim = 360
ms = 0.15

if mode_plot_signals:
    min_val = 0
    max_val = 3
    max_val_vmax = 3
    
    H,W=1,data.N_signals
    fig_size = [7,1.3]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    signal_colony = data.signals[it,:,:]
    for ks in range(0,data.N_signals):

        chrt+=1
        plt.subplot(H,W,chrt)
        order = np.argsort(signal_colony[:,ks])
        plt.scatter(data.X[it,order,0],data.X[it,order,1],ms,c=signal_colony[order,ks],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
        
        plt.title(data.signal_names[ks],fontsize=10)
        plt.axis('off')
        
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])

    plt.savefig(subdirectory_plot_fig + "/" + 'signals' + '_' + cond + file_suffix)
    
    

if mode_plot_predict_patterns:
    
    min_val = 0
    max_val = 9
    max_val_vmax = 5

    H,W=1,data.N_genes  
    fig_size = [data.N_genes,1.3]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    for kg in range(0,data.N_genes):

        chrt+=1
        plt.subplot(H,W,chrt)
        order = np.argsort(target_predict_colony[:,kg])
        plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_predict_colony[order,kg],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax) #,vmin=data.g_min[kg],vmax=data.g_max[kg]
        
        plt.title(data.gene_names[kg],fontsize=10)
        plt.axis('off')

        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
    
    plt.savefig(subdirectory_plot_fig + "/" + 'genes_sim_' + mode_reg + kernel + '_' + cond + file_suffix)
    
    
    fig_size = [data.N_genes,1.05]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    margin_top = 1-margin_bottom
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    for kg in range(0,data.N_genes):

        chrt+=1
        plt.subplot(H,W,chrt)
        order = np.argsort(target_colony[:,kg])
        plt.scatter(xx_colony[order],yy_colony[order],ms,c=target_colony[order,kg],cmap='YlGnBu',vmin=min_val,vmax=max_val_vmax)#,vmin=data.g_min[kg],vmax=data.g_max[kg])

        
        plt.axis('off')

        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
        
    plt.savefig(subdirectory_plot_fig + "/" + 'genes_expt' + '_' + cond + file_suffix)


    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    for kg in range(0,data.N_genes):
    
        chrt+=1
        plt.subplot(H,W,chrt)
        indices_pos = target_colony[:,kg]>thresh
        plt.scatter(xx_colony[indices_pos],yy_colony[indices_pos],3,color='b',alpha=0.3,zorder=-1,edgecolors='none')
        
        indices_pos = target_predict_colony[:,kg]>thresh
        plt.scatter(xx_colony[indices_pos],yy_colony[indices_pos],3,color='r',alpha=0.3,zorder=1,edgecolors='none')
        
        plt.axis('off')
    
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
        
    plt.savefig(subdirectory_plot_fig + "/" + 'genes_thresh_expt_sim' + '_' + cond + file_suffix)


    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=10*margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    N_bins_x = 20
    max_val = 6
    for kg in range(0,data.N_genes):
        chrt+=1
        plt.subplot(H,W,chrt)

        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(target_colony[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='b')
        plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color='b',alpha=0.2)
        
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(target_predict_colony[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='r')
        plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color='r',alpha=0.2)

        plt.xticks([])
        plt.yticks([])
        #plt.yticks([min_val,max_val])
        plt.ylim([min_val,max_val])
    
    plt.savefig(subdirectory_plot_fig + "/" + 'profile_expt_sim_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)


if mode_accuracy: 
    f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (ratios) = pickle.load(f)
    f.close()
    
    f = open(subdirectory_data + '/data_regression_av_random' + '_' + cond + ".pickle",'rb')
    (ratios_random_prop) = pickle.load(f)
    f.close()
    
    
    fs2 = 8
    fig_size = [3,4]   
    params = {
              'figure.figsize': fig_size,
              'xtick.labelsize': fs2,
              'ytick.labelsize': fs2,
              }
    plt.rcParams.update(params)
    
    plt.figure()
    
    if mode_accuracy_indices:
        indices = ~np.any(ratios[:,:,1]==0,axis=1)
        plt.errorbar(np.mean(ratios[indices,:,4],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,4],axis=0)),fmt='o',color='k',capsize=3,ms=8,label='')
        plt.errorbar(np.mean(ratios[indices,:,3],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,3],axis=0)),fmt='^',color='g',capsize=3,ms=3)
    else:
        plt.errorbar(np.mean(ratios[:,:,4],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[:,:,4],axis=0)),fmt='o',color='k',capsize=3,ms=8,label='')
        plt.errorbar(np.mean(ratios[:,:,3],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[:,:,3],axis=0)),fmt='^',color='g',capsize=3,ms=3)
    
    
    plt.errorbar(np.mean(ratios_random_prop[:,:,3],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_random_prop[:,:,3],axis=0)),fmt='s',color='grey',capsize=3,ms=3)


    plt.xlim([0,1])
    plt.ylim([-0.5,data.N_genes-0.5])
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.xlabel('precision')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'precision_av_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)


    plt.figure()
    if mode_accuracy_indices:
        indices = ~np.any(ratios[:,:,4]==0,axis=1)
        plt.errorbar(np.mean(ratios[indices,:,4],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,4],axis=0)),fmt='o',color='k',capsize=3,ms=8,label='')
        plt.errorbar(np.mean(ratios[indices,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,1],axis=0)),fmt='^',color='g',capsize=3,ms=3)
    else:
        plt.errorbar(np.mean(ratios[:,:,4],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[:,:,4],axis=0)),fmt='o',color='k',capsize=3,ms=8,label='')
        plt.errorbar(np.mean(ratios[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[:,:,1],axis=0)),fmt='^',color='g',capsize=3,ms=3)
    
        
        
    plt.errorbar(np.mean(ratios_random_prop[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_random_prop[:,:,1],axis=0)),fmt='s',color='grey',capsize=3,ms=3)


    plt.xlim([0,1])
    plt.ylim([-0.5,data.N_genes-0.5])
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.xlabel('sensitivity')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'sensitivity_av_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)
    
    
    plt.figure()
    if mode_accuracy_indices:
        indices = ~np.any(ratios[:,:,4]==0,axis=1)
        
        sensitivity = np.mean(ratios[indices,:,1],axis=0)
        precision = np.mean(ratios[indices,:,3],axis=0)
        
        F1 = 2*sensitivity*precision/(sensitivity+precision)
        
        plt.errorbar(F1,np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[indices,:,1],axis=0)),fmt='^',color='g',capsize=3,ms=3)
    else:
        
        sensitivity = np.mean(ratios[:,:,1],axis=0)
        precision = np.mean(ratios[:,:,3],axis=0)
        
        F1 = 2*sensitivity*precision/(sensitivity+precision)
        
        plt.errorbar(F1,np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios[:,:,1],axis=0)),fmt='^',color='g',capsize=3,ms=3)
    
        
        
    
    #plt.errorbar(np.mean(ratios_random_prop[:,:,1],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_random_prop[:,:,1],axis=0)),fmt='s',color='grey',capsize=3,ms=3)


    plt.xlim([0,1])
    plt.ylim([-0.5,data.N_genes-0.5])
    plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.xlabel('F1 score')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'F1_av_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)
    
    
    
if mode_plot_predict_eval:  
    
    min_val = 0
    max_val = 4
    
    N_bins = 30
    hist_av_all = np.zeros((data.N_genes,N_bins))
    hist_var_all = np.zeros((data.N_genes,N_bins))
    hist_cond_av_all = np.zeros((data.N_genes,N_bins,N_bins))
    for kg in range(0,data.N_genes):
        bins,hist,hist_cond,hist_av,hist_var = fns_plot.calc_prediction_hist(tar_test[:,kg], target_predict[:,kg], N_bins, min_val, max_val)#data.g_min[kg], data.g_max[kg])

        hist_av_all[kg,:] = hist_av
        hist_var_all[kg,:] = hist_var
        hist_cond_av_all[kg,:,:] = hist_cond
    
    
    fig_size = [3,3]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)

    plt.figure()

    for kg in range(0,data.N_genes):
        plt.plot(bins,hist_av_all[kg,:],'-',color=fns_plot.return_colmaps('genes',N_var=data.N_genes)[kg])
    
    plt.plot([0,1e5],[0,1e5],'-',color='k',lw=1)

    plt.xlim([min_val,max_val])
    plt.ylim([min_val,max_val])
    
    plt.xticks([min_val,max_val])
    plt.yticks([min_val,max_val])
    
    plt.xlabel('predicted gene expression')
    plt.ylabel('measured gene expression')

    plt.tight_layout()
    
    
    plt.savefig(subdirectory_plot_fig + "/" + 'av_sg_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)
    
    
    
    
    fig_size = [6,3]  
    fs2 = 6
    params = {
              'figure.figsize': fig_size,
              'xtick.labelsize': fs2,
              'ytick.labelsize': fs2,
              }
    plt.rcParams.update(params)
    
    N_var = data.N_genes
    names = data.gene_names
    
    Corr_all = np.zeros((N_var,N_var,2))
    
    for k1 in range(0,N_var):
        for k2 in range(0,N_var):
            Corr_all[k1,k2,0],_ = fns_plot.calc_pearsoncorr(tar_test[:,k1], tar_test[:,k2])
            Corr_all[k1,k2,1],_ = fns_plot.calc_pearsoncorr(target_predict[:,k1], target_predict[:,k2])
    
    
    H,W = 1,2
    
    plt.figure()
    plt.subplot(H,W,1)
    plt.title('predicted')
    plt.imshow(Corr_all[:,:,1],vmin=-1,vmax=1,cmap='RdBu_r')
    #plt.colorbar()
    
    plt.xticks(np.arange(0,N_var),labels=names,rotation=45)
    #plt.xticks([])
    plt.yticks(np.arange(0,N_var),labels=names)

    
    plt.subplot(H,W,2)
    plt.title('measured')
    plt.imshow(Corr_all[:,:,0],vmin=-1,vmax=1,cmap='RdBu_r')
    #plt.colorbar()
    
    plt.xticks(np.arange(0,N_var),labels=names,rotation=45)
    #plt.yticks(np.arange(0,N_var),labels=names)
    plt.yticks([])
    
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'g1g2_corr_matrix_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)
    
    
    fs2 = 10
    fig_size = [3.4,3]   
    params = {
              'figure.figsize': fig_size,
              'xtick.labelsize': fs2,
              'ytick.labelsize': fs2,
              }
    plt.rcParams.update(params)

    plt.figure()

    plt.plot([-1e5,1e5],[-1e5,1e5],'k',color='k',lw=1)
    plt.plot([0,0],[-1e5,1e5],'--',color='k',lw=0.8)
    plt.plot([-1e5,1e5],[0,0],'--',color='k',lw=0.8)

    plt.scatter(Corr_all[:,:,1],Corr_all[:,:,0],10,color='k')

    plt.xlim([-1,1])
    plt.ylim([-1,1])

    plt.xlabel('predicted correlation')
    plt.ylabel('measured correlation')

    plt.tight_layout()

    plt.savefig(subdirectory_plot_fig + "/" + 'g1g2_corr_matrix_scatter_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)

    
if mode_plot_fates:
    markers_colony_thresh = fns_plot.return_fates(target_colony,data.gene_names,thresh)
    markers_predict_colony = fns_plot.return_fates(target_predict_colony,data.gene_names,thresh)
    
    fs = 10
    fs2 = 5
    fig_size = [3,3]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    N_fates = 6
    ms = 2
    
    fig_size = [14,2.2]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    H,W = 1,N_fates+1
    
    for mode_sim in [0,1]:
        plt.figure()
        chrt=0
        
        chrt+=1
        plt.subplot(H,W,chrt)
        for m in range(N_fates):
            if mode_sim:
                indices = markers_predict_colony[0]==m
            else:
                indices = markers_colony_thresh[0]==m
            plt.scatter(xx_colony[indices],yy_colony[indices],color=fns_plot.return_colmaps('fates')[m],zorder=-m,s=ms)
        
        plt.axis('off')

        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
        
        
        for m in range(N_fates):
            chrt+=1
            plt.subplot(H,W,chrt)
            plt.title(data.fate_names[m])
            if mode_sim:
                indices = markers_predict_colony[0]==m
            else:
                indices = markers_colony_thresh[0]==m
            plt.scatter(xx_colony[:],yy_colony[:],color='lightgrey',s=3,zorder=-100)
            plt.scatter(xx_colony[indices],yy_colony[indices],color=fns_plot.return_colmaps('fates')[m],s=3,zorder=-m)
            
            plt.axis('off')
    
            plt.xticks([])
            plt.yticks([])
            
            plt.xlim([-lim,lim])
            plt.ylim([-lim,lim])
        
        plt.tight_layout()
        plt.savefig(subdirectory_plot_fig + "/" + 'fates_sim' + str(mode_sim) + '_colony_' + mode_reg + kernel + '_' + cond + file_suffix)
    
    