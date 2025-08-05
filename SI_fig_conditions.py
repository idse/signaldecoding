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

mode_plot_predict_patterns = 1
mode_plot_fates = 1
mode_plot_boundaries = 1
mode_accuracy = 1
mode_plot_predict_eval = 0

mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    cond = 'B200'
    it_cond = 2
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

mode_z = 1
mode_input = 's'
mode_classbal = 0


#standard
if mode_classbal:
    suffix_classbal = '_classbal'
else:
    suffix_classbal = ''
    
    
condition_train = 'B50'
subdirectory_train = directory + '/analysis_regression_sg_multi_z' + str(mode_z) + '_3x10_v2'
subdirectory_train_data = subdirectory_train + '/data'

subdirectory_plot = subdirectory_train + '/analysis_crosscond_train_' + condition_train + '_pred_' + cond
subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/SI_fig_conditions'
if not os.path.exists(subdirectory_plot_fig):
    os.mkdir(subdirectory_plot_fig)



thresh = 1

mode_reg = 'MLP'
kernel = 'relu'
bottleneck_layer = 3

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

"""
f = open(subdirectory_plot_data + '/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test, target_predict_train, target_predict,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
f.close()
"""

it = 0
f = open(subdirectory_plot_data + '/data_colony_it' + str(it) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(xx_colony, yy_colony, rr_colony, markers_colony, target_colony, target_predict_colony) = pickle.load(f)
f.close()

f = open(subdirectory_train_data + '/data_colony_it' + str(it) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(xx_colony, yy_colony, rr_colony, markers_colony, target_colony, target_predict_colony_within) = pickle.load(f)
f.close()


margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 1-0.2
hspace = 0.001
wspace = 0.001

lim = 360
ms = 0.15


if mode_plot_predict_patterns:
    
    min_val = 0
    max_val = 9
    max_val_vmax = 5

    H,W=1,data.N_genes  

    fig_size = [17,1.05]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    margin_top = 1-margin_bottom
    
    N_bins_x = 20
    radial_av_all = np.zeros((data.N_genes,N_bins_x,3))
    
    fig = plt.figure()
    chrt=0
    fig.subplots_adjust(left=margin_left, bottom=10*margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    
    max_val = 6
    for kg in range(0,data.N_genes):
        chrt+=1
        plt.subplot(H,W,chrt)

        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(target_colony[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='b')
        #plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color='b',alpha=0.2)
        radial_av_all[kg,:,0] = mean_x[:,0]
        
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(target_predict_colony[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='r')
        #plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color='r',alpha=0.2)
        radial_av_all[kg,:,1] = mean_x[:,0]
        
        bins_x,mean_x,var_x,P_x = fns_plot.calc_profile_meanvar(target_predict_colony_within[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
        plt.plot(bins_x,mean_x,color='g')
        #plt.fill_between(bins_x, mean_x[:,0]-np.sqrt(var_x[:,0]), mean_x[:,0]+np.sqrt(var_x[:,0]),color='r',alpha=0.2)
        radial_av_all[kg,:,2] = mean_x[:,0]
        
        plt.xticks([])
        plt.yticks([])
        #plt.yticks([min_val,max_val])
        plt.ylim([min_val,max_val])
    
    plt.savefig(subdirectory_plot_fig + "/" + 'profile_expt_sim_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)
  
    
    fig_size = [3,3]   
    params = {
              'figure.figsize': fig_size,
              'xtick.labelsize': fs2,
              'ytick.labelsize': fs2,
              }
    plt.rcParams.update(params)
    
    max_val = 3
    
    plt.figure()
    for kg in range(0,data.N_genes):
        #plt.scatter(mean_x_sim,mean_x_expt,color='k',s=10)
        plt.scatter(radial_av_all[kg,:,1],radial_av_all[kg,:,0],s=10,color='r')
        plt.scatter(radial_av_all[kg,:,2],radial_av_all[kg,:,0],s=10,color='g')

    plt.plot([0,1e5],[0,1e5],'-',color='grey',lw=0.5)

    plt.xlim([min_val,max_val])
    plt.ylim([min_val,max_val])

    plt.xticks([min_val,max_val])
    plt.yticks([min_val,max_val])

    plt.xlabel('pred. radial profile (a.u.)')
    plt.ylabel('exp. radial profile (a.u.)')
    
    plt.tight_layout()
    
    plt.savefig(subdirectory_plot_fig + "/" + 'profile_expt_sim_scatter_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)
  



    
if mode_plot_fates:
    markers_colony_thresh = fns_plot.return_fates(target_colony,data.gene_names,thresh)
    markers_predict_colony = fns_plot.return_fates(target_predict_colony,data.gene_names,thresh)
    markers_predict_colony_within = fns_plot.return_fates(target_predict_colony_within,data.gene_names,thresh)
    
    fs = 10
    fs2 = 5
    fig_size = [3,3]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    N_fates = 6
    ms = 2
    
    fig_size = [6,2.2]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    H,W = 1,3
    
    margin_top = 1-0.2
    
    fig = plt.figure()
    fig.subplots_adjust(left=margin_left, bottom=10*margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)
    
    chrt=0
    for mode_sim in [0,1,2]:
        chrt+=1
        plt.subplot(H,W,chrt)
        for m in range(N_fates):
            if mode_sim==0:
                indices = markers_predict_colony==m
            elif mode_sim==1:
                indices = markers_predict_colony_within==m
            elif mode_sim==2:
                indices = markers_colony_thresh==m
            plt.scatter(xx_colony[indices],yy_colony[indices],color=fns_plot.return_colmaps('fates')[m],s=ms,cmap='tab10',vmin=0,vmax=9,zorder=-m)
        
        plt.axis('off')

        plt.xticks([])
        plt.yticks([])
        
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])

    plt.savefig(subdirectory_plot_fig + "/" + 'fates_colony_' + mode_reg + kernel + '_' + cond + file_suffix)
    
    
    
    
if mode_plot_boundaries:   
    fig_size = [3,2.8]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    
    colors = ['r','g']

    plt.figure()
    plt.plot([0,350],[0,350],'-',color='grey')
    
    for mode_within in [0,1]:
        directory_base = directory + '/analysis_regression_sg_multi_z1_3x10_v2'
        if mode_within==0:
            suffix_file = '_across'
        elif mode_within==1:
            suffix_file = '_within'
        
        
        f = open(directory_base+'/data_boundaries' + suffix_file + '_' + mode_reg + kernel + ".pickle",'rb')
        boundaries_all = pickle.load(f)
        f.close()

        boundaries = boundaries_all[it_cond,:,:,:]

        plt.plot(boundaries[0,:,1],boundaries[0,:,0],'o',color=colors[mode_within],label=suffix_file[1:],clip_on=False)
        plt.plot(boundaries[1,:,1],boundaries[1,:,0],'s',color=colors[mode_within],clip_on=False)
        
    plt.xlim([0,350]) 
    plt.ylim([0,350]) 

    plt.xticks([0,100,200,300,350]) 
    plt.yticks([0,100,200,300,350]) 

    plt.legend()

    plt.xlabel('predicted boundary location')
    plt.ylabel('measured boundary location')
        
    plt.tight_layout()
    plt.savefig(subdirectory_plot_fig + "/" + 'SI_fates_boundaries_' + mode_reg + kernel + '_' + cond + file_suffix_pdf)


if mode_accuracy: 
    directory_base = directory + '/analysis_regression_sg_multi_z1_3x10_v2'

    subdirectory_plot = directory_base + '/analysis_crosscond_train_B50_pred_' + cond
    subdirectory_plot_data = subdirectory_plot + '/data'
    
    f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (ratios_across) = pickle.load(f)
    f.close()
    
    subdirectory_plot = directory_base
    subdirectory_plot_data = subdirectory_plot + '/data'
    
    f = open(subdirectory_plot_data + '/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
    (ratios_within) = pickle.load(f)
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
    
    fig_size = [3,4]   
    params = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)

    names = ['specificity','sensitivity','negative predictive value','precision','accuracy']

    for mode in [0,1,2,3,4]:
        name = names[mode]

        plt.figure()
        
        plt.errorbar(np.mean(ratios_random_prop[:,:,mode],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_random_prop[:,:,mode],axis=0)),fmt='s',color='grey',capsize=3,ms=3)


        indices = ~np.any(ratios_across[:,:,mode]==0,axis=1)
        plt.errorbar(np.mean(ratios_across[indices,:,mode],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_across[indices,:,mode],axis=0)),fmt='o',color='r',capsize=3,ms=3)
        
        indices = ~np.any(ratios_within[:,:,mode]==0,axis=1)
        plt.errorbar(np.mean(ratios_within[indices,:,mode],axis=0),np.arange(data.N_genes),xerr=np.sqrt(np.var(ratios_within[indices,:,mode],axis=0)),fmt='s',color='g',capsize=3,ms=3)


        plt.xlim([0,1])
        plt.ylim([-0.5,data.N_genes-0.5])
        plt.yticks(np.arange(data.N_genes),labels=data.gene_names)
        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        
        plt.xlabel(name)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(subdirectory_plot_fig + "/" + 'within_vs_across_fig_posneg_' + name + '_' + cond + file_suffix_pdf)



    