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

mode_z = 1
mode_input = 's'
mode_classbal = 0


#standard
if mode_classbal:
    suffix_classbal = '_classbal'
else:
    suffix_classbal = ''
    
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_3x10'
subdirectory_plot_data = subdirectory_plot + '/data'

subdirectory_plot_fig = directory + '/SI_fig_accuracy_continuous'
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

f = open(subdirectory_plot_data + '/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'rb')
(reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test, target_predict_train, target_predict,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
f.close()

it = 0
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

min_val = 0

H,W=9,data.N_genes  
fig_size = [13,8]
#[17,9*1.3]   
fs2 = 6
params = {
          'figure.figsize': fig_size,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

thresholds = np.ones(data.N_genes)
max_val = 6

ticks_on = 0

fig = plt.figure()
chrt=0
fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=margin_top, wspace=wspace, hspace=hspace)

corrs = np.zeros((data.N_genes,2))
for kg in range(0,data.N_genes):
    chrt+=1
    plt.subplot(H,W,chrt)
    plt.title(data.gene_names[kg],fontsize=10)
    
    N_max = target_predict.shape[0]
    N_plot = 4000
    indices = np.random.randint(0,N_max,N_plot)
    fns_plot.scatter_density(target_predict[indices,kg],tar_test[indices,kg])
    
    corrs[kg,0],_ = fns_plot.calc_pearsoncorr(target_predict[:,kg], tar_test[:,kg])
    corrs[kg,1],_ = fns_plot.calc_spearmancorr(target_predict[:,kg], tar_test[:,kg])
    
    max_val = np.nanpercentile(tar_test[indices,kg],99.9)
    
    plt.plot([0,max_val],[0,max_val],'-',color='grey',lw=0.5)

    
    plt.xlim([min_val,max_val])
    plt.ylim([min_val,max_val])
    
    if ticks_on:
        plt.xticks([min_val,max_val])
        plt.yticks([min_val,max_val])
    else:
        plt.xticks([])
        plt.yticks([])
    

   
N_bins = 30
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    
    max_val = np.nanpercentile(tar_test[indices,kg],99.9)
    
    bins,hist,hist_cond,hist_av,hist_var = fns_plot.calc_prediction_hist(tar_test[:,kg], target_predict[:,kg], N_bins, min_val, max_val)#data.g_min[kg], data.g_max[kg])
    
    plt.imshow(np.rot90(hist_cond,k=1),extent=[min_val,max_val,min_val,max_val],cmap='YlGnBu',vmin=0,vmax=0.4)
    
    
    plt.plot(bins,hist_av,'-k',lw=1.5)
    plt.plot(bins,hist_av-np.sqrt(hist_var),':k',lw=1)
    plt.plot(bins,hist_av+np.sqrt(hist_var),':k',lw=1)
    
    plt.plot([0,max_val],[0,max_val],'-',color='grey',lw=0.5)
    
    plt.xlim([min_val,max_val])
    plt.ylim([min_val,max_val])
    
    if ticks_on:
        plt.xticks([min_val,max_val])
        plt.yticks([min_val,max_val])
    else:
        plt.xticks([])
        plt.yticks([])

N_bins_x = 20
radial_av_all = np.zeros((data.N_genes,N_bins_x,2))
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)

    bins_x,mean_x_expt,var_x,P_x = fns_plot.calc_profile_meanvar(target_colony[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
    radial_av_all[kg,:,0] = mean_x_expt[:,0]
    
    bins_x,mean_x_sim,var_x,P_x = fns_plot.calc_profile_meanvar(target_predict_colony[:,kg][np.newaxis,:,np.newaxis],rr_colony[np.newaxis,:,np.newaxis],N_bins_x,data.r_max)
    radial_av_all[kg,:,1] = mean_x_sim[:,0]
    
    max_val_rad = 1.1*np.nanmax(mean_x_expt)
    
    plt.plot([0,max_val_rad],[0,max_val_rad],'-',color='grey',lw=0.5)
    plt.scatter(mean_x_sim,mean_x_expt,color='k',s=10)

    plt.xlim([min_val,max_val_rad])
    plt.ylim([min_val,max_val_rad])
    
    if ticks_on:
        plt.xticks([min_val,max_val])
        plt.yticks([min_val,max_val])
    else:
        plt.xticks([])
        plt.yticks([])
    
    
    
"""
N_bins = 20
for kg in range(0,data.N_genes):

    chrt+=1
    plt.subplot(H,W,chrt)
    
    max_val = np.nanpercentile(tar_test[indices,kg],99.9)
    edges = np.linspace(min_val,max_val,N_bins)
    
    plt.hist(tar_test[:,kg],edges,density=True,color='b',alpha=0.5)
    plt.hist(target_predict[:,kg],edges,density=True,color='r',histtype='step')
    
    plt.xlim([min_val,max_val])
    plt.xticks([])
    plt.yticks([])
    
    plt.yscale('log')
"""
    
plt.tight_layout()

plt.savefig(subdirectory_plot_fig + "/" + 'SI_fig_genes_cont_overview' + '_' + cond + file_suffix)
  


fs2 = 11
fig_size = [9,3]   
params = {
          'figure.figsize': fig_size,
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params)

file_suffix = '.pdf'

H,W = 1,3
ticks_on = 1
max_val = 6

plt.figure()
plt.subplot(H,W,1)

for kg in range(0,data.N_genes):
    #plt.scatter(mean_x_sim,mean_x_expt,color='k',s=10)
    plt.scatter(radial_av_all[kg,:,1],radial_av_all[kg,:,0],s=10,color=fns_plot.return_colmaps('genes',N_var=data.N_genes)[kg])

plt.plot([0,1e5],[0,1e5],'-',color='grey',lw=0.5)

plt.xlim([min_val,max_val])
plt.ylim([min_val,max_val])

if ticks_on:
    plt.xticks([min_val,max_val])
    plt.yticks([min_val,max_val])
else:
    plt.xticks([])
    plt.yticks([])

plt.xlabel('pred. radial profile (a.u.)')
plt.ylabel('exp. radial profile (a.u.)')


plt.subplot(H,W,2)

#analyze with same binning
N_bins = 30
hist_av_all = np.zeros((data.N_genes,N_bins))
hist_var_all = np.zeros((data.N_genes,N_bins))
hist_cond_av_all = np.zeros((data.N_genes,N_bins,N_bins))

max_val = 6
for kg in range(0,data.N_genes):
    
    
    bins,hist,hist_cond,hist_av,hist_var = fns_plot.calc_prediction_hist(tar_test[:,kg], target_predict[:,kg], N_bins, min_val, max_val)#data.g_min[kg], data.g_max[kg])
    
    hist_av_all[kg,:] = hist_av
    hist_var_all[kg,:] = hist_var
    hist_cond_av_all[kg,:,:] = hist_cond


for kg in range(0,data.N_genes):
    plt.plot(bins,hist_av_all[kg,:],'-',color=fns_plot.return_colmaps('genes',N_var=data.N_genes)[kg])

plt.plot([0,1e5],[0,1e5],'-',color='grey',lw=0.5)

#max_val = 1.1*np.nanmax(hist_av_all[:,:])
plt.xlim([min_val,max_val])
plt.ylim([min_val,max_val])

if ticks_on:
    plt.xticks([min_val,max_val])
    plt.yticks([min_val,max_val])
else:
    plt.xticks([])
    plt.yticks([])
    
plt.xlabel('pred. SC values (a.u.)')
plt.ylabel('exp. SC values (a.u.)')


plt.subplot(H,W,3)

plt.imshow(np.rot90(np.nanmean(hist_cond_av_all,axis=0),k=1),extent=[min_val,max_val,min_val,max_val],cmap='YlGnBu',vmin=0,vmax=0.3)
hist_av_all_av = np.nanmean(hist_av_all,axis=0)
hist_var_all_av = np.nanmean(hist_var_all,axis=0)

#"""
plt.plot(bins,hist_av_all_av,'-k')
plt.plot(bins,hist_av_all_av-np.sqrt(hist_var_all_av),':k')
plt.plot(bins,hist_av_all_av+np.sqrt(hist_var_all_av),':k')
#"""

plt.plot([0,1e5],[0,1e5],'-',color='grey',lw=0.5)

plt.xlim([min_val,max_val])
plt.ylim([min_val,max_val])

if ticks_on:
    plt.xticks([min_val,max_val])
    plt.yticks([min_val,max_val])
else:
    plt.xticks([])
    plt.yticks([])


plt.xlabel('pred. SC values (a.u.)')
plt.ylabel('exp. SC values (a.u.)')

plt.tight_layout()


plt.savefig(subdirectory_plot_fig + "/" + 'SI_fig_av_sg_' + mode_reg + kernel + '_' + cond + file_suffix)



N_var = data.N_genes
names = data.gene_names

Corr_all = np.zeros((N_var,N_var,2))

for k1 in range(0,N_var):
    for k2 in range(0,N_var):
        Corr_all[k1,k2,0],_ = fns_plot.calc_pearsoncorr(tar_test[:,k1], tar_test[:,k2])
        Corr_all[k1,k2,1],_ = fns_plot.calc_pearsoncorr(target_predict[:,k1], target_predict[:,k2])


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

