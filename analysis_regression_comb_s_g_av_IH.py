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

mode_z = 1
mode_classbal = 0

N_run = 10
N_split = 10

mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    #'_withSmad2_reorder_test'
    conditions = ['B50']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

name = 'regression'
mode_reg = 'MLP'

hidden_layer_sizes = (10,10,10)

train_size = 0.5
kernel = 'relu'
    
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''
    
subdirectory_plot = directory + '/analysis_regression_' + 'sg_comb' + suffix_classbal + '_z' + str(mode_z) + '_3x10'

if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)




reg = fns_plot.return_reg(mode_reg,kernel,hidden_layer_sizes=hidden_layer_sizes)

for cond in conditions:
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    N_var = data.N_signals
    
    target = data.genes[:,:,:]
    
    MI_comb = np.zeros((N_var,data.N_genes,N_split,N_run))
    MI_comb_disc = np.zeros((N_var,data.N_genes,N_split,N_run))
    MI_individ = np.zeros((N_var,data.N_genes,N_split,N_run))
    MI_individ_disc = np.zeros((N_var,data.N_genes,N_split,N_run))
    
    thresh = 1
    ratios = np.zeros((N_var,data.N_genes,2,N_split,N_run))
    ratios_conf = np.zeros((N_var,data.N_genes,2,N_split,N_run))
    accuracy = np.zeros((N_var,data.N_genes,N_split,N_run))
    
    for N_inc in range(1,N_var+1):
        print('N_inc=' + str(N_inc))
        feature = data.signals[:,:,:N_inc]
    
        if len(feature.shape)==2:
            feature = feature[:,:,np.newaxis]
            
        for it in range(N_split):
            if np.mod(it,2)==0:
                print(it)
            if mode_classbal==1:
                feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_classbal(data,feature,target,train_size=train_size)
            elif mode_classbal==2:
                feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_classbal_resample(data,feature,target,train_size=train_size)
            else:
                feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_v2(data,feature,target,train_size=train_size)
            
            for it_run in range(N_run):
                
                if mode_z:
                    feat_train_z,mean_feat_train,stdev_feat_train = fns_plot.do_zscore(feat_train)
                    tar_train_z,mean_tar_train,stdev_tar_train = fns_plot.do_zscore(tar_train)
                    feat_test_z,_,_ = fns_plot.do_zscore(feat_test,mean_feat_train,stdev_feat_train)
                    
                    reg.fit(feat_train_z,tar_train_z)
                    
                    target_predict_train = fns_plot.undo_zscore(reg.predict(feat_train_z),mean_tar_train,stdev_tar_train)
                    target_predict = fns_plot.undo_zscore(reg.predict(feat_test_z),mean_tar_train,stdev_tar_train)
                else:
                    reg.fit(feat_train,tar_train)
                    
                    target_predict_train = reg.predict(feat_train)
                    target_predict = reg.predict(feat_test)
                    
                    mean_feat_train,stdev_feat_train = 0,0
                    mean_tar_train,stdev_tar_train = 0,0
                    
                k = N_inc-1
                for kg in range(data.N_genes):
                    #information
                    MI_individ[k,kg,it,it_run] = fns_plot.calc_MI_sklearn(feat_test[:,k],tar_test[:,kg])
                    MI_individ_disc[k,kg,it,it_run] = fns_plot.calc_MI_sklearn(feat_test[:,k],tar_test[:,kg]>1)
                    
                    MI_comb[k,kg,it,it_run] = fns_plot.calc_MI_sklearn(tar_test[:,kg],target_predict[:,kg])
                    MI_comb_disc[k,kg,it,it_run] = fns_plot.calc_MI_sklearn(tar_test[:,kg]>1,target_predict[:,kg]>1)
                    
                    
                    #accuracy
                    pos_expt = tar_test[:,kg]>=thresh
                    pos_sim = target_predict[:,kg]>=thresh
                    
                    from sklearn.metrics import confusion_matrix
                    TN, FP, FN, TP = confusion_matrix(pos_expt, pos_sim).ravel()
    
                    ratios[k,kg,0,it,it_run] = fns_plot.diff_zero(TN,(TN+FP))
                    ratios[k,kg,1,it,it_run] = fns_plot.diff_zero(TP,(TP+FN))
                    
                    ratios_conf[k,kg,0,it,it_run] = fns_plot.diff_zero(TN,(TN+FN))
                    ratios_conf[k,kg,1,it,it_run] = fns_plot.diff_zero(TP,(TP+FP))
                    
                    accuracy[k,kg,it,it_run] = fns_plot.diff_zero((TP+TN),(TN + FP + FN + TP))

    f = open(subdirectory_plot_data+'/data_regression_MI_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
    pickle.dump((MI_individ,MI_individ_disc,MI_comb,MI_comb_disc), f)
    f.close()    
    
    f = open(subdirectory_plot_data+'/data_regression_accuracy_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
    pickle.dump((ratios,ratios_conf,accuracy), f)
    f.close()
            
            
            