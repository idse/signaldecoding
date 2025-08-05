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
mode_input = 's'
mode_classbal = 0


mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub' #
    conditions = ['B10','B200'] #,'B50'
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

name = 'regression'
mode_reg = 'MLP'

    
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''
    
subdirectory_plot = directory + '/analysis_regression_sweep_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z)
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)


train_size = 3#0.5

kernel = 'relu'

N_run = 10

for cond in conditions:
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    if mode_input == 's':
        feature = data.signals
    elif mode_input == 'r':
        feature = data.metricdist
    elif mode_input == 'n':
        feature = data.nuc_feat
    
    target = data.genes[:,:,:]

    if len(feature.shape)==2:
        feature = feature[:,:,np.newaxis]
    if len(target.shape)==2:
        target = target[:,:,np.newaxis]
    
    thresh = 1
    N_layers_all = [1,2,3,4,5]
    N_nodes_all = [1,2,4,6,8,10,15,20]
    
    N_N_layers = len(N_layers_all)
    N_N_nodes = len(N_nodes_all)
    
    ratios = np.zeros((N_N_layers,N_N_nodes,N_run,data.N_genes,2))
    ratios_conf = np.zeros((N_N_layers,N_N_nodes,N_run,data.N_genes,2))
    accuracy = np.zeros((N_N_layers,N_N_nodes,N_run,data.N_genes))
    MIs = np.zeros((N_N_layers,N_N_nodes,N_run,data.N_genes))
    
    for it_layer,N_layers in enumerate(N_layers_all):
        print('------')
        print(it_layer)
        for it_nodes,N_nodes in enumerate(N_nodes_all):
            print(it_nodes)
            hidden_layer_sizes = (N_nodes,)*N_layers
            reg = fns_plot.return_reg(mode_reg,kernel,hidden_layer_sizes=hidden_layer_sizes)
            
            for it in range(N_run):
                
                if np.mod(it,10)==0:
                    print(it)
                
                if mode_classbal==1:
                    feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_classbal(data,feature,target,train_size=train_size)
                elif mode_classbal==2:
                    feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_classbal_resample(data,feature,target,train_size=train_size)
                else:
                    feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_v2(data,feature,target,train_size=train_size)
                
                if mode_z:
                    feat_train_z,mean_feat_train,stdev_feat_train = fns_plot.do_zscore(feat_train)
                    tar_train_z,mean_tar_train,stdev_tar_train = fns_plot.do_zscore(tar_train)
                    feat_test_z,_,_ = fns_plot.do_zscore(feat_test,mean_feat_train,stdev_feat_train)
                    
                    reg.fit(feat_train_z,tar_train_z)
                    target_predict = fns_plot.undo_zscore(reg.predict(feat_test_z),mean_tar_train,stdev_tar_train)
                else:
                    reg.fit(feat_train,tar_train)
                    target_predict = reg.predict(feat_test)
                
                f = open(subdirectory_plot_data+'/data_regression_' + mode_reg + kernel + '_' + cond + '_run' + str(it) + ".pickle",'wb')
                pickle.dump((reg,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train), f)
                f.close()
                
                for kg in range(0,data.N_genes):
                    
                    pos_expt = tar_test[:,kg]>=thresh
                    pos_sim = target_predict[:,kg]>=thresh
                    
                    from sklearn.metrics import confusion_matrix
                    TN, FP, FN, TP = confusion_matrix(pos_expt, pos_sim).ravel()
                    #print(confusion_matrix(pos_expt, pos_sim).ravel())
                    
        
                    ratios[it_layer,it_nodes,it,kg,0] = fns_plot.diff_zero(TN,(TN+FP))
                    ratios[it_layer,it_nodes,it,kg,1] = fns_plot.diff_zero(TP,(TP+FN))
                    
                    ratios_conf[it_layer,it_nodes,it,kg,0] = fns_plot.diff_zero(TN,(TN+FN))
                    ratios_conf[it_layer,it_nodes,it,kg,1] = fns_plot.diff_zero(TP,(TP+FP))
                    
                    accuracy[it_layer,it_nodes,it,kg] = fns_plot.diff_zero((TP+TN),(TN + FP + FN + TP))

                    MIs[it_layer,it_nodes,it,kg] = fns_plot.calc_MI_sklearn(tar_test[:,kg],target_predict[:,kg])
            
    f = open(subdirectory_plot_data+'/data_regression_sweep_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
    pickle.dump((ratios,ratios_conf,accuracy,MIs), f)
    f.close()
    
        
        
        