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
mode_regression = 1

N_run = 100
thresh = 1

mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    conditions = ['B50']#,'B200']#['B10','B50','B200']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

if mode_regression:
    name = 'regression'
    mode_reg = 'SVR'
else:
    name = 'classification'
    mode_reg = 'SVC'
    
subdirectory_plot = directory + '/analysis_' + name + '_sg_single_z' + str(mode_z) + '_SVR'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)


train_size = 0.5

kernel = 'rbf'

reg = fns_plot.return_reg(mode_reg,kernel)

for cond in conditions:
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()

    feature = data.signals
    if mode_regression:
        target = data.genes[:,:,:]
    else:
        target = data.genes[:,:,:]>1

    if len(feature.shape)==2:
        feature = feature[:,:,np.newaxis]
    if len(target.shape)==2:
        target = target[:,:,np.newaxis]
        
        
    ratios = np.zeros((N_run,data.N_genes,5))
    
    for it in range(N_run):
        
        if np.mod(it,10)==0:
            print(it)
        
        feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test = fns_plot.test_train_split_cells_v2(data,feature,target,train_size=train_size)
        
        
        if mode_z:  
            feat_train_z,mean_feat_train,stdev_feat_train = fns_plot.do_zscore(feat_train)
            feat_test_z,_,_ = fns_plot.do_zscore(feat_test)
            
            if mode_regression:
                tar_train_z,mean_tar_train,stdev_tar_train = fns_plot.do_zscore(tar_train)
            
        if not mode_z or not mode_regression:
            mean_feat_train,stdev_feat_train = 0,0
            mean_tar_train,stdev_tar_train = 0,0
            
            
        target_predict_train = np.zeros(tar_train.shape)
        target_predict = np.zeros(tar_test.shape)
        
        for kg in range(0,data.N_genes):
            
            if mode_z and mode_regression:
                reg.fit(feat_train_z,tar_train_z[:,kg])
    
                target_predict_train[:,kg] = fns_plot.undo_zscore(reg.predict(feat_train_z)[:,np.newaxis],np.array([mean_tar_train[kg]]),np.array([stdev_tar_train[kg]]))[:,0]
                target_predict[:,kg] = fns_plot.undo_zscore(reg.predict(feat_test_z)[:,np.newaxis],np.array([mean_tar_train[kg]]),np.array([stdev_tar_train[kg]]))[:,0]
            elif mode_z and not mode_regression:
                reg.fit(feat_train_z,tar_train[:,kg])
                
                target_predict_train[:,kg] = reg.predict(feat_train_z)
                target_predict[:,kg] = reg.predict(feat_test_z)
            else:
                reg.fit(feat_train,tar_train[:,kg])
                
                target_predict_train[:,kg] = reg.predict(feat_train)
                target_predict[:,kg] = reg.predict(feat_test)
    
            pos_expt = tar_test[:,kg]>=thresh
            pos_sim = target_predict[:,kg]>=thresh
            
            from sklearn.metrics import confusion_matrix
            TN, FP, FN, TP = confusion_matrix(pos_expt, pos_sim).ravel()
            #print(confusion_matrix(pos_expt, pos_sim).ravel())
            

            ratios[it,kg,0] = fns_plot.diff_zero(TN,(TN+FP))
            ratios[it,kg,1] = fns_plot.diff_zero(TP,(TP+FN))
            
            ratios[it,kg,2] = fns_plot.diff_zero(TN,(TN+FN))
            ratios[it,kg,3] = fns_plot.diff_zero(TP,(TP+FP))
            
            ratios[it,kg,4] = fns_plot.diff_zero((TP+TN),(TN + FP + FN + TP))

    f = open(subdirectory_plot_data+'/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
    pickle.dump((ratios), f)
    f.close()
    
            
            
            