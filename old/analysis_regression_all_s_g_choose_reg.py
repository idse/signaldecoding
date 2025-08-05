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

it_reg = 1


mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub' #
    conditions = ['B50']
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
    
subdirectory_plot = directory + '/analysis_regression_' + mode_input + 'g_multi' + suffix_classbal + '_z' + str(mode_z) + '_LS_2x10_2_2x10_v2'
subdirectory_plot_data = subdirectory_plot + '/data'
subdirectory_plot_data_reg = subdirectory_plot + '/data_many_reg'


train_size = 3
kernel = 'relu'

for cond in conditions:
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    f = open(subdirectory_plot_data_reg + '/data_regression_' + mode_reg + kernel + '_' + cond + '_run' + str(it_reg) + ".pickle",'rb')
    (reg) = pickle.load(f)
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
        
        target_predict_train = fns_plot.undo_zscore(reg.predict(feat_train_z),mean_tar_train,stdev_tar_train)
        target_predict = fns_plot.undo_zscore(reg.predict(feat_test_z),mean_tar_train,stdev_tar_train)
    else:
        
        target_predict_train = reg.predict(feat_train)
        target_predict = reg.predict(feat_test)
        
        mean_feat_train,stdev_feat_train = 0,0
        mean_tar_train,stdev_tar_train = 0,0
    
    f = open(subdirectory_plot_data+'/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
    pickle.dump((reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test, target_predict_train, target_predict,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train), f)
    f.close()

    for it in range(data.N_sys):
        indices = ~np.isnan(data.X[it,:,0].ravel())
        xx_colony = data.X[it,indices,0]
        yy_colony = data.X[it,indices,1]
        rr_colony = data.metricdist[it,indices]
        feature_colony = feature[it,indices,:]
        target_colony = target[it,indices,:]
        markers_colony = data.markers[it,indices]
        
        if mode_z:
            feature_colony_z,_,_ = fns_plot.do_zscore(feature_colony,mean_feat_train,stdev_feat_train)
            target_predict_colony = fns_plot.undo_zscore(reg.predict(feature_colony_z),mean_tar_train,stdev_tar_train)
        else:
            target_predict_colony = reg.predict(feature_colony)
          
        f = open(subdirectory_plot_data+'/data_colony_it' + str(it) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
        pickle.dump((xx_colony, yy_colony, rr_colony, markers_colony, target_colony, target_predict_colony), f)
        f.close()
    
    