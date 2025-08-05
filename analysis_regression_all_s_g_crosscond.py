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

mode_av = 1
N_run = 100

mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub'
    condition_train = 'B50'
    conditions = ['B10','B50','B200']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

mode_reg = 'MLP'
kernel = 'relu'
    
subdirectory_train = directory + '/analysis_regression_sg_multi_z' + str(mode_z) + '_3x10_v2'
subdirectory_train_data = subdirectory_train + '/data'


for cond in conditions:
    print(cond)
    
    subdirectory_plot = subdirectory_train + '/analysis_crosscond_train_' + condition_train + '_pred_' + cond
    if not os.path.exists(subdirectory_plot):
        os.mkdir(subdirectory_plot)
    subdirectory_plot_data = subdirectory_plot + '/data'
    if not os.path.exists(subdirectory_plot_data):
        os.mkdir(subdirectory_plot_data)
    
    f = open(subdirectory_train_data + '/data_regression_' + mode_reg + kernel + '_' + condition_train + ".pickle",'rb')
    (reg, _, _, _, _, _, _, _, _, _, _,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
    f.close()

    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    feature = data.signals
    target = data.genes[:,:,:]

    if len(feature.shape)==2:
        feature = feature[:,:,np.newaxis]
    if len(target.shape)==2:
        target = target[:,:,np.newaxis]
    
    feature_clean,target_clean,metricdist_clean,markers_clean = fns_plot.clean_data_full(data,feature,target)

    if mode_z:
        feat_z,_,_ = fns_plot.do_zscore(feature_clean,mean_feat_train,stdev_feat_train)
        target_predict = fns_plot.undo_zscore(reg.predict(feat_z),mean_tar_train,stdev_tar_train)
    else:
        target_predict = reg.predict(feature_clean)
        
        mean_feat_train,stdev_feat_train = 0,0
        mean_tar_train,stdev_tar_train = 0,0
    
    
    feat_train = None
    feat_test = feature_clean
    tar_train = None
    tar_test = target_clean
    metricdist_train = None
    metricdist_test = metricdist_clean
    markers_train = None
    markers_test = markers_clean
    target_predict_train = None
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
            feature_colony_z,mean_train,stdev_train = fns_plot.do_zscore(feature_colony,mean_feat_train,stdev_feat_train)
            target_predict_colony = fns_plot.undo_zscore(reg.predict(feature_colony_z),mean_tar_train,stdev_tar_train)
        else:
            target_predict_colony = reg.predict(feature_colony)
          
        f = open(subdirectory_plot_data+'/data_colony_it' + str(it) + '_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
        pickle.dump((xx_colony, yy_colony, rr_colony, markers_colony, target_colony, target_predict_colony), f)
        f.close()
    
        
    if mode_av:
        
        thresh = 1
        ratios = np.zeros((N_run,data.N_genes,5))
        
        for it in range(N_run):
            
            if np.mod(it,10)==0:
                print(it)
                
            f = open(subdirectory_train_data + '/data_regression_' + mode_reg + kernel + '_' + condition_train + '_run' + str(it) + ".pickle",'rb')
            (reg,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train) = pickle.load(f)
            f.close()
                
            if mode_z:
                target_predict = fns_plot.undo_zscore(reg.predict(feat_z),mean_tar_train,stdev_tar_train)
            else:
                target_predict = reg.predict(feature_clean)
                
            for kg in range(0,data.N_genes):
                
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
        
        