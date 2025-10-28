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

mode_single = 1
mode_colony = 1
mode_av = 1
N_run = 100



mode_expt = 1
if mode_expt:
    directory = 'data_expt_20_scaled_norm_bgsub' #
    conditions = ['B50']#,'B200']#['B10','B50','B200']
else:
    directory = 'data_sim_v2'
    conditions = ['0']
print(directory)

subdirectory_data = directory + '/data'

name = 'regression'
mode_reg = 'MLP'
hidden_layer_sizes = (10,10,10)

    
if mode_classbal==1:
    suffix_classbal = '_classbal'
elif mode_classbal==2:
    suffix_classbal = '_classbalresample'
else:
    suffix_classbal = ''
    
ks = 0
    
subdirectory_plot = directory + '/analysis_regression_singlesignal_ks' + str(ks) + '_multi' + suffix_classbal + '_z' + str(mode_z) + '_3x10'
if not os.path.exists(subdirectory_plot):
    os.mkdir(subdirectory_plot)
subdirectory_plot_data = subdirectory_plot + '/data'
if not os.path.exists(subdirectory_plot_data):
    os.mkdir(subdirectory_plot_data)


train_size = 0.5

kernel = 'relu'

reg = fns_plot.return_reg(mode_reg,kernel,hidden_layer_sizes=hidden_layer_sizes)

for cond in conditions:
    print(cond)
    
    f = open(subdirectory_data + '/data_' + cond + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    
    feature = data.signals[:,:,ks]
    
    target = data.genes[:,:,:]

    if len(feature.shape)==2:
        feature = feature[:,:,np.newaxis]
    if len(target.shape)==2:
        target = target[:,:,np.newaxis]
    
    if mode_single:
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
            
            target_predict_train = fns_plot.undo_zscore(reg.predict(feat_train_z),mean_tar_train,stdev_tar_train)
            target_predict = fns_plot.undo_zscore(reg.predict(feat_test_z),mean_tar_train,stdev_tar_train)
        else:
            reg.fit(feat_train,tar_train)
            
            target_predict_train = reg.predict(feat_train)
            target_predict = reg.predict(feat_test)
            
            mean_feat_train,stdev_feat_train = 0,0
            mean_tar_train,stdev_tar_train = 0,0
        
        f = open(subdirectory_plot_data+'/data_regression_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
        pickle.dump((reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test, target_predict_train, target_predict,mean_feat_train,stdev_feat_train,mean_tar_train,stdev_tar_train), f)
        f.close()
    
    if mode_colony:
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
    
    
    if mode_av:
        
        thresh = 1
        ratios = np.zeros((N_run,data.N_genes,5))
        MIs = np.zeros((N_run,data.N_genes))
        
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
                

                ratios[it,kg,0] = fns_plot.diff_zero(TN,(TN+FP))
                ratios[it,kg,1] = fns_plot.diff_zero(TP,(TP+FN))
                
                ratios[it,kg,2] = fns_plot.diff_zero(TN,(TN+FN))
                ratios[it,kg,3] = fns_plot.diff_zero(TP,(TP+FP))
                
                ratios[it,kg,4] = fns_plot.diff_zero((TP+TN),(TN + FP + FN + TP))

                MIs[it,kg] = fns_plot.calc_MI_sklearn(tar_test[:,kg],target_predict[:,kg])
        
        f = open(subdirectory_plot_data+'/data_regression_av_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
        pickle.dump((ratios), f)
        f.close()
        
        f = open(subdirectory_plot_data+'/data_regression_MI_av_' + mode_reg + kernel + '_' + cond + ".pickle",'wb')
        pickle.dump((MIs), f)
        f.close()
    
    
        
        
        
        