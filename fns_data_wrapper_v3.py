#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:52:27 2020

@author: D.Brueckner
"""

import numpy as np
        
class FixedData_minimal(object): #more general than v1
 
    def __init__(self,X,R,S,G,Z,N,signal_names,gene_names,fate_names,signal_names_simple,nuc_feat_names,rad,conditions=None):
        
        self.N_sys = X.shape[0]
        self.N_part_max = X.shape[1]
        self.N_signals = S.shape[2]
        if G is None:
            self.N_genes = 0
        else:
            self.N_genes = G.shape[2]
        self.N_markers = Z.shape[2]
        self.N_nuc_feat = N.shape[2]
        self.N_dim = 2
        self.rad = rad
        
        self.signal_names = signal_names_simple
        self.signal_names_full = signal_names
        self.gene_names = gene_names
        self.fate_names = fate_names
        self.nuc_feat_names = nuc_feat_names
        self.conditions = conditions
        
        if self.fate_names is not None:
            self.N_fates = len(self.fate_names)
        
        self.X = np.empty((self.N_sys,self.N_part_max,self.N_dim))
        self.X[:] = np.nan 
        for it in range(0,self.N_sys):
            for d in range(0,self.N_dim): #spatial dimensions
                x_com = np.nanmean(X[it,:,d])
                self.X[it,:,d] = X[it,:,d] - x_com
        
        self.radialdist = R[:,:,0]
        #self.metricdist = R[:,:,1]
        self.graphdist = R[:,:,2]
        self.metricdist = R[:,:,3]
        
        self.signals = S
        self.genes = G
        self.markers = Z
        self.nuc_feat = N

        self.N_part = np.zeros(self.N_sys, dtype=np.int64)
        for it in range(self.N_sys):
            N_non_nan_indices = len([i for i, x in enumerate(~np.isnan(self.X[it,:,0])) if x])
            if N_non_nan_indices > 0:
                self.N_part[it] = int(N_non_nan_indices)
            else:
                self.N_part[it] = 0 #so empty organoids have N_part = nan
        
        
        self.r = np.sqrt(self.X[:,:,0]**2 + self.X[:,:,1]**2)
        
        #phi has 0 on the left x-axis and goes from 0 to 2pi
        self.phi = np.arctan2( self.X[:,:,1],self.X[:,:,0] ) + np.pi

        self.R = np.nanmax(self.r,axis=1)
        self.density_av = self.N_part/(np.pi*self.R**2)
        
        self.r_max = np.nanmax(self.r)
        self.s_max = np.zeros(self.N_signals)
        self.s_min = np.zeros(self.N_signals)
        for k in range(0,self.N_signals):
            self.s_max[k] = np.nanpercentile(self.signals[:,:,k].ravel(), 99.9)
            self.s_min[k] = np.nanmin(self.signals[:,:,k].ravel())
        self.g_max = np.zeros(self.N_genes)
        self.g_min = np.zeros(self.N_genes)
        for k in range(0,self.N_genes):
            self.g_max[k] = np.nanpercentile(self.genes[:,:,k].ravel(), 99.9)
            self.g_min[k] = np.nanmin(self.genes[:,:,k].ravel())