import numpy as np
import pandas as pd
import os
import dill as pickle

import fns_data_wrapper_v3 as fns_data_wrapper

import fns_plotting_scripts as fns_plot

ID = 20
#directory = 'data_expt_' + str(ID) + '_scaled_norm_bgsub'
directory = 'data_expt_' + str(ID) + '_scaled_norm_bgsub_oldTBX6'

subdirectory_data = directory + '/data'

file_name = subdirectory_data + '/' + 'exp' + str(ID) + '_data_bgsub_scaled_250716' + '.csv'

data = pd.read_csv(file_name)

signal_names = ['pSMAD1','ncYAP','BCat','LEF1','ncSMAD23','cytopERK','cytopAKT']#,'ncSMAD2']
signal_names_simplified = ['SMAD1','YAP','BCat','LEF1','SMAD23','ERK','AKT']#,'SMAD2']
gene_names = ['ISL1','GATA3','HAND1','CDX2','TFAP2C','SOX17_1','PRDM1','SNAI1','MIXL1','TBXT','EOMES','TBX6','OCT4','NANOG','SOX2','OTX2']
gene_names_simplified = ['ISL1','GATA3','HAND1','CDX2','TFAP2C','SOX17','PRDM1','SNAI1','MIXL1','TBXT','EOMES','TBX6','OCT4','NANOG','SOX2','OTX2']
marker_names = ['threshold_clusters']
nuc_feat_names = ['nucArea', 'nucVolume','nucMajorAxis', 'nucMinorAxis', 'nucCircularity']

N_signals = len(signal_names)
N_genes = len(gene_names)
N_markers = len(marker_names)
N_nuc_feat = len(nuc_feat_names)

N_entries = len(data.Colony)
N_organoids_max = np.max(data.Colony)
N_cells_max = int(2*(N_entries/N_organoids_max))

rad = 350 #nominal radius

if N_markers>0:
    fate_names = np.unique(data.threshold_clusters)
    print(fate_names)
    fate_names_ordered = ['AMLC','PGCLC','PSLC','meso','pluri','ecto','endo','mitotic','none']
    fate_names_save = ['AMLC','PGCLC','PSLC','meso','pluri','ecto','junk']
else:
    fate_names_save = None

X = np.zeros((N_organoids_max,N_cells_max,2))
X[:] = np.nan

R = np.zeros((N_organoids_max,N_cells_max,4))
R[:] = np.nan

S = np.zeros((N_organoids_max,N_cells_max,N_signals))
S[:] = np.nan

G = np.zeros((N_organoids_max,N_cells_max,N_genes))
G[:] = np.nan

Z = np.zeros((N_organoids_max,N_cells_max,max(N_markers,1))) #for discrete fates!
Z[:] = np.nan

N = np.zeros((N_organoids_max,N_cells_max,N_nuc_feat)) #for discrete fates!
N[:] = np.nan

conditions = [[]]*N_organoids_max

microns_per_px = 0.325

ind2_prev = 0
ind3 = 0
for it in range(0,N_entries):
    
    ind2 = data.Colony[it]-1
    if ind2 > ind2_prev:
        ind3 = 0
        
    conditions[ind2] = data.condition[it]

    X[ind2,ind3,0] = data.X[it]*microns_per_px
    X[ind2,ind3,1] = data.Y[it]*microns_per_px
    
    R[ind2,ind3,0] = data.RadialDist[it]
    R[ind2,ind3,1] = data.MetricDist[it]
    R[ind2,ind3,2] = data.GraphDist[it]
    R[ind2,ind3,3] = data.CircleEdgeDist[it]
    
    for it_signal in range(0,N_signals):
        S[ind2,ind3,it_signal] = data[signal_names[it_signal]][it]
        
    for it_gene in range(0,N_genes):
        if it_gene == 11:
            G[ind2,ind3,it_gene] = data[gene_names[it_gene]][it]*0.8
        else:
            G[ind2,ind3,it_gene] = data[gene_names[it_gene]][it]
        
    for it_nuc_feat in range(0,N_nuc_feat):
        N[ind2,ind3,it_nuc_feat] = data[nuc_feat_names[it_nuc_feat]][it]
        
    if N_markers>0:
        for it_marker in range(0,N_markers):
            fate = data[marker_names[it_marker]][it]
            fate_index = fate_names_ordered.index(fate)
            if fate_index>=6: #endo, mitotic, none -> junk
                Z[ind2,ind3,it_marker] = 6
            else:
                Z[ind2,ind3,it_marker] = fate_index
    
    
    ind3 += 1
    ind2_prev = ind2



condition_names = []
for item in conditions: 
    if item not in condition_names: 
        condition_names.append(item) 
N_cond = len(condition_names)

for condition in condition_names:
    if len(condition)>0:
    
        condition_truefalse = np.array([c==condition for c in conditions])
        N_organoids_here = sum(condition_truefalse)
        
        X_here = X[condition_truefalse,:,:]
        R_here = R[condition_truefalse,:,:]
        S_here = S[condition_truefalse,:,:]
        G_here = G[condition_truefalse,:,:]
        Z_here = Z[condition_truefalse,:,:]
        N_here = N[condition_truefalse,:,:]
        
        data_fixed = fns_data_wrapper.FixedData_minimal(X_here,R_here,S_here,G_here,Z_here,N_here,signal_names,gene_names_simplified,fate_names_save,signal_names_simplified,nuc_feat_names,rad)
        
        f = open(subdirectory_data+'/data_' + condition + ".pickle",'wb')
        pickle.dump((data_fixed), f)
        f.close()



        
        
        
        
