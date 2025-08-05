#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:01:25 2022

@author: D.Brueckner
"""

import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib as mpl

def diff_zero(x,y):
    if y!=0:
        return x/y
    else:
        return 0

def return_colmaps(mode,N_var=None):
    
    if mode == 'conditions':
        colors = ['cornflowerblue','green','darkred'] #B10, B50, B100
    elif mode == 'fates':
        #cmap = mpl.cm.get_cmap('tab10')
        #colors = cmap(list(np.linspace(0,1,10)))
        colors = np.array([[90/255,166/255,71/255,1],[227/255,143/255,52/255,1],
                 [211/255,62/255,43/255,1], [140/255,40/255,93/255,1],
                 [75/255,167/255,158/255,1], [49/255,118/255,181/255,1]])
    elif mode == 'signals':
        if N_var==None:
            N_var = 7
        cmap = mpl.cm.get_cmap('tab10')
        colors = cmap(list(np.linspace(0.2,1,N_var)))
    elif mode == 'genes':
        if N_var==None:
            N_var = 17
        cmap = mpl.cm.get_cmap('plasma')
        colors = cmap(list(np.linspace(0,1,N_var)))
    elif mode == 'replicates':
        cmap = mpl.cm.get_cmap('tab10')
        colors = cmap(list(np.linspace(0,1,N_var)))
    return colors


def get_gene_index(gene_name,gene_names):
    return np.where([x==gene_name for x in gene_names])[0][0]

def do_thresh(x,gene_index,thresh):
    return x[gene_index]>thresh

def return_fates_old(G,gene_names,thresh=1):
    """
    PGCLC: TFAP2C and (SOX17_1 or PRDM1 or NANOG)
    AMLC: ISL1 or GATA3 (you could argue that maybe just ISL1 would be better, but I have found that this works pretty well)
    PSLC: (TBXT or EOMES or MIXL1) and ~TBX6
    meso: (TBXT or MIXL1 or EOMES) and TBX6
    pluri: SOX2 and NANOG
    ecto: SOX2 and ~NANOG
    
    fates = ['AMLC', 'PGCLC', 'PSLC', 'ecto', 'meso', 'pluri', 'junk']
    """
    N_cells = G.shape[0]
    
    markers_colony = np.zeros(N_cells)
    markers_colony[:] = np.nan
    
    TFAP2C = get_gene_index('TFAP2C',gene_names)
    SOX17 = get_gene_index('SOX17',gene_names)
    PRDM1 = get_gene_index('PRDM1',gene_names)
    NANOG = get_gene_index('NANOG',gene_names)
    ISL1 = get_gene_index('ISL1',gene_names)
    GATA3 = get_gene_index('GATA3',gene_names)
    TBXT = get_gene_index('TBXT',gene_names)
    TBX6 = get_gene_index('TBX6',gene_names)
    EOMES = get_gene_index('EOMES',gene_names)
    MIXL1 = get_gene_index('MIXL1',gene_names)
    SOX2 = get_gene_index('SOX2',gene_names)
    
    for j in range(N_cells):
        g = G[j,:]>=thresh
        if g[TFAP2C] and ( g[SOX17] or g[PRDM1]): # or g[NANOG] 
            marker = 1 #PGCLC
        elif g[ISL1] or g[GATA3]:
            marker = 0 #amnion
        elif ( g[TBXT] or g[EOMES] or g[MIXL1] ):
            if g[TBX6]:
                marker = 4 #meso
            else:
                marker = 2 #PS
        elif g[SOX2]:
            if g[NANOG]:
                marker = 5 #pluri
            else:
                marker = 3 #ecto
        else:
            marker = 6 #junk
        markers_colony[j] = marker
    
    return markers_colony


def calc_MI_regression(x,y,mode_discrete=False,hidden_layer_sizes=(10, 10, 10)):
    #only feature x can be high-dim
    from sklearn.model_selection import train_test_split
    feat_train, feat_test, tar_train, tar_test = train_test_split(x, y, test_size=0.5, random_state=42)
    if not mode_discrete:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor
        
    else:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier
    sklearn_model = model(
        hidden_layer_sizes=hidden_layer_sizes,  
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42)
    
    sklearn_model.fit(feat_train, tar_train.ravel())  # sklearn expects 1D target
    target_predict = sklearn_model.predict(feat_test)
    target_train_predict = sklearn_model.predict(feat_train)
            
    from sklearn.feature_selection import mutual_info_regression as mutual_info
    MI_test = mutual_info(tar_test.ravel().reshape(-1, 1),target_predict.ravel())[0]/np.log(2)
    MI_train = mutual_info(tar_train.ravel().reshape(-1, 1),target_train_predict.ravel())[0]/np.log(2)
    
    return MI_test,MI_train

def fate_names():
    # return fate names in order they are numbered below in return_fates
    return ['amnion','PGCLC','PS','meso','pluri','ecto']

def return_fates(G,gene_names,thresh=1):
    """
    PGCLC = TFAP2C > 1 & SOX17_1 > 1
    meso = TBXT > 1 & TBX6 > 1 & ~PGCLC
    AMLC = ISL1 > 1 & ~PGCLC & ~meso
    PSLC = TBXT > 1 & TBX6 < 1 & ~PGCLC & ~AMLC
    pluri = SOX2 > 1 & NANOG > 1 & ~PGCLC & ~meso & ~PSLC & ~AMLC
    ecto = SOX2 > 1 & NANOG < 1 & ~PGCLC & ~meso & ~PSLC & ~AMLC
    
    fates = ['AMLC','PGCLC','PSLC','meso','pluri','ecto', 'junk']
    """
    N_cells = G.shape[0]
    
    markers_colony = np.zeros(N_cells)
    markers_colony[:] = np.nan
    
    TFAP2C = get_gene_index('TFAP2C',gene_names)
    SOX17 = get_gene_index('SOX17',gene_names)
    #PRDM1 = get_gene_index('PRDM1',gene_names)
    NANOG = get_gene_index('NANOG',gene_names)
    ISL1 = get_gene_index('ISL1',gene_names)
    #GATA3 = get_gene_index('GATA3',gene_names)
    TBXT = get_gene_index('TBXT',gene_names)
    TBX6 = get_gene_index('TBX6',gene_names)
    #EOMES = get_gene_index('EOMES',gene_names)
    #MIXL1 = get_gene_index('MIXL1',gene_names)
    SOX2 = get_gene_index('SOX2',gene_names)
    
    
    for j in range(N_cells):
        g = G[j,:]>=thresh
        if g[TFAP2C] and g[SOX17]:
            marker = 1 #PGCLC
        elif g[TBXT] and g[TBX6]:
            marker = 3 #meso
        elif g[ISL1]:
            marker = 0 #amnion
        elif g[TBXT] and not g[TBX6]:
            marker = 2 #PS
        elif g[SOX2]:
            if g[NANOG]:
                marker = 4 #pluri
            else:
                marker = 5 #ecto
        else:
            marker = 6 #junk
        markers_colony[j] = marker
    
    return markers_colony

def get_neighbor_vertex_ids_from_vertex_id(vertex_id, triangulation):
        index_pointers, indices = triangulation.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids
    
def calc_clist_2d(points): 
    N_cells = points.shape[0]
    from scipy.spatial import Delaunay
    triang = Delaunay(points)
    c_list = []
    for j in range(0,N_cells):
        result_ids = get_neighbor_vertex_ids_from_vertex_id(j,triang)
        c_list.append(list(result_ids))
    return c_list


def equal_freq_bins(data, N_bins):
    indices = ~np.isnan(data)
    data = data[indices]
    nlen = len(data)
    return np.interp(np.linspace(0, nlen, N_bins + 1),np.arange(nlen),np.sort(data))

def calc_prediction_hist(tar_test, target_predict, N_bins, g_min, g_max, min_count=5):
    edges = np.linspace(g_min,g_max,N_bins+1)
    bins = (edges[1:] + edges[:-1])/2
    
    #x is predicted, y is observed
    hist,_,_ = np.histogram2d(target_predict,tar_test,edges)
    hist = hist/np.sum(hist)
    
    hist_cond = np.zeros(hist.shape)
    for b in range(N_bins):
        sumcounts = np.sum(hist[b,:]) #normalize each column
        if sumcounts>0:
            hist_cond[b,:] = hist[b,:]/sumcounts
            
        
    hist_av = np.zeros(N_bins)
    hist_var = np.zeros(N_bins)
    hist_sq_sum = np.zeros(N_bins)
    count = np.zeros(N_bins)
    N_entries = len(tar_test)
    for i in range(N_entries):
        for b in range(N_bins):
            #condition on predicted data, average observed (test) data
            #this avoids 'errors in variables' issues
            if(target_predict[i]  >= edges[b] and target_predict[i] < edges[b+1]):
                hist_av[b] += tar_test[i]
                hist_sq_sum[b] += tar_test[i]**2
                count[b] += 1
    for b in range(N_bins):
        if count[b]>min_count:
            hist_av[b] /= count[b]
            hist_var[b] = hist_sq_sum[b]/count[b] - hist_av[b]**2
        else:
            hist_av[b] = np.nan
            hist_var[b] = np.nan
    
    return bins,hist,hist_cond,hist_av,hist_var

def clean_data(feature,target):
    N_sys = feature.shape[0]
    N_part_max = feature.shape[1]
    N_feat = feature.shape[2]
    N_tar = target.shape[2]
    
    indices = ~np.isnan(feature[:,:,0].ravel())
    for k in range(0,N_feat): #multiply with all signal conditions
        indices *= ~np.isnan(feature[:,:,k].ravel())
    for k in range(0,N_tar): #multiply with all signal conditions
        indices *= ~np.isnan(target[:,:,k].ravel())
        
    feature_clean = feature.reshape((int(N_sys*N_part_max),N_feat))[indices,:]
    target_clean = target.reshape((int(N_sys*N_part_max),N_tar))[indices,:]
    return feature_clean,target_clean,indices

def clean_data_full(data,feature,target):
    N_sys = feature.shape[0]
    N_part_max = feature.shape[1]
    
    feature_clean,target_clean,indices = clean_data(feature,target)
        
    metricdist_clean = data.metricdist.reshape((int(N_sys*N_part_max)))[indices]
    markers_clean = data.markers.reshape((int(N_sys*N_part_max)))[indices]
    return feature_clean,target_clean,metricdist_clean,markers_clean


def test_train_split(feature,target,train_size=3):
    #split test/train by colony
    N_sys = feature.shape[0]
    N_tar = target.shape[2]
    
    test_size = N_sys - train_size
    feature_train = feature[-train_size:,:,:] #use last colonies for training
    feature_test = feature[:test_size,:,:]
    target_train = target[-train_size:,:,:] #use last colonies for training
    target_test = target[:test_size,:,:]
    
    #clean train data
    feat_train,tar_train,_ = clean_data(feature_train,target_train)
    
    #clean test data
    feat_test,tar_test,_ = clean_data(feature_test,target_test)
    
    if N_tar == 1:
        tar_train = tar_train.ravel()
        tar_test = tar_test.ravel()
    
    return feat_train,feat_test,tar_train,tar_test

def test_train_split_cells(feature,target,train_size=0.5):
    N_tar = target.shape[2]
    
    feature_clean,target_clean,_ = clean_data(feature,target)
    
    from sklearn.model_selection import train_test_split 
    feat_train, feat_test, tar_train, tar_test = train_test_split(feature_clean,target_clean, random_state=104,  test_size=1-train_size,  shuffle=True) 
    
    if N_tar == 1:
        tar_train = tar_train.ravel()
        tar_test = tar_test.ravel()
    
    return feat_train,feat_test,tar_train,tar_test


def test_train_split_cells_v2(data,feature,target,train_size=0.5):
    #include metricdist in output
    N_tar = target.shape[2]
    
    feature_clean,target_clean,metricdist_clean,markers_clean = clean_data_full(data,feature,target)
    
    N_entries = len(feature_clean)
    N_train = int(train_size*N_entries)
    #N_test = N_entries-N_train
    indices_train = np.random.randint(0,N_entries,N_train)
    
    mask = np.full(N_entries,False,dtype=bool)
    mask[indices_train] = True

    feat_train = feature_clean[mask]
    tar_train = target_clean[mask]
    metricdist_train = metricdist_clean[mask].ravel()
    markers_train = markers_clean[mask].ravel()
    
    feat_test = feature_clean[~mask]
    tar_test = target_clean[~mask]
    metricdist_test = metricdist_clean[~mask].ravel()
    markers_test = markers_clean[~mask].ravel()
    
    if N_tar == 1:
        tar_train = tar_train.ravel()
        tar_test = tar_test.ravel()
    
    return feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test



def test_train_split_cells_classbal(data,feature,target,train_size=0.5):
    N_tar = target.shape[2]
    
    feature_clean,target_clean,metricdist_clean,markers_clean = clean_data_full(data,feature,target)
    
    #exclude junk fate
    m_junk = 6
    indices_nojunk = np.where(markers_clean!=m_junk)[0]
    N_entries = len(indices_nojunk)
    feature_clean,target_clean,metricdist_clean,markers_clean = feature_clean[indices_nojunk],target_clean[indices_nojunk],metricdist_clean[indices_nojunk],markers_clean[indices_nojunk]
    
    #class balancing
    N_fates = 6
    N_entries_fates = np.zeros(N_fates)
    for m in range(N_fates): 
        N_entries_fates[m] = len(np.where(markers_clean==m)[0])
    N_train_per_fate = int(train_size*min(N_entries_fates))
    
    indices_train = []
    for m in range(N_fates):
        indices_fate = np.where(markers_clean==m)[0]
        
        #sample WITHOUT replacement
        indices_train_fate = np.random.choice(indices_fate,N_train_per_fate,replace=False)
        indices_train.append(indices_train_fate)
    
    indices_train_list = [x for xs in indices_train for x in xs]
    indices_train_array = np.array(indices_train_list)

    #got weird inconsistencies with mask, so use index list here
    all_ind = np.arange(N_entries)
    indices_test_array = all_ind[~np.isin(all_ind, indices_train_array)]
    
    feat_train = feature_clean[indices_train_array]
    tar_train = target_clean[indices_train_array]
    metricdist_train = metricdist_clean[indices_train_array].ravel()
    markers_train = markers_clean[indices_train_array].ravel()
    
    feat_test = feature_clean[indices_test_array]
    tar_test = target_clean[indices_test_array]
    metricdist_test = metricdist_clean[indices_test_array].ravel()
    markers_test = markers_clean[indices_test_array].ravel()
    
    if N_tar == 1:
        tar_train = tar_train.ravel()
        tar_test = tar_test.ravel()
    
    return feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test


def test_train_split_cells_classbal_resample(data,feature,target,train_size=0.5):
    N_tar = target.shape[2]
    
    feature_clean,target_clean,metricdist_clean,markers_clean = clean_data_full(data,feature,target)
    
    #exclude junk fate
    m_junk = 6
    indices_nojunk = np.where(markers_clean!=m_junk)[0]
    N_entries = len(indices_nojunk)
    feature_clean,target_clean,metricdist_clean,markers_clean = feature_clean[indices_nojunk],target_clean[indices_nojunk],metricdist_clean[indices_nojunk],markers_clean[indices_nojunk]
    
    #class balancing
    N_fates = 6
    N_entries_fates = np.zeros(N_fates)
    for m in range(N_fates): 
        N_entries_fates[m] = len(np.where(markers_clean==m)[0])
    N_train_per_fate = int(train_size*max(N_entries_fates))
    
    indices_train_unique = []
    indices_train = []
    for m in range(N_fates):
        indices_fate = np.where(markers_clean==m)[0]

        indices_train_fate_unique = np.random.choice(indices_fate,int(train_size*N_entries_fates[m]),replace=False)
        indices_train_unique.append(indices_train_fate_unique)
        
        indices_train_fate = np.random.choice(indices_train_fate_unique,N_train_per_fate,replace=True)
        indices_train.append(indices_train_fate)
    
    indices_train_unique_array = np.array([x for xs in indices_train_unique for x in xs])
    indices_train_array = np.array([x for xs in indices_train for x in xs])

    #got weird inconsistencies with mask, so use index list here
    all_ind = np.arange(N_entries)
    indices_test_array = all_ind[~np.isin(all_ind, indices_train_unique_array)]
    
    feat_train = feature_clean[indices_train_array]
    tar_train = target_clean[indices_train_array]
    metricdist_train = metricdist_clean[indices_train_array].ravel()
    markers_train = markers_clean[indices_train_array].ravel()
    
    feat_test = feature_clean[indices_test_array]
    tar_test = target_clean[indices_test_array]
    metricdist_test = metricdist_clean[indices_test_array].ravel()
    markers_test = markers_clean[indices_test_array].ravel()
    
    if N_tar == 1:
        tar_train = tar_train.ravel()
        tar_test = tar_test.ravel()
    
    return feat_train,feat_test,tar_train,tar_test,metricdist_train,metricdist_test,markers_train,markers_test


def get_activations(clf, X):
    #https://stackoverflow.com/questions/46728937/retrieve-final-hidden-activation-layer-output-from-sklearns-mlpclassifier
    hidden_layer_sizes = clf.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],
                                     layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations

def calc_false_neg(target,target_predict,thresh):
    
    pos_expt = target>=thresh
    pos_sim = target_predict>=thresh
    
    from sklearn.metrics import confusion_matrix
    N_true_neg, N_false_pos, N_false_neg, N_true_pos = confusion_matrix(pos_expt, pos_sim).ravel()
    
    N_cells = len(pos_expt)
    N_pos_expt = N_true_pos+N_false_neg
    N_neg_expt = N_cells - N_pos_expt

    N_matrix = np.zeros((2,2)) #indices = true, predicted
    N_matrix[1,1] = N_true_pos
    N_matrix[0,0] = N_true_neg
    N_matrix[1,0] = N_false_neg
    N_matrix[0,1] = N_false_pos
    
    P_true_pos = N_true_pos/N_pos_expt
    P_false_neg = 1-P_true_pos #(N_pos_expt-N_pos_both)/N_pos_expt
    
    P_true_neg = N_true_neg/N_neg_expt
    P_false_pos = 1-P_true_neg #(N_neg_expt-N_neg_both)/N_neg_expt
    
    P_matrix = np.zeros((2,2)) #indices = true, predicted
    P_matrix[1,1] = P_true_pos
    P_matrix[0,0] = P_true_neg
    P_matrix[1,0] = P_false_neg
    P_matrix[0,1] = P_false_pos
    
    return N_matrix,P_matrix

def do_zscore(X,mean=None,stdev=None):
    X_zscore = np.zeros(X.shape)
    if mean is None and stdev is None:
        mean = np.nanmean(X,axis=0)
        stdev = np.sqrt(np.nanvar(X,axis=0))
    
    if len(X.shape)>1:
        N_k = X.shape[1]
        for k in range(N_k):
            X_zscore[:,k] = (X[:,k] - mean[k])/stdev[k]
    else:
        X_zscore = (X[:] - mean)/stdev
        
    return X_zscore,mean,stdev

def undo_zscore(X_zscore,means,stdevs):
    X = np.zeros(X_zscore.shape)
    if len(X.shape)>1:
        N_k = X_zscore.shape[1]
        for k in range(N_k):
            X[:,k] = X_zscore[:,k]*stdevs[k] + means[k]
    else:
        X = X_zscore[:]*stdevs + means
        
    return X

def return_reg(mode_reg,kernel,hidden_layer_sizes=(50,50),C=1,max_iter=500):
    if mode_reg == 'SVR':
        from sklearn.svm import SVR
        reg = SVR(C=C,kernel=kernel,gamma='scale',cache_size=1000)  
    elif mode_reg == 'linear':
        kernel = ''
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
    elif mode_reg == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor()
    elif mode_reg == 'GBR':
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor()
    elif mode_reg == 'HGBR':
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor()
    elif mode_reg == 'SVC':
        from sklearn.svm import SVC
        reg = SVC()
    elif mode_reg == 'MLP':
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,activation=kernel,max_iter=max_iter)#solver='adam') #random_state=1, max_iter=2000, tol=0.1,
    elif mode_reg == 'MLPC':
        from sklearn.neural_network import MLPClassifier
        reg = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iter)#solver='adam') #random_state=1, max_iter=2000, tol=0.1,
    
    elif mode_reg == 'BernoulliRBM':
        from sklearn.neural_network import BernoulliRBM
        reg = BernoulliRBM()
    else:
        print('ERR: regression mode not found')
    return reg


def scatter_density(xdata_in,ydata_in,tol = [5,95],ms=1,cmap='viridis',alpha=1):
    
    indices = ( ~np.isnan(xdata_in.ravel()) ) * ( ~np.isnan(ydata_in.ravel()) )
    
    xdata = xdata_in.ravel()[indices]
    ydata = ydata_in.ravel()[indices]
    
    from scipy.stats import gaussian_kde
    idx = np.random.choice(xdata.shape[0], size=300, replace=False) #not sure if performing the KDE on only a subset of the data is necessary for speed?
    xy = np.vstack([xdata,ydata])
    c = gaussian_kde(xy[:,idx])(xy)
    c = np.log(1+c)
    cmin, cmax = np.percentile(c, tol)
    plt.scatter(xdata,ydata, s=ms, c=c, vmin=cmin, vmax=cmax, cmap=cmap, alpha=alpha)


def calc_profile_meanvar(X,pos_all,N_bins_x,r_max):
    
    bins_x,P_x,X_flat,_ = calc_X_sorted(X,pos_all,N_bins_x,r_max)
    dx = bins_x[1]-bins_x[0]
    N_var = X_flat.shape[2]

    mean_x = np.zeros((N_bins_x,N_var))
    var_x = np.zeros((N_bins_x,N_var))
    P_x = np.zeros((N_bins_x))
    for b_x in range(0,N_bins_x):
        #https://stackoverflow.com/questions/59582731/get-non-zero-and-not-nan-elements-column-wise
        indices = (~np.isnan(X_flat[b_x,:,:])).all(axis=1)&~(X_flat[b_x,:,:]==1).any(axis=1)

        mean_x[b_x,:] = np.mean(X_flat[b_x,indices,:],axis=0)
        var_x[b_x,:] = np.var(X_flat[b_x,indices,:],axis=0)
        
        P_x[b_x] = len(X_flat[b_x,indices,0])
        
    P_x = P_x/np.sum(P_x)/dx
    
    return bins_x,mean_x,var_x,P_x


def calc_profile_poserror(X,pos_all,N_bins_x,r_max,overlap=1):
    
    if overlap==1:
        bins_x,P_x,X_flat,_ = calc_X_sorted(X,pos_all,N_bins_x,r_max)
    else:
        bins_x,P_x,X_flat,_,_,_ = calc_X_sorted_overlap(X,pos_all,N_bins_x,r_max,overlap)
    
    dx = bins_x[1:]-bins_x[:-1]
    N_var = X.shape[2]
    
    bins_x_sym = (bins_x[1:]+bins_x[:-1])/2
    
    mean_x = np.zeros((N_bins_x,N_var))
    var_x = np.zeros((N_bins_x,N_var))
    C_matrix_x = np.zeros((N_bins_x,N_var,N_var))
    C_matrix_x_inv = np.zeros((N_bins_x,N_var,N_var))
    for b_x in range(0,N_bins_x):
        #https://stackoverflow.com/questions/59582731/get-non-zero-and-not-nan-elements-column-wise
        indices = (~np.isnan(X_flat[b_x,:,:])).all(axis=1)&~(X_flat[b_x,:,:]==1).any(axis=1)
        
        mean_x[b_x,:] = np.mean(X_flat[b_x,indices,:],axis=0)
        var_x[b_x,:] = np.var(X_flat[b_x,indices,:],axis=0)
        C_matrix_x[b_x,:,:] = np.cov(X_flat[b_x,indices,:].T)
        C_matrix_x_inv[b_x,:,:] = np.linalg.inv(C_matrix_x[b_x,:,:])
        
    diff_profile_all = np.diff(mean_x[:,:],axis=0)/dx[:,np.newaxis]

    sum_pos_error=0
    for i in range(0,N_var):
        for j in range(0,N_var):
            sum_pos_error += diff_profile_all[:,i]*C_matrix_x_inv[:-1,i,j]*diff_profile_all[:,j]
    pos_error = np.sqrt(1/sum_pos_error)
    
    return bins_x,mean_x,var_x,pos_error,bins_x_sym,C_matrix_x


def calc_poserror_smooth(X,pos_all,N_bins_x_raw,r_max,N_bins_x=1000,lam=100):

    bins_x_raw,mean_x_raw,var_x_raw,pos_error_raw,P_x_raw,C_matrix_x_raw = calc_profile_poserror(X,pos_all,N_bins_x_raw,r_max)
    
    from scipy.interpolate import make_smoothing_spline

    N_var = X.shape[2]
    bins_x = np.linspace(bins_x_raw[0],bins_x_raw[-1],N_bins_x)
    bins_x_sym = (bins_x[1:]+bins_x[:-1])/2
    mean_x = np.zeros((N_bins_x,N_var))
    var_x = np.zeros((N_bins_x,N_var))
    C_matrix_x = np.zeros((N_bins_x,N_var,N_var))
    for k in range(0,N_var):
        func_interp_av = make_smoothing_spline(bins_x_raw, mean_x_raw[:,k], lam=lam)
        mean_x[:,k] = func_interp_av(bins_x)
        
        for k2 in range(0,N_var):
            func_interp_covar = make_smoothing_spline(bins_x_raw, C_matrix_x_raw[:,k,k2], lam=lam)
            C_matrix_x[:,k,k2] = func_interp_covar(bins_x)
            
            if k==k2:
                var_x[:,k] = func_interp_covar(bins_x)

    dx = bins_x[1:]-bins_x[:-1]
    diff_profile = np.diff(mean_x,axis=0)/dx[:,np.newaxis]

    C_matrix_x_inv = np.zeros((N_bins_x,N_var,N_var))
    for b_x in range(0,N_bins_x):
        C_matrix_x_inv[b_x,:,:] = np.linalg.inv(C_matrix_x[b_x,:,:])

    sum_pos_error=0
    for i in range(0,N_var):
        for j in range(0,N_var):
            sum_pos_error += diff_profile[:,i]*C_matrix_x_inv[:-1,i,j]*diff_profile[:,j]
    pos_error = np.sqrt(1/sum_pos_error)
    
    return bins_x,mean_x,var_x,pos_error,bins_x_sym,C_matrix_x



def calc_regression_poserror(target_predict,tar_test,pos_all,N_bins_x,r_max):
    bins_x,P_x,target_predict_sorted,_ = calc_X_sorted(target_predict[np.newaxis,:,np.newaxis],pos_all[np.newaxis,:],N_bins_x,r_max)
    bins_x,P_x,tar_test_sorted,_ = calc_X_sorted(tar_test[np.newaxis,:,np.newaxis],pos_all[np.newaxis,:],N_bins_x,r_max)
    err_test_x = np.zeros(N_bins_x)
    for b_x in range(0,N_bins_x):
        err_test_x[b_x] = np.sqrt(np.nanmean(np.square(target_predict_sorted[b_x,:]-tar_test_sorted[b_x,:])))
    
    return bins_x,err_test_x


#determines an X array of signals/genes that is sorted by binned position b_x
def calc_X_sorted(X,pos_all,N_bins_x,r_max=None):

    if len(X.shape)==3:
        N_sys = X.shape[0]
        N_cells = X.shape[1]
        N_var = X.shape[2]
        
        X_sorted = np.zeros((N_bins_x,N_sys,N_cells,N_var))
        X_sorted[:] = np.nan
        count_x = np.zeros(N_bins_x,dtype=np.int64)
        count = 0
        
        #edges_x = np.linspace(0,r_max,N_bins_x+1)
        edges_x = equal_freq_bins(pos_all.ravel(), N_bins_x)
        bins_x = (edges_x[1:] + edges_x[:-1])/2.
        for it in range(N_sys):
            for j in range(0,N_cells):
                for b_x in range(0,N_bins_x):
                    if(pos_all[it,j] >= edges_x[b_x] and pos_all[it,j] < edges_x[b_x+1]):
                        X_sorted[b_x,it,count_x[b_x],:] = X[it,j,:]
                        count_x[b_x] += 1
                        count += 1
        
        P_x = count_x/count
        
        X_flat = X_sorted.reshape((N_bins_x,N_sys*N_cells,N_var))
        
        return bins_x,P_x,X_flat,X_sorted
    else:
        print('calc_X_sorted shape error')


def calc_X_sorted_overlap(X,pos_all,N_bins_x,r_max,overlap):

    if len(X.shape)==3:
        N_sys = X.shape[0]
        N_cells = X.shape[1]
        N_var = X.shape[2]
        
        X_sorted = np.zeros((N_bins_x,N_sys,N_cells,N_var))
        X_sorted[:] = np.nan
        
        pos_sorted = np.zeros((N_bins_x,N_sys,N_cells))
        pos_sorted[:] = np.nan
        
        count_x = np.zeros(N_bins_x,dtype=np.int64)
        count = 0
        
        edges_x = np.linspace(0,r_max,N_bins_x+1)
        bins_x = (edges_x[1:] + edges_x[:-1])/2.
        bin_width = bins_x[1]-bins_x[0]
        width = overlap*bin_width
        for it in range(N_sys):
            for j in range(0,N_cells):
                for b_x in range(0,N_bins_x):
                    if(pos_all[it,j] >= bins_x[b_x]-width/2 and pos_all[it,j] < bins_x[b_x]+width/2):
                        X_sorted[b_x,it,count_x[b_x],:] = X[it,j,:]
                        pos_sorted[b_x,it,count_x[b_x]] = pos_all[it,j]
                        count_x[b_x] += 1
                        count += 1
        
        P_x = count_x/count
        
        X_flat = X_sorted.reshape((N_bins_x,N_sys*N_cells,N_var))
        pos_flat = pos_sorted.reshape((N_bins_x,N_sys*N_cells))
        
        return bins_x,P_x,X_flat,X_sorted,pos_flat,pos_sorted
    else:
        print('calc_X_sorted shape error')


"""
def estimate_MI_regression(data,feature,target,mode_reg='MLP',kernel='relu',hidden_layer_sizes=(10,10,10)):
    #high dim feature, low dim target

    N_k = feature.shape[2]
    
    indices = ~np.isnan(target.ravel())
    for k in range(0,N_k): #multiply with all signal conditions
        indices *= ~np.isnan(feature[:,:,k].ravel())


    reg, feat_train, feat_test, tar_train, tar_test, metricdist_train,metricdist_test,markers_train,markers_test = regression_v4(data,feature,target,mode_reg=mode_reg,kernel=kernel,train_size=0.5,hidden_layer_sizes=hidden_layer_sizes)
    
    target_predict = reg.predict(feat_test)
    
    from sklearn.feature_selection import mutual_info_regression as mutual_info
    MI = mutual_info(tar_test.reshape(-1, 1),target_predict)/np.log(2)
    
    return MI


def estimate_MI_regression_clean(feature,target,mode_reg='MLP',kernel='relu',hidden_layer_sizes=(10,10,10),train_size=0.5):
    #high dim feature, low dim target
    #feature = feature[np.newaxis,:]
    #target = target[np.newaxis,:]
    
    if len(feature.shape)==1:
        feature = feature[:,np.newaxis]
    if len(target.shape)==1:
        target = target[:,np.newaxis]
        
    #feat_train,feat_test,tar_train,tar_test = test_train_split_cells(feature,target,train_size=train_size)
    from sklearn.model_selection import train_test_split 
    feat_train, feat_test, tar_train, tar_test = train_test_split(feature,target, random_state=104,  test_size=1-train_size,  shuffle=True) 
    
    reg = return_reg(mode_reg,kernel)
    reg.fit(feat_train,tar_train)

    target_predict = reg.predict(feat_test)

    from sklearn.feature_selection import mutual_info_regression as mutual_info
    MI = mutual_info(tar_test.reshape(-1, 1),target_predict)/np.log(2)
    
    return MI
"""

def calc_spearmancorr(x1,x2): #only feature can be discrete
    from scipy.stats import spearmanr
    X1 = x1.ravel()
    X2 = x2.ravel()
    
    indices = (~np.isnan(X1)*~np.isnan(X2))
    
    corr = spearmanr(X1[indices].reshape(-1, 1),X2[indices])
    return corr

def calc_pearsoncorr(x1,x2): #only feature can be discrete
    from scipy.stats import pearsonr
    X1 = x1.ravel()
    X2 = x2.ravel()
    
    indices = (~np.isnan(X1)*~np.isnan(X2))
    
    corr = pearsonr(X1[indices].reshape(-1, 1)[:,0],X2[indices])
    return corr

def calc_explained_variance(x1,x2): #only feature can be discrete
    from sklearn.metrics import explained_variance_score
    X1 = x1.ravel()
    X2 = x2.ravel()
    
    indices = (~np.isnan(X1)*~np.isnan(X2))
    
    corr = explained_variance_score(X1[indices].reshape(-1, 1)[:,0],X2[indices])
    return corr

def calc_MI_sklearn(feature,target,mode='cont'): 
    if mode=='cont':
        from sklearn.feature_selection import mutual_info_regression as mutual_info #only feature can be discrete
    elif mode=='disc':
        from sklearn.feature_selection import mutual_info_classif as mutual_info #both can be discrete
    
    X = feature.ravel()
    y = target.ravel()
    
    indices = (~np.isnan(X)*~np.isnan(y))
    
    MI = mutual_info(X[indices].reshape(-1, 1),y[indices])[0]/np.log(2)
    return MI

def entropy_patterning(x): #for discrete fates
    
    x_ravel = x.ravel()[~np.isnan(x.ravel())]
    x_int = np.array([int(a) for a in x_ravel])
    
    nbins = x_int.max() + 1 #maximum integer value + 1 is number of bins
    counts = np.bincount(x_int, minlength=nbins)
    p = counts / np.sum(counts)
    entropy_pat = entropy(p,base=2)
    return entropy_pat

def entropy_array(P):
    return -np.sum(P.ravel()*np.log2(P.ravel(),where=P.ravel()>0))

def entropy_gaussian(cov,N):
    detC = np.linalg.det(cov)
    if detC>0:
        #https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        result = 0.5*N*(1 + np.log2(2*np.pi)) + 0.5*np.log2(detC)
    else:
        result = np.nan
    return result

def calc_P_g_given_x(X,pos_all,N_bins_g,N_bins_x,g_max,r_max):
    N_sys = X.shape[0]
    N_cells = X.shape[1]
    
    edges_x = np.linspace(0,r_max,N_bins_x)
    edges_g = np.linspace(0,g_max,N_bins_g)
    
    Nhist = np.zeros((N_bins_x-1,N_bins_g-1))
    count_x = np.zeros((N_bins_x-1))
    count = 0

    for k in range(0,N_sys):
        for j in range(0,N_cells):
            for b_x in range(0,N_bins_x-1):
                if(pos_all[k,j] >= edges_x[b_x] and pos_all[k,j] < edges_x[b_x+1]):
                    for b_g in range(0,N_bins_g-1):
                        if(X[k,j] >= edges_g[b_g] and X[k,j] < edges_g[b_g+1]):
                            Nhist[b_x,b_g] += 1
                            count_x[b_x] += 1
                            count += 1
                
    P_g_given_x = np.zeros((N_bins_x-1,N_bins_g-1))
    for b_x in range(0,N_bins_x-1):
        if count_x[b_x]>0:
            P_g_given_x[b_x,:] = Nhist[b_x,:]/count_x[b_x]
    P_x = count_x/count
    bins_x = (edges_x[1:] + edges_x[:-1])/2.
    bins_g = (edges_g[1:] + edges_g[:-1])/2.
    return bins_g,bins_x,P_g_given_x,P_x


def calc_P_g1g2(X1,X2,N_bins_g1,N_bins_g2,g1_max,g2_max,g1_min=0,g2_min=0):
    N_sys = X1.shape[0]
    N_cells = X1.shape[1]
    
    edges_g1 = np.linspace(g1_min,g1_max,N_bins_g1)
    edges_g2 = np.linspace(g2_min,g2_max,N_bins_g2)
    
    Nhist = np.zeros((N_bins_g1-1,N_bins_g2-1))
    count = 0

    for k in range(0,N_sys):
        for j in range(0,N_cells):
            for b_g1 in range(0,N_bins_g1-1):
                if(X1[k,j] >= edges_g1[b_g1] and X1[k,j] < edges_g1[b_g1+1]):
                    for b_g2 in range(0,N_bins_g2-1):
                        if(X2[k,j] >= edges_g2[b_g2] and X2[k,j] < edges_g2[b_g2+1]):
                            Nhist[b_g1,b_g2] += 1
                            count += 1
                
    P_g1g2 = Nhist/count
    bins_g1 = (edges_g1[1:] + edges_g1[:-1])/2.
    bins_g2 = (edges_g2[1:] + edges_g2[:-1])/2.
    return bins_g1,bins_g2,P_g1g2


def calc_g_av(X,pos_all,N_bins_x,g_max,r_max):
    N_sys = X.shape[0]
    N_cells = X.shape[1]
    
    edges_x = np.linspace(0,r_max,N_bins_x)
    
    
    count_x = np.zeros((N_bins_x-1))
    g_sum = np.zeros((N_bins_x-1))
    g_sum_sq = np.zeros((N_bins_x-1))
    count = 0

    for k in range(0,N_sys):
        for j in range(0,N_cells):
            for b_x in range(0,N_bins_x-1):
                if(pos_all[k,j] >= edges_x[b_x] and pos_all[k,j] < edges_x[b_x+1]):
                    if ~np.isnan(X[k,j]):
                        count_x[b_x] += 1
                        g_sum[b_x] += X[k,j]
                        g_sum_sq[b_x] += X[k,j]**2
                        
                    count += 1
                
    P_x = count_x/count
    g_av = g_sum/count_x
    g_var = g_sum_sq/count_x - g_av**2  
    bins_x = (edges_x[1:] + edges_x[:-1])/2.
    return bins_x,P_x,g_av,g_var


def calc_fluctuations(X,pos_all,N_bins_x,g_max,r_max):
    N_sys = X.shape[0]
    N_cells = X.shape[1]
    
    bins_x,P_x,g_av,g_var = calc_g_av(X,pos_all,N_bins_x,g_max,r_max)
    
    edges_x = np.linspace(0,r_max,N_bins_x)
    
    X_fluc = np.zeros(X.shape)
    X_fluc[:] = np.nan
    X_fluc_norm = np.zeros(X.shape)
    X_fluc_norm[:] = np.nan

    for k in range(0,N_sys):
        for j in range(0,N_cells):
            for b_x in range(0,N_bins_x-1):
                if(pos_all[k,j] >= edges_x[b_x] and pos_all[k,j] < edges_x[b_x+1]):
                    if ~np.isnan(X[k,j]):                      
                        X_fluc[k,j] = X[k,j] - g_av[b_x]
                        X_fluc_norm[k,j] = (X[k,j] - g_av[b_x])/np.sqrt(g_var[b_x])
                
    return X_fluc,X_fluc_norm



def calc_PI_gx(X,pos_all,N_bins_g,N_bins_x,g_max,r_max):

    bins_g,bins_x,P_g_given_x,P_x=calc_P_g_given_x(X,pos_all,N_bins_g,N_bins_x,g_max,r_max)
    
    edges = np.linspace(0,g_max,N_bins_g)
    P_g,_ = np.histogram(X.ravel(),edges)
    P_g = P_g/np.sum(P_g)
    
    S1 = entropy(P_g)
    
    S2 = 0
    for b_x in range(0,N_bins_x-1):
        S2 += P_x[b_x]*entropy(P_g_given_x[b_x,:])
    
    MI = S1 - S2
    return MI


def calc_PI_gx_vs_x(X,pos_all,N_bins_g,N_bins_x,g_max,r_max):

    bins_g,bins_x,P_g_given_x,P_x=calc_P_g_given_x(X,pos_all,N_bins_g,N_bins_x,g_max,r_max)
    
    edges = np.linspace(0,g_max,N_bins_g)
    P_g,_ = np.histogram(X.ravel(),edges)
    P_g = P_g/np.sum(P_g)
    
    S1 = entropy(P_g)
    
    MI_all = np.zeros(N_bins_x-1)
    for b_x in range(0,N_bins_x-1):
        MI_all[b_x] = S1 - entropy(P_g_given_x[b_x,:])

    return MI_all,bins_x




def calc_P_x_given_g(P_g_given_x):
    N_bins_g = P_g_given_x.shape[1]
    
    P_x_given_g = np.flipud(np.rot90(P_g_given_x,k=1))

    for b_g in range(0,N_bins_g):
        normalization = np.sum(P_x_given_g[b_g,:])
        if normalization>0:
            P_x_given_g[b_g,:] = P_x_given_g[b_g,:]/normalization
            
    return P_x_given_g


def calc_xstar(X,g_max,P_x_given_g,bins_x):
    N_sys = X.shape[0]
    N_cells = X.shape[1]
    N_bins_g = P_x_given_g.shape[0]+1
    
    xstar_array = np.zeros(X.shape)
    xstar_array[:] = np.nan
    
    #P_xstar_given_x = np.zeros((N_bins_r,N_bins_r))
    edges = np.linspace(0,g_max,N_bins_g)
    for it in range(0,N_sys):
        for j in range(0,N_cells):
            for b_g in range(0,N_bins_g-1):
                if(X[it,j] >= edges[b_g] and X[it,j] < edges[b_g+1]):
                    b_g_here = b_g
                    b_x_here = np.argmax(P_x_given_g[b_g_here,:])
                    xstar_array[it,j] = bins_x[b_x_here]
    return xstar_array


def calc_P_xstar_given_x(X,pos_all,g_max,r_max,it,P_x_given_g):
    N_cells = X.shape[1]
    N_bins_g = P_x_given_g.shape[0]+1
    N_bins_x = P_x_given_g.shape[1]+1
    
    edges_x = np.linspace(0,r_max,N_bins_x)
    
    P_xstar_given_x_sum = np.zeros((N_bins_x-1,N_bins_x-1))
    count_x = np.zeros(N_bins_x-1)
    
    edges = np.linspace(0,g_max,N_bins_g)
    for j in range(0,N_cells):
        for b_g in range(0,N_bins_g-1):
            if(X[it,j] >= edges[b_g] and X[it,j] < edges[b_g+1]):
                b_g_here = b_g
                for b_x in range(0,N_bins_x-1):
                    if(pos_all[it,j] >= edges_x[b_x] and pos_all[it,j] < edges_x[b_x+1]):
                        b_x_here = b_x
                        P_xstar_given_x_sum[b_x_here,:] += P_x_given_g[b_g_here,:]
                        count_x[b_x_here] += 1

    P_xstar_given_x = np.zeros((N_bins_x-1,N_bins_x-1))
    for b_x in range(0,N_bins_x-1):
        if count_x[b_x]>0:
            P_xstar_given_x[b_x,:] = P_xstar_given_x_sum[b_x,:]/count_x[b_x]

    return P_xstar_given_x


def calc_PI_xstar(P_xstar_given_x,P_x):
    N_cells = P_xstar_given_x.shape[0]
    
    P_xstar = np.sum(P_xstar_given_x,axis=0)
    P_xstar /= np.sum(P_xstar)
    
    S1 = entropy(P_xstar)
    
    S2 = 0
    for b_x in range(0,N_cells):
        S2 += P_x[b_x]*entropy(P_xstar_given_x[b_x,:])
    
    MI = S1 - S2
    
    return MI


