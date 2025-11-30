import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter1d
import sklearn
import torch
import torch.optim as optim
import os
from skimage import io as imio
from skimage import exposure
import sys
sys.path.append('/Users/idse/repos/signaldecoding/2D_gastruloids_v5')
import fns_plotting_scripts as fns_plot
import fns_NN

colmap_fates = {'AMLC':[90/255,166/255,71/255,1],'PGCLC':[227/255,143/255,52/255,1],
                 'PSLC':[211/255,62/255,43/255,1], 'meso':[140/255,40/255,93/255,1],
                 'pluri':[75/255,167/255,158/255,1], 'ecto':[49/255,118/255,181/255,1], 
                'endo':[227/255,179/255,61/255,1],'other':[0.8,0.8,0.8,1]}

#------------------------------------------------------------------------------------------------------------
# RAW DATA VISUALIZATION
#------------------------------------------------------------------------------------------------------------

def getStainSchemeFromDir(dataDir, rdStr='RD'):
    # EXTRACT STAINING SCHEME FROM DIRECTORY NAMES
    # 
    # we assume the image data is stored in the a subdirectory for each round of staining
    # with the name <PREFIX><rdStr><ROUNDNUMBER>_<STAIN1>_<STAIN2>_<STAIN3>
    # where <PREFIX> is a fixed prefix (experiment name) and <STAINX> is the name of the stain (both not containing _)
    # <ROUNDNUMBER> is assumed to range over consecutive integers starting at 1
    
    # get the subdirs that contain the data for each round
    stainDataDirs = [g for g in os.listdir(dataDir) if '_'+rdStr in g]
    
    # sort so order in list corresponds to round numbers
    stainDataDirs.sort()
    
    # determine which element of the split directory name corresponds to the round number
    rdidx = [i for i,j in enumerate(str.split(stainDataDirs[0],'_')) if j.startswith(rdStr)][0]; 
    
    # list of stains for each round
    rd2stains = [str.split(g,'_')[rdidx+1:] for g in stainDataDirs];
    
    # make flat list of unique stains in experiment
    allstains = []; 
    for i in range(0,len(rd2stains)):
        allstains = list(set(allstains + rd2stains[i]))
    
    # create dictionary to get rounds in which each stain occurs (can be multiple time)
    stain2rd = {};
    for s in allstains:
        stain2rd[s] = [i+1 for i,j in enumerate(rd2stains) if s in j]

    return stain2rd, rd2stains, stainDataDirs

def getImageFilename(coli, stain, dataDir, rdStr='RD', imtype='MIP'):

    stain2rd, rd2stains, stainDataDirs = getStainSchemeFromDir(dataDir, rdStr)
    
    # add _N for round N stain if stain was repeated, 
    # defaults to first round in which a stain occurs
    s = str.split(stain,'_');
    stain = s[0];
    if len(s)==1:
        stainrep = 0;
    else:
        stainrep = int(s[1])-1;
    
    if stain.startswith('DAPI') :
        ci = 0;
        if len(s)==2:
            rd = stainrep + 1;
        else:
            rd = 1;
    else:
        rd = stain2rd[stain][stainrep] - 1;
        ci = rd2stains[rd].index(s[0]) + 1;
        
    if imtype == 'MIP':    
        
        base_dir = os.path.join(dataDir, stainDataDirs[rd], 'MIP')
        file_base = 'stitched_MIP_p{0}_w{1}_t0000'
        # Colony numbers start from 1 but filenames start from 0 (col 1 = p0000)
        fname_base = file_base.format('%.4d' % (coli-1), '%.4d' % ci)
        filepath_tif = os.path.join(base_dir, fname_base + '.tif')
        filepath_jpg = os.path.join(base_dir, fname_base + '.jpg')
    
        # Check if .tif file exists
        if os.path.exists(filepath_tif):
            fname = filepath_tif
        elif os.path.exists(filepath_jpg):
            fname = filepath_jpg
        else:
            filepath = None  # Or raise an error, or handle as needed
            raise FileNotFoundError("File " + filepath_jpg + " or .tif does not exist")

    elif imtype == 'segOverlay':

        # CHECK THAT THIS IS THE RIGHT FORMAT
        filepath = os.path.join(dataDir, stainDataDirs[rd], 'MIP', 'aligned_segoverlay_p{0}.tif');
        fname = filepath.format('%.4d' % (coli-1));
        
    else:
        print('imtype not recognized, should be MIP or segOverlay')
        
    return fname

def adjust_contrast(im, tol):
    Imin, Imax = np.percentile(im[im>0], tol)
    im_rescale = exposure.rescale_intensity(im, in_range=(Imin,Imax),out_range=(0,255))
    return(im_rescale)

def makeRGBoverlay(coli, markers, dataDir, rdStr='RD', Ilim=None):
    
    MIPca = {}

    for i, m in zip(range(0,len(markers)), markers):

        fname = getImageFilename(coli, m, dataDir, rdStr) 
        print('loading', fname)
        MIP = imio.imread(fname)
        if Ilim==None:
            Imin, Imax = np.percentile(MIP[MIP>0], tol[m])
        else:
            Imin, Imax = Ilim[m]        
        MIPca[m] = exposure.rescale_intensity(MIP, in_range=(Imin,Imax),out_range=(0,255))

    
    if len(markers)==3:
        RGBoverlay = np.stack([MIPca[markers[0]], MIPca[markers[1]], MIPca[markers[2]]], axis = 2)
    elif len(markers)==2:
        RGBoverlay = np.stack([MIPca[markers[0]], MIPca[markers[1]], 0*MIPca[markers[0]]], axis = 2)
    else: 
        RGBoverlay = MIPca[markers[0]];
        
    RGBoverlay=RGBoverlay.astype(np.uint8)
        
    return RGBoverlay
    
#------------------------------------------------------------------------------------------------------------
# COMPATIBILTIY
#------------------------------------------------------------------------------------------------------------

# convert data to David's format 
def data2david(data, features):

    # Step 1: Features and Colonies
    df = data[features + ['Colony']]
    features = [c for c in df.columns if c != 'Colony']
    colonies = sorted(df['Colony'].unique())
    n_colonies = len(colonies)
    n_features = len(features)

    # Step 2: Max cells per colony
    max_cells = df.groupby('Colony').size().max()

    # Initialize array: (colonies, cells, features)
    arr = np.full((n_colonies, max_cells, n_features), np.nan)
    
    # Fill
    for i, colony_num in enumerate(colonies):
        colony_data = df[df['Colony'] == colony_num][features].values
        arr[i, :len(colony_data), :] = colony_data

    return arr

#------------------------------------------------------------------------------------------------------------
# ANALYSIS
#------------------------------------------------------------------------------------------------------------

def return_fates(data, thresh=1):
    """
    define fates based on fate marker expression
    """

    fate_names = ['AMLC','PGCLC','ecto','pluri','meso','PSLC','other'] # 'endo', 
    #fate_names = 

    if thresh==1:
        thresh = {'TFAP2C':1, 'SOX17':1, 'NANOG':1, 'ISL1':1, 'TBXT':1, 'TBX6':1, 'SOX2':1}
        SOX2_bg = 0.1
    else:
        SOX2_bg = 100
    
    TFAP2C = data['TFAP2C'] > thresh['TFAP2C']
    SOX17 = data['SOX17'] > thresh['SOX17']
    NANOG = data['NANOG'] > thresh['NANOG']
    ISL1 = data['ISL1'] > thresh['ISL1']
    TBXT = data['TBXT'] > thresh['TBXT']
    TBX6 = data['TBX6'] > thresh['TBX6']
    SOX2 = data['SOX2'] > thresh['SOX2']

    TBXT_scaled = data['TBXT']/thresh['TBXT']
    NANOG_scaled = data['NANOG']/thresh['NANOG']
    SOX2_scaled = data['SOX2']/thresh['SOX2']

    PGCLC = TFAP2C & SOX17 
    #endo = SOX17 & ~PGCLC
    meso = TBXT & TBX6 & ~PGCLC #& ~endo
    AMLC = ISL1 & ~PGCLC & ~meso # & ~endo  
    PSLC = (TBXT_scaled > NANOG_scaled) & TBXT & ~PGCLC & ~meso & ~AMLC  #& ~endo # (data['TBXT'] > 100)
    pluri = (NANOG_scaled > TBXT_scaled) & NANOG & SOX2 & ~PGCLC & ~meso & ~AMLC & ~PSLC  # & ~endo 
    ecto = ~NANOG & (SOX2_scaled > TBXT_scaled) & (data['SOX2'] > SOX2_bg) & ~PGCLC & ~meso & ~AMLC &~pluri &~PSLC #  & ~endo 
    AMLC = (AMLC | TFAP2C) & ~(ecto | pluri | PSLC | meso | PGCLC) # | endo
    other = ~(ecto | pluri | PSLC | AMLC | meso | PGCLC) #  | endo
    
    # # OLD defs
    # PGCLC = TFAP2C & SOX17 
    # AMLC = ISL1 & ~PGCLC
    # meso = TBXT & TBX6 & ~PGCLC & ~AMLC
    # PSLC = TBXT & ~TBX6 & ~PGCLC & ~AMLC 
    # pluri = SOX2 & NANOG & ~PGCLC & ~meso & ~PSLC & ~AMLC
    # ecto = SOX2 & ~NANOG & ~PGCLC & ~meso & ~PSLC & ~AMLC 
    # other = ~(ecto | pluri | PSLC | AMLC | meso | PGCLC)

    labels = np.empty(data.shape[0], dtype='<U5')  # or dtype=str
    labels[PGCLC] = "PGCLC"
    labels[meso] = "meso"
    #labels[endo] = "endo"
    labels[AMLC] = "AMLC"
    labels[PSLC] = "PSLC"
    labels[pluri] = "pluri"
    labels[ecto] = "ecto"
    labels[other] = "other"
    
    return labels, fate_names

def calcPerformance(data, pred_subs, thresh, gene_names):
                    
    performances = dict()

    for cond in np.unique(data['condition']): 
    
        fates, fate_names = return_fates(data, thresh=thresh)
        idx = (data['condition'] == cond) & (fates != 'other')
        data_cond = data[idx]
        
        fates_cond, _ = return_fates(data_cond, thresh=thresh)
        
        # Dictionary to collect all data before creating DataFrame
        all_data = {marker: {} for marker in gene_names + ['fate_macro'] + fate_names}
        
        for pred_sub_name, pred_sub in pred_subs.items():
    
            pred_sub = pred_sub['avg'][idx]
            
            # For collecting metrics across runs
            acc_dict = {marker: [] for marker in gene_names + ['fate_macro'] + fate_names}
            f1_dict  = {marker: [] for marker in gene_names + ['fate_macro'] + fate_names}
            precision_dict  = {marker: [] for marker in gene_names + ['fate_macro'] + fate_names}
            recall_dict  = {marker: [] for marker in gene_names + ['fate_macro'] + fate_names}

            # random guessing comparison
            f1_rand_dict = {marker: [] for marker in gene_names + ['fate_macro'] + fate_names}
            
            fates_pred, _ = return_fates(pred_sub, thresh=thresh)
    
            for colID in np.unique(data_cond['Colony']):
    
                colidx = data_cond['Colony']==colID

                #---------------- fate scores -------------------
                
                f1 = sklearn.metrics.f1_score(fates_cond[colidx], fates_pred[colidx], average='macro', zero_division=0)
                accuracy = sklearn.metrics.accuracy_score(fates_cond[colidx], fates_pred[colidx])
                precision = sklearn.metrics.precision_score(fates_cond[colidx], fates_pred[colidx], average='macro', zero_division=0)
                recall = sklearn.metrics.recall_score(fates_cond[colidx], fates_pred[colidx], average='macro', zero_division=0)
                    
                acc_dict['fate_macro'].append(accuracy)
                f1_dict['fate_macro'].append(f1)
                precision_dict['fate_macro'].append(precision)
                recall_dict['fate_macro'].append(recall)

                # macro average of f1 score for random guessing based on probability of fate is 1/n
                n = len([fn for fn in fate_names if fn!='other'])
                f1_rand_dict['fate_macro'].append(1/n)

                f1 = sklearn.metrics.f1_score(fates_cond[colidx], fates_pred[colidx], average=None, labels=fate_names, zero_division=0)
                precision = sklearn.metrics.precision_score(fates_cond[colidx], fates_pred[colidx], average=None, labels=fate_names, zero_division=0)
                recall = sklearn.metrics.recall_score(fates_cond[colidx], fates_pred[colidx], average=None, labels=fate_names, zero_division=0)
                
                for i,f in enumerate(fate_names):
                    if not np.isnan(f1[i]):
                        f1_dict[f].append(f1[i])
                        precision_dict[f].append(precision[i])
                        recall_dict[f].append(recall[i])
                        f1_rand_dict[f].append(np.sum(fates_cond[colidx]==f)/len(fates_cond[colidx]))

                #---------------- marker scores -------------------
                
                for marker in gene_names:
                        
                    markerpos = data_cond[colidx][marker].to_numpy() > thresh[marker]
                    markerpos_pred = pred_sub[colidx][marker].to_numpy() > thresh[marker] 
        
                    f1 = sklearn.metrics.f1_score(markerpos, markerpos_pred, average='binary', pos_label=1, zero_division=0)
                    accuracy = sklearn.metrics.accuracy_score(markerpos, markerpos_pred)
                    precision = sklearn.metrics.precision_score(markerpos, markerpos_pred, average='binary', pos_label=1, zero_division=0)
                    recall = sklearn.metrics.recall_score(markerpos, markerpos_pred, average='binary', pos_label=1, zero_division=0)
        
                    # Populate dicts
                    acc_dict[marker].append(accuracy)
                    f1_dict[marker].append(f1)
                    precision_dict[marker].append(precision)
                    recall_dict[marker].append(recall)
                    f1_rand_dict[marker].append(np.sum(markerpos)/len(markerpos))
            
            # Store all metrics for this pred_sub_name
            # Use nanmean and nanstd to handle NaN values
            fate_names = [f for f in fate_names if f != 'other']
            
            for k in gene_names + ['fate_macro'] + fate_names:

                # there is no accuracy for individual fates, just an overall accuracy of the N-way classification
                if k not in fate_names:
                    all_data[k][f"{pred_sub_name}-acc-avg"] = np.nanmean(acc_dict[k])
                    all_data[k][f"{pred_sub_name}-acc-std"] = np.nanstd(acc_dict[k])
                all_data[k][f"{pred_sub_name}-f1-avg"] = np.nanmean(f1_dict[k])
                all_data[k][f"{pred_sub_name}-f1-std"] = np.nanstd(f1_dict[k])
                all_data[k][f"{pred_sub_name}-f1_rand-avg"] = np.nanmean(f1_rand_dict[k])
                all_data[k][f"{pred_sub_name}-f1_rand-std"] = np.nanstd(f1_rand_dict[k])
                all_data[k][f"{pred_sub_name}-precision-avg"] = np.nanmean(precision_dict[k])
                all_data[k][f"{pred_sub_name}-precision-std"] = np.nanstd(precision_dict[k])
                all_data[k][f"{pred_sub_name}-recall-avg"] = np.nanmean(recall_dict[k])
                all_data[k][f"{pred_sub_name}-recall-std"] = np.nanstd(recall_dict[k])
                
        # Create DataFrame from all_data dictionary at once
        performance = pd.DataFrame.from_dict(all_data, orient='index')
    
        # Add average row for gene_names only (excluding 'fate')            
        avg_row = performance.loc[gene_names].mean()
        performance.loc['marker_avg'] = avg_row
        
        # Reorder columns: all acc columns first, then all f1 columns, etc
        acc_cols = [col for col in performance.columns if '-acc-' in col]
        f1_cols = [col for col in performance.columns if '-f1-' in col]
        precision_cols = [col for col in performance.columns if '-precision-' in col]
        recall_cols = [col for col in performance.columns if '-recall-' in col]
        f1_rand_cols = [col for col in performance.columns if '-f1_rand-' in col]
        performance = performance[acc_cols + f1_cols + precision_cols + recall_cols + f1_rand_cols]
    
        performances[cond]=performance
        
    return performances, recall_dict
    


class VIB:

    def __init__(self, feat_train, tar_train, hyperparam=None):

        # VIB hyperparameters (for signal input)
        defaults = {
            'LATENT_DIM' : 2,
            'HIDDEN_DIM' : 64,
            'N_LAYERS' : 2,
            'EPOCHS' : 800,
            'LEARNING_RATE' : 1e-3,
            'BETA' : 0.01
        }
        hyperparam = hyperparam or {} # if not provided make an empty dict
        hyperparam = {**defaults, **hyperparam} # merge dictionaries second overrides first
 
        N_DIM_INPUT = feat_train.shape[1]
        N_DIM_OUTPUT = tar_train.shape[1]

        self.tar_train = tar_train
        
        # Standardize
        self.scaler_X_run = sklearn.preprocessing.StandardScaler()
        self.scaler_Y_run = sklearn.preprocessing.StandardScaler()
        
        feat_train_z = self.scaler_X_run.fit_transform(feat_train)
        tar_train_z = self.scaler_Y_run.fit_transform(tar_train)
        
        # Convert to torch
        X_train_run = torch.FloatTensor(feat_train_z)
        Y_train_run = torch.FloatTensor(tar_train_z)
        
        # Create new VIB model (fresh random initialization each run)
        self.model = fns_NN.FlexibleVIB(
            input_dim=N_DIM_INPUT,
            output_dim=N_DIM_OUTPUT,
            latent_dim=hyperparam['LATENT_DIM'],
            hidden_dim=hyperparam['HIDDEN_DIM'],
            n_layers=hyperparam['N_LAYERS'],
            encoder_type='nonlinear',
            decoder_type='nonlinear'
        )
        
        # Train
        _ = self.train(
            X_train_run, Y_train_run,
            epochs=hyperparam['EPOCHS'],
            lr=hyperparam['LEARNING_RATE'],
            beta=hyperparam['BETA'],
            verbose=False
        )

    def predict(self, feat_test):
        
        feat_test_z = self.scaler_X_run.transform(feat_test)
        X_test_run = torch.FloatTensor(feat_test_z)

        self.model.eval()
        with torch.no_grad():
            target_predict_z = self.model(X_test_run)[0].numpy()

        # Inverse transform
        target_predict = self.scaler_Y_run.inverse_transform(target_predict_z)

        # convert back to dataframe if appropriate
        if type(feat_test) == pd.core.frame.DataFrame and type(self.tar_train) == pd.core.frame.DataFrame:
            target_predict = pd.DataFrame(target_predict, index=feat_test.index, columns=self.tar_train.columns)
    
        return target_predict

    def train(self, X_train, Y_train, epochs=800, lr=1e-3, #X_val, Y_val, 
              beta=1.0, patience=10, min_delta=1e-4, verbose=False, print_every=200):
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        recon_losses = []
    
        for epoch in range(epochs):
            
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X_train)
            target = Y_train
            
            recon_loss = torch.nn.MSELoss()(recon, target)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()
            
            recon_losses.append(recon_loss.item())
            
            if verbose and (epoch + 1) % print_every == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
        
        return recon_losses
        
        # train_losses = []
        # val_losses = []
        # best_val_loss = float('inf')
        # epochs_no_improve = 0
    
        # for epoch in range(epochs):
            
        #     # Training step
        #     self.model.train()
        #     optimizer.zero_grad()
        #     recon, mu, logvar = self.model(X_train)
            
        #     target = Y_train
        #     loss, recon_loss, kl_loss = compute_loss(recon, target, mu, logvar, beta=beta)
        #     loss.backward()
        #     optimizer.step()
        #     train_losses.append(recon_loss.item())
    
        #     # Validation step
        #     self.model.eval()
        #     with torch.no_grad():
        #         recon_val, mu_val, logvar_val = self.model(X_val)
        #         val_target = Y_val
        #         _, val_recon_loss, _ = compute_loss(recon_val, val_target, mu_val, logvar_val, beta=beta)
        #         val_losses.append(val_recon_loss.item())
    
        #     # Early stopping check
        #     if val_recon_loss.item() < best_val_loss - min_delta:
        #         best_val_loss = val_recon_loss.item()
        #         epochs_no_improve = 0
        #     else:
        #         epochs_no_improve += 1
    
        #     if verbose and (epoch + 1) % print_every == 0:
        #         print(f'Epoch {epoch+1}/{epochs}, Train Recon: {recon_loss.item():.4f}, '
        #               f'Val Recon: {val_recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
        #     if epochs_no_improve >= patience:
        #         if verbose:
        #             print(f"Early stopping at epoch {epoch+1}. Best val recon loss: {best_val_loss:.4f}")
        #         break
    
        # return train_losses, val_losses



class Metadata: 

    def __init__(self):

        self.xres = np.nan;
        self.yres = np.nan;
        self.channels = [];
        self.conditions = [];

    def conditionStartPos(self, condition):
        # provide the first position for absa given condition
        print('TODO: implement')

    @property
    def nChannels(self):
        return len(self.channels)

        
class Position:
    
    def __init__(self, data, posID, meta, features):

        self.cellData = dict();
        
        self.nCells = data.shape[0]
        self.condition = data['condition'].iloc[0]; # called well in matlab
        
        self.cellData['XY'] = data[['X','Y']];
        self.cellData['features'] = data[features];
        self.cellData['intensities'] = data[meta.channels];
        
        self.ID = posID

    def scatter(self, channel, ms=1, vmin=None, vmax=None, ax=None, thresh=None, cmap='YlGnBu', tol=(1,99)):
        # make a scatter plot of the colony
        # 
        # channel: color channel 
        # ms : scatter point size
        # vmin, vmax : min and max color

        if ax==None:
            fig, ax = plt.subplots(1,1)
        
        color = self.cellData['intensities'][channel]
        order = color.sort_values().index;

        if thresh!=None:
            color = color > thresh
        else:
            if vmin is None:
                vmin = np.percentile(color, tol[0])
            if vmax is None:
                vmax = np.percentile(color, tol[1])

        X = (self.cellData['XY']['X'] - self.center[0])*self.resolution
        Y = (self.cellData['XY']['Y'] - self.center[1])*self.resolution
        
        ax.scatter(X[order], Y[order], s=ms, c=color[order], cmap=cmap, vmin=vmin, vmax=vmax ,edgecolors='none')
        ax.set_aspect('equal') 
        ax.axis('off');

        return (vmin, vmax)

    def scatter_fates(self, ms=1, legend=True, ax=None, thresh=1, fate='all'):
        
        data = self.cellData['intensities'];
        fates, fate_names = return_fates(data, thresh)
        colors = fns_plot.return_colmaps('fates')

        X = (self.cellData['XY']['X'] - self.center[0])*self.resolution
        Y = (self.cellData['XY']['Y'] - self.center[1])*self.resolution
        
        if ax==None:
            fig, ax = plt.subplots(1,1)
        
        for i, f in enumerate(fate_names):

            idx = fates == f;
            
            if (fate == 'all') or (fate == f):
                scatter = ax.scatter(X[idx], Y[idx], color=colmap_fates[f], s=ms, edgecolors='none');
            else:
                scatter = ax.scatter(X[idx], Y[idx], color='lightgray', s=ms, edgecolors='none');
                
        if legend:
            ax.legend(fate_names)
        ax.set_aspect('equal') 
        ax.axis('off');
        
class Colony(Position):
    # Colony extends Position to include features and methods specific to disc-shaped micropatterned colonies, like radiusMicron and makeRadialProfile(..)

    def __init__(self, data, posID, meta, features, nominalRadius):

        super().__init__(data, posID, meta, features)

        self.resolution = meta.xres
        self.radiusMicron = nominalRadius
        self.radiusPixel = nominalRadius/meta.xres
        
        self.center = np.array(data[['X','Y']].mean()) # could further clean up by excluding cells outside the colony as in matlab
        radialOffset = 0 # np.mean(np.sqrt(data[data['MetricDist'] == 0].nucArea/np.pi))*meta.xres # optional if we want to compensate for the fact that the true edge is the mean nuclear location of the nuclei on the edge
        self.cellData['XY'] = self.cellData['XY'].assign(edgeDist=data['CircleEdgeDist'] + radialOffset)
        self.trueRadiusMicron = data['RadialDist'].iloc[0] + data['CircleEdgeDist'].iloc[0] + radialOffset # the true radius is not explicitly saved here but can be recovered like this

        
    def calcRadialProfiles(self, cellsPerBin=100, overlap=50, dr=10):
        # overlap is not really necessary if we do a Gaussian smoothing afterwards, which works better
        # dr : spacing of regular radial grid on which interpolation is performed
        
        # first create bins with equal numbers of cells and calculate the mean edgeDist & mean,std intensity in those bins
        Nbins = round(self.nCells/cellsPerBin);
        edges = np.linspace(0,  self.nCells-1, Nbins+1).round().astype(int);

        Rs = self.cellData['XY']['edgeDist'];
        I = Rs.sort_values().index;
        
        r_tmp = np.zeros(Nbins)
        profile_tmp = pd.DataFrame(columns=self.cellData['intensities'].columns, index=range(Nbins))
        profile_tmp_std = pd.DataFrame(columns=self.cellData['intensities'].columns, index=range(Nbins))
        
        for i in range(Nbins):
            
            start = max(edges[0], edges[i]-overlap)
            stop = min(edges[i+1]+overlap, edges[-1])
            ptidx = np.arange(start, stop+1) 
            
            sel = I[ptidx]
            r_tmp[i] = np.mean(Rs[sel])
            profile_tmp.loc[i, :] = np.nanmean(self.cellData['intensities'].loc[sel, :], axis=0)
            profile_tmp_std.loc[i, :] = np.nanstd(self.cellData['intensities'].loc[sel, :], axis=0)

        r_tmp[-1] = self.trueRadiusMicron; # for center bin, the r value should be the center, not the average r value of the points in it
        
        # now interpolate on evenly spaced radial bins that allow easy averaging between colonies
        margin = self.radiusMicron/10;
        maxR = self.radiusMicron + margin
        # I am adding 10% negative R values to the grid on which it will linearly extrapolate, to deal with boundary effects for smoothing on the edge 
        # (otherwise positional error goes high on edge because mirroring the data for smoothing makes the gradient zero there)
        Ngrid = round((maxR + margin)/dr)
        self.radialGrid = np.linspace(-margin, maxR, Ngrid)
        
        self.radialProfiles = pd.DataFrame(columns=self.cellData['intensities'].columns, index=range(Ngrid))
        self.radialProfiles_std = pd.DataFrame(columns=self.cellData['intensities'].columns, index=range(Ngrid))
        
        for channel in self.cellData['intensities'].columns:
            
            interpolator = interp.interp1d(r_tmp, profile_tmp[channel], kind='linear', fill_value='extrapolate', bounds_error=False)
            self.radialProfiles[channel] = interpolator(self.radialGrid);

            interpolator = interp.interp1d(r_tmp, profile_tmp_std[channel], kind='linear', fill_value='extrapolate', bounds_error=False)
            self.radialProfiles_std[channel] = interpolator(self.radialGrid);

    def calcPosError(self, sigma=1):
        # sig: standard deviation (in bins) for Gaussian smoothing of profile and std

        r = self.radialGrid
        y = self.radialProfiles.apply(lambda col: gaussian_filter1d(col, sigma=sigma))
        yerr = self.radialProfiles_std.apply(lambda col: gaussian_filter1d(col, sigma=sigma))
        dy = y.apply(lambda col: np.gradient(col,r))
        self.posError = yerr/np.abs(dy)*(100/self.radiusMicron);

            
    def plotRadialProfile(self, channel):
        # plot radial mean and std for some channel
        
        r = self.radialGrid
        y = self.radialProfiles[channel]
        yerr = self.radialProfiles_std[channel]
        # cut off the plots where there is no data
        # actual bin radii may be off from true radius by dr so keeping some margin, also dr stored after Colony.calcRadialProfiles, clean up later
        dr = 10
        idx = r < self.trueRadiusMicron + dr
        
        plt.plot(r[idx], y[idx])
        plt.fill_between(r[idx], y[idx] - yerr[idx], y[idx] + yerr[idx], alpha=0.3, color='blue', edgecolor='none')
        
        plt.gca().set_box_aspect(1)
        plt.ylabel("intensity")
        plt.xlabel(r"edge distance ($\mu m$)")
        plt.xlim((0, self.radiusMicron));

    def plotPosError(self, channel, mode='all'):

        r = self.radialGrid
        perr = self.posError[channel]
        # cut off the plots where there is no data
        # actual bin radii may be off from true radius by dr so keeping some margin, also dr stored after Colony.calcRadialProfiles, clean up later
        dr = 10
        idx = r < self.trueRadiusMicron + dr

        plt.plot(r[idx], perr[idx])
        plt.gca().set_box_aspect(1)
        plt.ylabel('pos error (%)')
        plt.xlabel(r"edge distance ($\mu m$)")
        
        plt.ylim(0,30)
        plt.xlim(0, self.radiusMicron)


class Experiment:
    # collect all data associated with an experiment and functions to analyze at that level
    
    def __init__(self, positions, meta):

        self.meta = meta;
        self.positions = positions;


class MPexperiment(Experiment):
    # extend general experiment to the case of disc-shaped micropatterned colonies
    
    def __init__(self, colonies, meta):

        super().__init__(colonies, meta);

        self.radiusMicron = dict()
        self.trueRadiusMicron = dict()
        self.radialProfiles = dict()
        self.radialProfiles_std = dict()
        self.radialProfilesTotal = dict()
        self.radialProfilesTotal_std = dict()
        self.radialProfilesTotal_cov = dict()
        self.radialGrids = dict()
        self.posErrorTotal = dict()
        self.posErrorExtrinsic = dict()
        self.posErrorIntrinsic = dict()
        
    def calcRadialProfiles(self, cellsPerBin=200, overlap=100):
        # overlap is not really necessary here if we do a Gaussian smoothing afterwards, which works better
        
        for cond in self.meta.conditions:

            conditionCols = [c for c in self.positions.values() if c.condition==cond]
            
            # we define the radius for all data combined a the maximum radius of the individual colonies
            self.trueRadiusMicron[cond] = max([c.trueRadiusMicron for c in self.positions.values() if c.condition==cond])
            
            #-------------------------------
            # mean and std of colony means
            #-------------------------------
        
            conditionProfiles = [c.radialProfiles for c in conditionCols];
            self.radialGrids[cond] = conditionCols[1].radialGrid;
            self.radiusMicron[cond] = conditionCols[1].radiusMicron;
            self.radialProfiles[cond] = pd.concat(conditionProfiles).groupby(level=0).mean();
            self.radialProfiles_std[cond] = pd.concat(conditionProfiles).groupby(level=0).std();
    
            #-----------------------
            # total mean and cov 
            #-----------------------

            nCellsCond = np.sum([c.nCells for c in conditionCols])
            XYCond = pd.concat([c.cellData['XY'] for c in conditionCols]);
            
            # first create bins with equal numbers of cells and calculate the mean edgeDist & mean,std intensity in those bins
            Nbins = round(nCellsCond/cellsPerBin);
            edges = np.linspace(0,  nCellsCond-1, Nbins+1).round().astype(int);
            print(cond + ', Nbins = ' + str(Nbins));
            
            Rs = pd.concat([c.cellData['XY']['edgeDist'] for c in conditionCols]);
            intensities = pd.concat([c.cellData['intensities'] for c in conditionCols]);

            I = Rs.sort_values().index;
                
            r_tmp = np.zeros(Nbins)
            profile_tmp = pd.DataFrame(columns=intensities.columns, index=range(Nbins))
            profile_tmp_cov = np.zeros((Nbins, self.meta.nChannels, self.meta.nChannels));
            
            for i in range(Nbins):

                start = max(edges[0], edges[i]-overlap)
                stop = min(edges[i+1]+overlap, edges[-1])
                ptidx = np.arange(start, stop+1) 

                sel = I[ptidx]
                r_tmp[i] = np.mean(Rs[sel])
                profile_tmp.loc[i, :] = np.nanmean(intensities.loc[sel, :], axis=0)
                profile_tmp_cov[i,:,:] = np.cov(intensities.loc[sel, :].T);
            
            r_tmp[-1] = self.trueRadiusMicron[cond] # max(Rs); # for center bin, the r value should be the center, not the average r value of the points in it
            
            # now interpolate on evenly spaced radial bins that allow easy averaging between colonies
            self.radialProfilesTotal[cond] = pd.DataFrame(columns=intensities.columns)
            Ngrid = self.radialGrids[cond].shape[0];
            self.radialProfilesTotal_cov[cond] = np.zeros((Ngrid, self.meta.nChannels, self.meta.nChannels))
            
            for channel in self.meta.channels:
                
                # if you get an error from interpolate saying invalid "invalid value encountered in divide", it is because the same r_tmp value occurs twice because cellPerBin < 2* number of cells with edgeDist 0
                interpolator = interp.interp1d(r_tmp, profile_tmp[channel], kind='linear', fill_value='extrapolate', bounds_error=False)
                self.radialProfilesTotal[cond][channel] = interpolator(self.radialGrids[cond]);

            # define std in terms of covariance matrix for consistency
            self.radialProfilesTotal_std[cond] = pd.DataFrame(columns=intensities.columns)
            for i in range(self.meta.nChannels):
                for j in range(self.meta.nChannels):
                    
                    interpolator = interp.interp1d(r_tmp, profile_tmp_cov[:,i,j], kind='linear', fill_value='extrapolate', bounds_error=False)
                    self.radialProfilesTotal_cov[cond][:,i,j] = interpolator(self.radialGrids[cond]);

                # np.abs added because extrapolation below r=0 leads to negative values and warning sometimes, but those values play no role anyway
                self.radialProfilesTotal_std[cond][self.meta.channels[i]] = np.sqrt(np.abs(self.radialProfilesTotal_cov[cond][:,i,i]))

    def calcPosError(self, totalsets=None, sigma=1):
            # sig: standard deviation (in bins) for Gaussian smoothing of profile and std
            # totalsets: dictionary with sets of channels for which to calculte total pos error, e.g. signals

            for cond in self.meta.conditions:

                r = self.radialGrids[cond]
                Ngrid = r.shape[0]
                R = self.radiusMicron[cond]
                
                # pos error for individual channels based on colony mean and variance (extrinsic error)
                #------------------------------------------------------------------------------------------------
                y = self.radialProfiles[cond].apply(lambda col: gaussian_filter1d(col, sigma=sigma))
                yerr = self.radialProfiles_std[cond].apply(lambda col: gaussian_filter1d(col, sigma=sigma))
                dy = y.apply(lambda col: np.gradient(col,r))
                self.posErrorExtrinsic[cond] = yerr/np.abs(dy)*(100/self.radiusMicron[cond]);
                
                # total pos error for subsets
                conditionCols = [c for c in self.positions.values() if c.condition==cond]
                
                for subset_key, subset in totalsets.items():

                    # calculate covariance matrix over colonies for each radial bin
                    conditionProfiles = [c.radialProfiles[subset] for c in conditionCols] 
                    cube = np.stack([df.values for df in conditionProfiles], axis=0)
                    cov = np.zeros((Ngrid, len(subset), len(subset)))
                    for k in range(Ngrid):
                        cov[k,:,:] = np.cov(cube[:,k,:].T) # np.cov expects shape (num_columns, num_dataframes), so transpose

                    # smooth the covariance
                    for i in range(len(subset)):
                        for j in range(len(subset)):
                            cov[:,i,j] = gaussian_filter1d(cov[:,i,j], sigma=sigma)

                    # invert
                    invcov = np.zeros((Ngrid, len(subset), len(subset)))
                    for k in range(Ngrid):
                        invcov[k,:,:] = np.linalg.pinv(cov[k,:,:]) # The covariance matrix is normal. For a normal matrix, the pseudoinverse ⁠annihilates the kernel of ⁠ A and acts as a traditional inverse of ⁠ A {\displaystyle A}⁠ on the subspace orthogonal to the kernel. 
                    
                    y = self.radialProfiles[cond][subset].apply(lambda col: gaussian_filter1d(col, sigma=1))
                    dy = y.apply(lambda col: np.gradient(col,r)).to_numpy();
                    
                    perr_tot = np.zeros(Ngrid);
                    for k in range(Ngrid):
                        perr_tot[k] = 1/np.sqrt(dy[k,:].dot(invcov[k,:,:].dot(dy[k,:])))*(100/self.radiusMicron[cond])

                    self.posErrorExtrinsic[cond][subset_key] = perr_tot
                    
                # pos error for individual channels based on mean and variance of all cells combined (total error)
                #------------------------------------------------------------------------------------------------
                y = self.radialProfilesTotal[cond].apply(lambda col: gaussian_filter1d(col, sigma=sigma))
                yerr = self.radialProfilesTotal_std[cond].apply(lambda col: gaussian_filter1d(col, sigma=sigma))
                dy = y.apply(lambda col: np.gradient(col,r))
                perr = yerr/np.abs(dy)*(100/self.radiusMicron[cond])
                self.posErrorTotal[cond] = perr

                # total pos error for subsets
                for subset_key, subset in totalsets.items():
                
                    # restrict cov to subset before inverting
                    subidx = [i for i,s in enumerate(self.meta.channels) if s in subset];
                    cov = self.radialProfilesTotal_cov[cond][np.ix_(range(Ngrid), subidx, subidx)]
                    # smooth the coveriance in space 
                    for i in range(len(subidx)):
                        for j in range(len(subidx)):
                            cov[:,i,j] = gaussian_filter1d(cov[:,i,j], sigma=sigma)
                            
                    invcov = np.zeros((Ngrid, len(subidx), len(subidx)))
                    for k in range(Ngrid):
                        invcov[k,:,:] = np.linalg.pinv(cov[k,:,:])
                        
                    y = self.radialProfilesTotal[cond][subset].apply(lambda col: gaussian_filter1d(col, sigma=sigma))
                    dy = y.apply(lambda col: np.gradient(col,r)).to_numpy();
                    perr_tot = np.zeros(Ngrid);
                    for k in range(Ngrid):
                        perr_tot[k] = 1/np.sqrt(np.abs(dy[k,:].dot(invcov[k,:,:].dot(dy[k,:]))))*(100/R)                        
                        
                    self.posErrorTotal[cond][subset_key] = perr_tot

                # pos error for individual channels based on average of single cell error of individual colonies 
                #------------------------------------------------------------------------------------------------
                # is this the right way to get intrinsic error? 
                conditionCols = [c for c in self.positions.values() if c.condition==cond]
                conditionPosError = [c.posError for c in conditionCols]
                self.posErrorIntrinsic[cond] = pd.concat(conditionPosError).groupby(level=0).mean()
                
                # could also sum variances in sigma_r(r*) (below), makes no difference, or average numerator and denominator separately
                # conditionPosVars = [c.posError**2 for c in conditionCols];
                # self.posErrorIntrinsic[cond] = pd.concat(conditionPosVars).groupby(level=0).mean().applymap(np.sqrt)
            
    def plotRadialProfiles(self, channel, condition, mode='cells', sigma=0, color = 'blue', ax=None, normalize=False):

        r = self.radialGrids[condition];
        dr = 10
        idx = r < self.trueRadiusMicron[condition] + dr 
    
        if ax is None:
            _, ax = plt.subplots(1,1)
        
        if mode=='cells':
            y = self.radialProfilesTotal[condition][channel]
            yerr = self.radialProfilesTotal_std[condition][channel]
            if sigma>0:
                y = gaussian_filter1d(y, sigma=sigma)
                yerr = gaussian_filter1d(yerr, sigma=sigma)
            if normalize:
                yerr = yerr/(max(y)-min(y))
                y = y/(max(y)-min(y))
            h, = ax.plot(r[idx], y[idx], color=color)
            ax.fill_between(r[idx], y[idx] - yerr[idx], y[idx] + yerr[idx], alpha=0.3, color=color, edgecolor='none')
        
        elif mode=='colonies':
            y = self.radialProfiles[condition][channel]
            yerr = self.radialProfiles_std[condition][channel]
            if normalize:
                yerr = yerr/(max(y)-min(y))
                y = (y-min(y))/(max(y)-min(y))
                # yerr = yerr/max(y)
                # y = y/max(y)
            h, = ax.plot(r[idx], y[idx], color=color)
            ax.fill_between(r[idx], y[idx] - yerr[idx], y[idx] + yerr[idx], alpha=0.3, color=color, edgecolor='none')
            # ADD THIS: return statement was missing for colonies mode
            
        elif mode=='colonies_individual':
            conditionCols = [c for c in self.positions.values() if c.condition==condition]
            h = []  # Initialize as list
            for c in conditionCols:
                line, = ax.plot(r[idx], c.radialProfiles[channel][idx], color=color)
                h.append(line)
            ax.legend([c.ID for c in conditionCols])
            return h  # Return list of handles for individual mode
        
        ax.set_box_aspect(1)
        ax.set_ylabel("intensity")
        ax.set_xlabel(r"edge distance ($\mu m$)")
        ax.set_xlim((0, self.radiusMicron[condition]))
        ax.set_ylim(bottom=0)    
    
        return h 
    
    def plotPosError(self, condition, channel, mode='et', sigma=0):
        # mode : string of first letters of errors to show: (e)xtrinsic, i(ntrinsic), t(otal)
        
        r = self.radialGrids[condition]

        # cut off the plots where there is no data
        # actual bin radii may be off from true radius by dr so keeping some margin, also dr stored after Colony.calcRadialProfiles, clean up later
        dr = 10
        idx = r < self.trueRadiusMicron[condition] + dr 

        if 'e' in mode:
            perr = self.posErrorExtrinsic[condition][channel]
            if sigma>0:
                perr = gaussian_filter1d(perr, sigma=sigma)
            plt.plot(r[idx], perr[idx])
            
        if 'i' in mode:
            perr = self.posErrorIntrinsic[condition][channel]
            if sigma>0:
                perr = gaussian_filter1d(perr, sigma=sigma)
            plt.plot(r[idx], perr[idx])
            
        if 't' in mode:
            perr = self.posErrorTotal[condition][channel]
            if sigma>0:
                perr = gaussian_filter1d(perr, sigma=sigma)
            plt.plot(r[idx], perr[idx])
            
        plt.gca().set_box_aspect(1)
        plt.ylabel('pos error (%)')
        plt.xlabel(r"edge distance ($\mu m$)")

        plt.ylim(0,30)
        plt.xlim(0, self.radiusMicron[condition])
