import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter1d
import sklearn
import torch
import torch.optim as optim

import sys
sys.path.append('/Users/idse/repos/signaldecoding/2D_gastruloids_v5')
import fns_plotting_scripts as fns_plot
import fns_NN
    
def return_fates(data, thresh=1):
    """
    define fates based on fate marker expression
    """
    
    fate_names = ['AMLC','PGCLC','PSLC','meso','pluri','ecto', 'endo', 'other']

    if thresh==1:
        thresh = {'TFAP2C':1, 'SOX17':1, 'NANOG':1, 'ISL1':1, 'TBXT':1, 'TBX6':1, 'SOX2':1}
    
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
    endo = SOX17 & ~PGCLC
    meso = TBXT & TBX6 & ~PGCLC & ~endo
    AMLC = ISL1 & ~PGCLC & ~meso & ~endo  
    ecto = ~NANOG & (SOX2_scaled > TBXT_scaled) & (data['SOX2'] > 100)  & ~PGCLC & ~meso & ~endo & ~AMLC 
    PSLC = (TBXT_scaled > NANOG_scaled) & TBXT & ~PGCLC & ~meso & ~endo & ~AMLC & ~ecto # (data['TBXT'] > 100)
    pluri = (NANOG_scaled > TBXT_scaled) & NANOG & SOX2 & ~PGCLC & ~meso & ~endo & ~AMLC & ~ecto & ~PSLC 
    AMLC = (AMLC | TFAP2C) & ~(ecto | pluri | PSLC | meso | PGCLC | endo)
    other = ~(ecto | pluri | PSLC | AMLC | meso | PGCLC | endo)

    # # OLD defs
    # PGCLC = TFAP2C & SOX17 
    # meso = TBXT & TBX6 & ~PGCLC
    # AMLC = ISL1 & ~PGCLC & ~meso
    # PSLC = TBXT & ~SOX2 & ~TBX6 & ~PGCLC & ~AMLC
    # pluri = SOX2 & NANOG & ~PGCLC & ~meso & ~PSLC & ~AMLC
    # ecto = ~NANOG & SOX2 & ~PGCLC & ~meso & ~PSLC & ~AMLC 
    # other = ~(ecto | pluri | PSLC | AMLC | meso | PGCLC)

    labels = np.empty(data.shape[0], dtype='<U5')  # or dtype=str
    labels[PGCLC] = "PGCLC"
    labels[meso] = "meso"
    labels[endo] = "endo"
    labels[AMLC] = "AMLC"
    labels[PSLC] = "PSLC"
    labels[pluri] = "pluri"
    labels[ecto] = "ecto"
    labels[other] = "other"
    
    return labels, fate_names

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

    
    # def train(self, X_data, Y_data, epochs=800, lr=1e-3, 
    #             beta=1.0, verbose=False, print_every=200):
    #     """
    #     Train VIB model
        
    #     Parameters:
    #     -----------
    #     model : nn.Module
    #         VAE or VIB model
    #     X_data : torch.Tensor
    #         Input data
    #     Y_data : torch.Tensor
    #         Output data (used for VIB, ignored for VAE)
    #     is_vae : bool
    #         If True, train as VAE (X->X), else train as VIB (X->Y)
    #     epochs : int
    #         Number of training epochs
    #     lr : float
    #         Learning rate
    #     beta : float
    #         Weight for KL divergence
    #     verbose : bool
    #         If True, print training progress
    #     print_every : int
    #         Print progress every N epochs
            
    #     Returns:
    #     --------
    #     recon_losses : list
    #         List of reconstruction losses per epoch
    #     """
    #     optimizer = optim.Adam(self.model.parameters(), lr=lr)
    #     recon_losses = []
        
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
            
    #         recon, mu, logvar = self.model(X_data)
    #         target = Y_data

    #         # compute loss
    #         recon_loss = torch.nn.MSELoss()(recon, target)
    #         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    #         loss = recon_loss + beta*kl_loss
            
    #         loss.backward()
    #         optimizer.step()
            
    #         recon_losses.append(recon_loss.item())
            
    #         if verbose and (epoch + 1) % print_every == 0:
    #             print(f'  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
    #                   f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
        
    #     return recon_losses
    

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

    def train(self, X_train, Y_train, X_val, Y_val, epochs=800, lr=1e-3,
              beta=1.0, patience=10, min_delta=1e-4, verbose=False, print_every=200):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
    
        for epoch in range(epochs):
            # Training step
            self.model.train()
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X_train)
            target = Y_train
            loss, recon_loss, kl_loss = compute_loss(recon, target, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_losses.append(recon_loss.item())
    
            # Validation step
            self.model.eval()
            with torch.no_grad():
                recon_val, mu_val, logvar_val = self.model(X_val)
                val_target = Y_val
                _, val_recon_loss, _ = compute_loss(recon_val, val_target, mu_val, logvar_val, beta=beta)
                val_losses.append(val_recon_loss.item())
    
            # Early stopping check
            if val_recon_loss.item() < best_val_loss - min_delta:
                best_val_loss = val_recon_loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
    
            if verbose and (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Recon: {recon_loss.item():.4f}, '
                      f'Val Recon: {val_recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best val recon loss: {best_val_loss:.4f}")
                break
    
        return train_losses, val_losses


# # convert data to David's format 
# def data2david(data, features):

#     # Step 1: Features and Colonies
#     df = data[features + ['Colony']]
#     features = [c for c in df.columns if c != 'Colony']
#     colonies = sorted(df['Colony'].unique())
#     n_colonies = len(colonies)
#     n_features = len(features)

#     # Step 2: Max cells per colony
#     max_cells = df.groupby('Colony').size().max()

#     # Initialize array: (colonies, cells, features)
#     arr = np.full((n_colonies, max_cells, n_features), np.nan)
    
#     # Fill
#     for i, colony_num in enumerate(colonies):
#         colony_data = df[df['Colony'] == colony_num][features].values
#         arr[i, :len(colony_data), :] = colony_data
    
#     return arr
    
# def test_train_split_colonies(feature, target, train_size=3):

#     # Split test/train by colony
#     N_sys = feature.shape[0]
#     N_tar = target.shape[2]
    
#     test_size = N_sys - train_size
    
#     # Use first N=test_size colonies for testing, so colony 1 can be used for plotting
#     feature_test = feature[:test_size, :, :]
#     target_test = target[:test_size, :, :]
    
#     # Use last colonies for training
#     feature_train = feature[-train_size:, :, :] 
#     target_train = target[-train_size:, :, :]

#     feat_train, tar_train, _ = fns_NN.clean_data(feature_train, target_train)
#     feat_test, tar_test, _ = fns_NN.clean_data(feature_test, target_test)

#     if N_tar == 1:
#         tar_train = tar_train.ravel()
#         tar_test = tar_test.ravel()
    
#     return (feat_train, feat_test, tar_train, tar_test)


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

    def scatter(self, channel, ms=1, vmin=None, vmax=None, ax=None, thresh=None):
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
                vmin = np.percentile(color, 1)
            if vmax is None:
                vmax = np.percentile(color, 99)
        
        ax.scatter(self.cellData['XY']['X'][order], self.cellData['XY']['Y'][order], s=ms, c=color[order], cmap='YlGnBu', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal') 
        ax.axis('off');

        return (vmin, vmax)

    def scatter_fates(self, ms=10, legend=True, ax=None, thresh=1):
        
        data = self.cellData['intensities'];
        marker_clusters, fate_names = return_fates(data, thresh)
        colors = fns_plot.return_colmaps('fates')

        if ax==None:
            fig, ax = plt.subplots(1,1)
        
        for i, cl in enumerate(fate_names):

            idx = marker_clusters==cl;
            X = self.cellData['XY']['X'][idx]
            Y = self.cellData['XY']['Y'][idx]
            scatter = ax.scatter(X,Y,color=colors[i], s=ms, edgecolors='none');
            
        if legend:
            ax.legend(fate_names)
        ax.set_aspect('equal') 
        ax.axis('off');
        
class Colony(Position):
    # Colony extends Position to include features and methods specific to disc-shaped micropatterned colonies, like radiusMicron and makeRadialProfile(..)

    def __init__(self, data, posID, meta, features, nominalRadius):

        super().__init__(data, posID, meta, features)

        self.radiusMicron = nominalRadius
        self.radiusPixel = nominalRadius/meta.xres
        
        self.center = data[['X','Y']].mean() # could further clean up by excluding cells outside the colony as in matlab
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
