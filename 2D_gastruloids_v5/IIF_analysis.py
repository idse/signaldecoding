import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy as sp
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter1d
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score

from scipy import stats

import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#print("Using device:", device)
import torch.optim as optim

import os
from skimage import io as imio
from skimage import exposure
import sys
sys.path.append('/Users/idse/repos/signaldecoding/2D_gastruloids_v5')
import fns_plotting_scripts as fns_plot
import fns_NN

from scipy.stats import ttest_rel
import statsmodels.stats
import statsmodels.stats.multitest
import time
import warnings

colmap_fates = {'AMLC':[90/255,166/255,71/255,1],'PGCLC':[227/255,143/255,52/255,1],
                 'PSLC':[211/255,62/255,43/255,1], 'meso':[140/255,40/255,93/255,1],
                 'pluri':[75/255,167/255,158/255,1], 'ecto':[49/255,118/255,181/255,1], 
                'endo':[227/255,179/255,61/255,1],'other':[0.8,0.8,0.8,1],'junk':[0.8,0.8,0.8,1]} # junk for backwards compatibility

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


def makeRGBbase(col, coli, markers, dataDir, crop_margin, center_x_margin, center_y_margin,
                rdStr='Rd', Ilim=None, percentiles=None, tol=None,
                partial_MIP=False, starting_slice=6, stack_length=5, pie_sectors=False,
                grayscale_background=False, outer_margin=50, boundary_width=7):
    """Build and return the base RGB composite."""

    def load_MIP(mrkr):
        if partial_MIP:
            return makeMIPs(coli, mrkr, dataDir, rdStr, starting_slice, stack_length)
        fname = getImageFilename(coli, mrkr, dataDir, rdStr, imtype='MIP')
        print('loading', fname)
        return imio.imread(fname)

    def get_Ilimits(MIP, mrkr):
        if Ilim is not None and mrkr in Ilim:
            lo, hi = Ilim[mrkr]
        elif percentiles is not None and mrkr in percentiles:
            lo, hi = np.percentile(MIP[MIP > 0], percentiles[mrkr])
        else:
            lo, hi = np.percentile(MIP[MIP > 0], (tol or {}).get(mrkr, [1, 99]))
        return float(lo), float(hi)

    def rescale(MIP, mrkr):
        return exposure.rescale_intensity(MIP, in_range=get_Ilimits(MIP, mrkr), out_range=(0, 255))

    # Load & rescale markers
    MIPca  = {m: rescale(load_MIP(m), m) for m in markers}
    center = (col[coli].cellData['XY'].loc[:, ['X', 'Y']].mean(axis=0)
          + [center_x_margin, center_y_margin]).to_numpy() 
    radius = col[coli].radiusPixel + crop_margin

    # Pad to guarantee crop never hits boundary
    pad    = int(np.ceil(radius)) + outer_margin
    center = center + pad
    MIPca  = {m: np.pad(img, pad, mode='constant', constant_values=0)
              for m, img in MIPca.items()}

    h, w = list(MIPca.values())[0].shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Build composite
    if pie_sectors and len(markers) > 1:
        N     = len(markers)
        angle = np.mod(np.arctan2(Y - center[1], X - center[0]) + np.pi / 2, 2 * np.pi)

        grayscale_composite = np.zeros((h, w), dtype=np.float32)
        sector_masks        = []
        for i in range(N):
            angle_start = i * 2 * np.pi / N
            angle_end   = (i + 1) * 2 * np.pi / N
            mask_sector = (
                (angle >= angle_start) & (angle < angle_end) if angle_end > angle_start
                else (angle >= angle_start) | (angle < angle_end)
            )
            sector_masks.append(mask_sector)
            grayscale_composite += MIPca[markers[i]] * mask_sector

        from scipy.ndimage import binary_dilation
        lines = np.zeros((h, w), dtype=bool)
        for mask_sector in sector_masks:
            lines |= binary_dilation(mask_sector, structure=create_circular_struct_elem(boundary_width)) ^ mask_sector

        RGBbase = np.stack([grayscale_composite] * 3, axis=2)
        RGBbase[lines] = 255

    else:
        channels  = [MIPca[m].astype(np.float32) for m in markers[:3]]
        channels += [np.zeros((h, w), dtype=np.float32)] * (3 - len(channels))
        RGBbase   = np.stack(channels, axis=2)

    RGBbase = np.clip(RGBbase, 0, 255).astype(np.uint8)

    # Crop to square
    side   = int(np.ceil(radius * 2)) + 2 * outer_margin
    cx, cy = np.round(center).astype(int)
    RGBbase = RGBbase[cy - side // 2 : cy + side // 2,
                      cx - side // 2 : cx + side // 2]


    #  Circular alpha mask
    hh, ww = RGBbase.shape[:2]
    Y2, X2 = np.ogrid[:hh, :ww]
    mask   = (np.sqrt((Y2 - hh / 2) ** 2 + (X2 - ww / 2) ** 2) <= radius).astype(np.uint8) * 255
    RGBbase = np.dstack([RGBbase, mask])

    # Return base image + metadata needed for overlay step
    meta = dict(radius=radius, outer_margin=outer_margin,
                grayscale_background=grayscale_background,
                center=center, pad=pad, h=h, w=w,
                dataDir=dataDir, coli=coli, rdStr=rdStr,
                Ilim=Ilim, percentiles=percentiles, tol=tol)
    return RGBbase, meta


def applyOverlay(RGBbase, meta, overlay_marker, overlay_alpha=1):
    """Apply a single overlay channel to a prebuilt base image."""

    def get_Ilimits(MIP, mrkr):
        Ilim, percentiles, tol = meta['Ilim'], meta['percentiles'], meta.get('tol')
        if Ilim is not None and mrkr in Ilim:
            lo, hi = Ilim[mrkr]
        elif percentiles is not None and mrkr in percentiles:
            lo, hi = np.percentile(MIP[MIP > 0], percentiles[mrkr])
        else:
            lo, hi = np.percentile(MIP[MIP > 0], (tol or {}).get(mrkr, [1, 99]))
        return float(lo), float(hi)

    radius               = meta['radius']
    grayscale_background = meta['grayscale_background']
    outer_margin         = meta['outer_margin']
    pad                  = meta['pad']
    center               = meta['center']

    # Remove alpha before blending 
    if RGBbase.shape[-1] == 4:
        alpha_channel = RGBbase[..., 3]          # save it
        RGBbase_rgb   = RGBbase[..., :3]         # work on RGB only
    else:
        alpha_channel = None
        RGBbase_rgb   = RGBbase

    # Load & rescale overlay marker
    fname       = getImageFilename(meta['coli'], overlay_marker, meta['dataDir'], meta['rdStr'])
    print('loading overlay', fname)
    MIP_overlay = imio.imread(fname)
    MIP_overlay = exposure.rescale_intensity(
        MIP_overlay, in_range=get_Ilimits(MIP_overlay, overlay_marker), out_range=(0, 255)
    )

    # Pad + crop overlay to match base image geometry
    MIP_overlay  = np.pad(MIP_overlay, pad, mode='constant', constant_values=0)
    side         = int(np.ceil(radius * 2)) + 2 * outer_margin
    cx, cy       = np.round(center).astype(int)
    overlay_crop = MIP_overlay[cy - side // 2 : cy + side // 2,
                               cx - side // 2 : cx + side // 2].astype(np.float32) / 255.0

    # Blend onto RGB base
    RGBoverlay = RGBbase_rgb.astype(np.float32)
    if grayscale_background:
        colored    = overlay_crop[..., None] * np.array([0, 1, 1]) * 255
        RGBoverlay = ((1 - overlay_alpha * overlay_crop[..., None]) * RGBoverlay
                      + overlay_alpha * overlay_crop[..., None] * colored)
    else:
        green          = np.zeros((*RGBbase_rgb.shape[:2], 3), dtype=np.float32)
        green[:, :, 0] = overlay_crop * 255
        RGBoverlay    += overlay_alpha * green

    RGBoverlay = np.clip(RGBoverlay, 0, 255).astype(np.uint8)

    # Recompute circular mask
    if alpha_channel is not None:
        RGBoverlay = np.dstack([RGBoverlay, alpha_channel])
    else:
        hh, ww = RGBoverlay.shape[:2]
        Y, X   = np.ogrid[:hh, :ww]
        mask   = (np.sqrt((Y - hh / 2) ** 2 + (X - ww / 2) ** 2) <= radius).astype(np.uint8) * 255
        RGBoverlay = np.dstack([RGBoverlay, mask])

    return RGBoverlay


def makeRGBoverlay_colisectors(cols, colis, marker, dataDirs,
                                crop_margin, reference_cols,
                                center_x_margins, center_y_margins,
                                rdStr='Rd', percentiles=None,
                                partial_MIP=False, starting_slice=6, stack_length=5,
                                outer_margin=0):
    """
    Create a pie-sector composite of grayscale MIPs, one sector per colony.
    Intensity limits are averaged across reference colonies.
    """
    

    def load_MIP(coli, dataDir):
        if partial_MIP:
            return makeMIPs(coli, marker, dataDir, rdStr, starting_slice, stack_length)
        return imio.imread(getImageFilename(coli, marker, dataDir, rdStr))
    

    # Intensity limits from reference colonies
    pct  = percentiles.get(marker, [1, 99]) if percentiles else [1, 99]
    lims = [np.percentile(load_MIP(ref, dataDirs[i])[load_MIP(ref, dataDirs[i]) > 0], pct)
            for i, ref in enumerate(reference_cols)]
    Imin = np.mean([l[0] for l in lims])
    Imax = np.mean([l[1] for l in lims])
    
    # Load, rescale, get centers and radii
    MIPs    = []
    centers = []
    radii   = []
    
    for i, coli in enumerate(colis):
        mip = exposure.rescale_intensity(
            load_MIP(coli, dataDirs[i]).astype(np.float32),
            in_range=(Imin, Imax), out_range=(0, 255)
        )
        MIPs.append(mip)
        centers.append(
            cols[i][coli].cellData['XY'][['X', 'Y']].mean().values
            + [center_x_margins[i], center_y_margins[i]]
        )
        radii.append(cols[i][coli].radiusPixel + crop_margin)
    
    # Canvas setup
    side        = int(np.ceil(max(radii) * 2))
    h = w       = side + 2 * outer_margin
    cx = cy     = h // 2
    Y, X        = np.ogrid[:h, :w]
    
    def crop_to_canvas(mip, center):
        """Crop MIP around center into side x side canvas"""
        cx_src, cy_src = np.round(center).astype(int)
        half           = side // 2
        canvas         = np.zeros((side, side), dtype=np.float32)
        
        y1 = max(cy_src - half, 0);  y2 = min(cy_src + half, mip.shape[0])
        x1 = max(cx_src - half, 0);  x2 = min(cx_src + half, mip.shape[1])
        yo = max(0, half - cy_src);  xo = max(0, half - cx_src)
        
        canvas[yo:yo + (y2 - y1), xo:xo + (x2 - x1)] = mip[y1:y2, x1:x2]
        return canvas
    

    # Pie sectors
    N         = len(colis)
    angle_map = np.mod(np.arctan2(Y - cy, X - cx) + np.pi / 2, 2 * np.pi)
    composite = np.zeros((h, w), dtype=np.float32)
    
    sector_masks = []
    for i in range(N):
        a0   = i * 2 * np.pi / N
        a1   = (i + 1) * 2 * np.pi / N
        mask = (angle_map >= a0) & (angle_map < a1)
        sector_masks.append(mask)
        
        # Place cropped MIP into padded canvas
        cropped             = np.zeros((h, w), dtype=np.float32)
        om                  = outer_margin
        cropped[om:om+side, om:om+side] = crop_to_canvas(MIPs[i], centers[i])
        composite          += cropped * mask
    
    # Sector boundary lines
    from scipy.ndimage import binary_dilation
    struct  = create_circular_struct_elem(10)
    lines   = np.zeros((h, w), dtype=bool)
    for mask in sector_masks:
        lines |= binary_dilation(mask, structure=struct) ^ mask
    composite[lines] = 255
    
    # Circular alpha mask + output
    r_map        = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    alpha        = ((r_map <= max(radii) + outer_margin) * 255).astype(np.uint8)
    gray         = np.clip(composite, 0, 255).astype(np.uint8)
    RGBoverlay   = np.dstack([gray, gray, gray, alpha])
    
    return RGBoverlay
    
def create_circular_struct_elem(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    struct_elem = x**2 + y**2 <= radius**2
    return struct_elem
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


# ------------------------------------------------------------------------------------------------------------
# EXPERIMENT LOADER (generic, project-agnostic)
# ------------------------------------------------------------------------------------------------------------

class ExperimentConfig:
    # Stores project-specific configuration
    def __init__(self, exp_params, signal_names, feature_names):
        self.exp_params    = exp_params
        self.signal_names  = signal_names
        self.feature_names = feature_names


def exp_loader(exp_name, config, normalize=True, normalization_condition='B50'):
    exp_dir, csv_file, rename_dict, conditions, thresh, junk_csv = config.exp_params[exp_name]
    gene_names = list(thresh.keys())
    
    data = pd.read_csv(os.path.join(exp_dir, csv_file))
    data.rename(columns=rename_dict, inplace=True)
    
    if junk_csv:
        junk = pd.read_csv(os.path.join(exp_dir, junk_csv)) == 2
        data = data[~junk.iloc[:, 0].values]
    
    valid_mask = ~(data[config.signal_names + gene_names].isna() |
                   (data[config.signal_names + gene_names] < 0)).any(axis=1)
    data = data[valid_mask].reset_index(drop=True)
    
    if normalize:
        norm_data = data[data['condition'] == normalization_condition]
        norm_cols = config.signal_names + gene_names
        data_z    = data.copy()
        data_z[norm_cols] = (
            (data[norm_cols] - norm_data[norm_cols].mean()) /
             norm_data[norm_cols].std()
        )
    else:
        data_z = data.copy()
    
    meta            = Metadata()
    meta.xres       = meta.yres = 0.325
    meta.channels   = config.signal_names + gene_names
    meta.conditions = conditions
    
    return exp_dir, data, data_z, thresh, meta, gene_names


def create_experiment(data, meta, feature_names, cellsPerBin=50):
    colonies = {
        colID: Colony(data[data['Colony'] == colID], colID, meta,
                      features=feature_names, nominalRadius=350)
        for colID in data['Colony'].unique()
    }
    for col in colonies.values():
        col.calcRadialProfiles(cellsPerBin=cellsPerBin, overlap=0, dr=8)
        col.calcPosError(sigma=1)
    
    exp = MPexperiment(colonies, meta)
    exp.calcRadialProfiles(cellsPerBin=cellsPerBin, overlap=0)
    return colonies, exp


class ExperimentResults:
    def __init__(self, exp_id, config):
        self.exp_dir, self.data, self.data_z, \
        self.thresh, self.meta, self.gene_names = exp_loader(exp_id, config)
        
        self.signal_names  = config.signal_names
        self.feature_names = config.feature_names
        self.col,   self.exp   = create_experiment(self.data,   self.meta, config.feature_names)
        self.col_z, self.exp_z = create_experiment(self.data_z, self.meta, config.feature_names)


def normalize_to_ctrl_profile(data_z, exp_z, signal_names, ctrl_cond='B50'):
   
    # Normalize signals so ctrl radial profile min=0 and max=1
    signal_min = {s: exp_z.radialProfiles[ctrl_cond][s].min() for s in signal_names}
    signal_max = {s: exp_z.radialProfiles[ctrl_cond][s].max() for s in signal_names}
    
    data_norm = data_z.copy()
    for s in signal_names:
        data_norm[s] = (data_z[s] - signal_min[s]) / (signal_max[s] - signal_min[s])
    
    return data_norm, signal_min, signal_max


#----------------------------------------------------------------------------------------------------------------------------------
# ANALYSIS: CROSSTALK PREDICTION (KNN)
#----------------------------------------------------------------------------------------------------------------------------------

from sklearn.neighbors import NearestNeighbors

# Apply various smoothing methods to signal data
def apply_smoothing(signals, k_val, mode='gaussian'):
    # signals: numpy array

    distances, indices = NearestNeighbors(n_neighbors= k_val, n_jobs=-1).fit(
        sklearn.preprocessing.StandardScaler().fit_transform(signals)).kneighbors()
    
    if mode == 'simple':
        neighbor_signals = signals[indices[:, :k_val]]
        smoothed = neighbor_signals.mean(axis=1)
        
    elif mode == 'gaussian':
        neighbor_signals = signals[indices[:, :k_val]]
        sigma = np.maximum(distances[:, k_val-1:k_val], 1e-10)
        weights = np.exp(-0.5 * (distances[:, :k_val] / sigma) ** 2)
        smoothed = np.einsum('ij,ijk->ik', 
                                weights / weights.sum(axis=1, keepdims=True), 
                                neighbor_signals)
    
    return smoothed

#----------------------------------------------------------------------------------

def determine_threshold(data, perturbing_sigs, perturb_mode='inh', use_profile=None, percentile=1):
    # perturb_mode = 'inh' : change to 'act' for activating drug
    # use_profile : pass an experiment object from which to use the radial profile min/max as thresholds

    if perturb_mode == 'inh':
        if use_profile:
            return np.min(use_profile.radialProfiles['B50'][perturbing_sigs])
        else:
            return np.percentile(data[perturbing_sigs], percentile)
    elif perturb_mode == 'act':
        if use_profile:
            return np.max(use_profile.radialProfiles['B50'][perturbing_sigs])
        else:
            return np.percentile(data[perturbing_sigs], 100 - percentile)
    else:
        raise ValueError(f"Unknown perturb_mode: {perturb_mode}")

#----------------------------------------------------------------------------------

# Create perturbed dataset using KNN from low-perturbation cells
def create_perturbed_data(data, signal_names, perturbing_sigs, threshold, perturb_mode='inh', do_naive=True,
                         k_nn=5, distance_signals=None, preserve_signals=None):

    # data: only the reference data for projection (so restrict e.g. to B50 before calling if that is the goal)

    # Z-normalize
    data_z = data.copy()
    data_z[signal_names] = (data[signal_names] - data[signal_names].mean()) / data[signal_names].std()

    # Define subset based on perturbation
    if perturb_mode == 'inh':
        subset_mask = (data[perturbing_sigs].values <= threshold).all(axis=1)
    elif perturb_mode == 'act':
        subset_mask = (data[perturbing_sigs].values >= threshold).all(axis=1)
    else:
        raise ValueError(f"Unknown perturb_mode: {perturb_mode}")

    if do_naive:
        data_perturbed = data.copy()
        data_perturbed[perturbing_sigs] = threshold

        data_perturbed_z = (data_perturbed[signal_names] - data[signal_names].mean()) / data[signal_names].std()
        return data_perturbed, data_perturbed_z

    else:
        data_subset = data[subset_mask]
        data_z_subset = data_z[subset_mask]
        
        
        # Distance signals
        if distance_signals is None:
            distance_signals = [s for s in signal_names if s not in perturbing_sigs]
        
        # Fit KNN and find neighbors
        knn_model = NearestNeighbors(n_neighbors=k_nn, metric='euclidean')
        knn_model.fit(data_z_subset[distance_signals])
        distances, indices = knn_model.kneighbors(data_z[distance_signals])
        
        # Gaussian weighted averaging
        subset_signals = data_subset[signal_names].values
        sigma = np.maximum(distances[:, -1:], 1e-10)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        mean_signals = np.einsum('ij,ijk->ik', 
                                weights / weights.sum(axis=1, keepdims=True), 
                                subset_signals[indices])

        # Gaussian weighted averaging (z_norm)
        subset_signals_z = data_z_subset[signal_names].values
        sigma = np.maximum(distances[:, -1:], 1e-10)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        mean_signals_z = np.einsum('ij,ijk->ik', 
                                weights / weights.sum(axis=1, keepdims=True), 
                                subset_signals_z[indices])
        
        # Create output
        data_perturbed = data.copy()
        data_perturbed[signal_names] = mean_signals

        # Create z-normalized output
        data_perturbed_z = data_z.copy()
        data_perturbed_z[signal_names] = mean_signals_z
        
        # Optional: Preserve specific signals in control
        if preserve_signals:
            for sig in preserve_signals:
                if sig in signal_names:
                    data_perturbed[sig] = data[sig]
                    data_perturbed_z[sig] = data_z[sig]
        
        return data_perturbed, data_perturbed_z


def run_vib_predictions(data_dir, data_train, data_test, signal_names, gene_names, 
                       hyperparam, N_run = 3, output_prefix='vib_pred'):
    
    feat_names = signal_names
    vib = {}
    predictions = {}

    feat_train = data_train[feat_names]
    tar_train = data_train[gene_names]
    feat_test = data_test[feat_names]
    
    for run in range(N_run):
        #print(f"  VIB Run {run+1}/{N_run}")
        start = time.time()

        # set seeds for reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(run)
    
        vib[run] = VIB(feat_train, tar_train, hyperparam)
    
        if not os.path.exists(data_dir + '/model_' + str(run) + '_' + str(len(signal_names)) + 'D' + '.pth'):
            print('run VIB')
            _ = vib[run].train(verbose=False)
            torch.save(vib[run].model, data_dir + '/model_' + str(run) + '_' + str(len(signal_names)) + 'D' + '.pth')
        else:
            print('load VIB')
            vib[run].model = torch.load(data_dir + '/model_' + str(run) + '_' + str(len(signal_names)) + 'D' + '.pth', map_location=device, weights_only=False)
        
        # Predict
        tar_predict = vib[run].predict(feat_test)
        
        pred_df = data_test.copy()
        pred_df.loc[tar_predict.index, tar_predict.columns] = tar_predict
        predictions[run] = pred_df
        
    
    # Average predictions
    avg_pred = predictions[0].copy()
    avg_genes = pd.concat([df[gene_names] for df in predictions.values()]).groupby(level=0).mean()
    avg_pred[gene_names] = avg_genes
    predictions['avg'] = avg_pred
    
    return vib, avg_pred, predictions


def fate_histogram(data_cond, exp_result):
    n_bins = 20  # adjust this number as needed
    thresh_boundary = 0.5 # boundaries are half maximum of fate
    colors = list(colmap_fates.values())
    fate_names_ordered = list(colmap_fates.keys())
    N_fates = len(fate_names_ordered)
        
    fates, fate_names = return_fates(data_cond, thresh=exp_result.thresh)
    
    # bin by radial distance
    binning = pd.qcut(data_cond['CircleEdgeDist'], q=n_bins, labels=False, duplicates='drop').to_frame(name='r_bin')
    binning['fate'] = fates

    # get bin centers in microns
    bin_edges = pd.qcut(data_cond['CircleEdgeDist'], q=n_bins, duplicates='drop', retbins=True)[1]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # fate counts per bin
    fate_counts = binning.groupby(['r_bin', 'fate']).size().unstack(fill_value=0)
    fate_counts = fate_counts.reindex(columns=fate_names_ordered, fill_value=0)

    # normalize per fate (each fate peaks at 1) for boundary detection
    fate_counts_byfate = fate_counts.div(fate_counts.sum(axis=0), axis=1)
    fate_counts_byfate_norm = fate_counts_byfate.div(fate_counts_byfate.max(axis=0).replace(0, 1), axis=1)

    # normalize per bin for bar plot
    fate_counts_norm = fate_counts.div(fate_counts.sum(axis=1), axis=0)
    
    # create the stacked bar graph
    # distribution bar plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bottom = np.zeros(len(bin_centers))
    for fi, fate in enumerate(fate_names_ordered):
        if fate not in fate_counts_norm.columns:
            continue
        vals = fate_counts_norm[fate].values
        ax.bar(bin_centers, vals, width=bin_widths, bottom=bottom,
            color=colors[fi], align='center')
        bottom += vals
    ax.set_xticks([bin_centers[0], bin_centers[-1]], labels=['edge', 'center'])
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlabel(None)
    
    ax.set_box_aspect(0.3)
    sw = 2
    for spine in ax.spines.values():
        spine.set_linewidth(sw)
    

    plt.tight_layout()


def kde_on_grid(L1s, L2s, L1grid_scaled, L2grid_scaled, data_percentile=1):
    values      = np.vstack([L1s, L2s])
    kernel      = sp.stats.gaussian_kde(values)
    kde_at_data = kernel(values)
    vmin        = np.percentile(kde_at_data, data_percentile)
    kde_grid    = kernel(
        np.vstack([L1grid_scaled.ravel(), L2grid_scaled.ravel()])
    ).reshape(L1grid_scaled.shape)
    levels      = np.linspace(vmin, kde_grid.max(), 10)
    print(f"  {np.sum(kde_at_data >= vmin)/len(L1s)*100:.1f}% inside contours")
    return kde_grid, levels

def fit_zscore_alignment(data_self, data_ref, columns, condition='B50'):
    """Self-standardize each column using data_self's own B50 mean/std, then rescale to
    data_ref's B50 mean/std -- proper cross-experiment alignment (unlike a StandardScaler
    fit only on data_ref and applied directly to data_self's raw values, which assumes the
    two already share similar raw scale and doesn't correct for genuine distributional
    differences between experiments)."""
    self_cond = data_self[data_self['condition'] == condition]
    ref_cond = data_ref[data_ref['condition'] == condition]

    align_params = {}
    for col in columns:
        self_mean, self_std = self_cond[col].mean(), self_cond[col].std()
        ref_mean, ref_std = ref_cond[col].mean(), ref_cond[col].std()
        scale = ref_std / self_std
        shift = ref_mean - self_mean * scale
        align_params[col] = (scale, shift)
    return align_params


def _run_vib_self_impl(data, feat_names, gene_names, hyperparam, N_run, out_dir, out_prefix,
                        seed_offset, label, load_results, allow_training, recompute=False):
    """Shared implementation for run_vib_self/load_vib_self -- see those docstrings. Not
    called directly. Leave-one-colony-out CV on data (caller pre-filters to whatever
    condition(s) should participate, e.g. data[data['condition']=='B50'] -- pooling
    multiple conditions from the SAME experiment needs no alignment, since they share the
    same imaging session/calibration; run_vib_joint_experiments handles pooling across
    DIFFERENT experiments, which does need alignment)."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    colony_idx = np.unique(data['Colony'])
    tag = f'[{label}] ' if label else ''
    runs = {}
    for it in range(N_run):
        out_path = f'{out_dir}/{out_prefix}_run{it}.csv' if out_dir else None
        if out_path is not None and not recompute and os.path.exists(out_path):
            if load_results:
                print(f'{tag}Run {it}/{N_run} -- already exists, loading: {out_path}')
                runs[it] = pd.read_csv(out_path, index_col=0)
            else:
                print(f'{tag}Run {it}/{N_run} -- already exists, skipping load: {out_path}')
            continue
        if not allow_training:
            raise FileNotFoundError(
                f'{tag}Run {it}/{N_run} not cached at {out_path} -- run_vib_self must be run '
                f'first to train it; load_vib_self never trains.')
        print(f'{tag}Run {it}/{N_run}')
        start = time.time()
        torch.manual_seed(seed_offset + it)
        np.random.seed(seed_offset + it)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed_offset + it)
        pred_df = data.copy()
        for test_colony in colony_idx:
            print(f'{tag}  col: {test_colony}')
            train_colonies = np.setdiff1d(colony_idx, test_colony)
            data_test = data[data['Colony'].isin([test_colony])]
            data_train = data[data['Colony'].isin(train_colonies)]
            vib = VIB(data_train[feat_names], data_train[gene_names], hyperparam)
            _ = vib.train(verbose=False)
            tar_predict = vib.predict(data_test[feat_names])
            pred_df.loc[tar_predict.index, tar_predict.columns] = tar_predict
        if out_path is not None:
            pred_df.to_csv(out_path)
        if load_results:
            runs[it] = pred_df
        print(f'{tag}Elapsed time: {time.time() - start:.1f} seconds')
    if not load_results:
        return None
    avg_pred = runs[0].copy()
    avg_pred[gene_names] = pd.concat([df[gene_names] for df in runs.values()]).groupby(level=0).mean()
    runs['avg'] = avg_pred
    return runs


def run_vib_self(data, feat_names, gene_names, hyperparam, N_run=3, out_dir=None, out_prefix='',
                  recompute=False, seed_offset=0, label=''):
    """Trains and caches to disk anything missing for this config (leave-one-colony-out CV
    on data -- caller pre-filters to the desired condition(s), e.g.
    data[data['condition']=='B50']), without loading results into memory. Use load_vib_self
    if you need the predictions back. To also predict OTHER conditions from a model trained
    here (e.g. B10/B200 from a B50-trained model), use run_vib_cross_condition separately --
    that's no longer bundled into this function."""
    _run_vib_self_impl(data, feat_names, gene_names, hyperparam, N_run, out_dir, out_prefix,
                        seed_offset, label, load_results=False, allow_training=True, recompute=recompute)


def load_vib_self(data, feat_names, gene_names, hyperparam, N_run=3, out_dir=None, out_prefix='',
                   seed_offset=0, label=''):
    """Same as run_vib_self, but returns {it: pred_df, 'avg': mean_pred_df}. Never trains,
    regardless of cache state -- raises FileNotFoundError if a run isn't cached yet. No
    recompute parameter for the same reason: this function should never be capable of
    triggering a retrain."""
    return _run_vib_self_impl(data, feat_names, gene_names, hyperparam, N_run, out_dir, out_prefix,
                               seed_offset, label, load_results=True, allow_training=False)


def run_vib_cross_condition(data_train, data_test, feat_names, gene_names, hyperparam,
                             N_run=3, out_dir=None, out_prefix='', recompute=False, seed_offset=0,
                             label=''):
    """Train on ALL of data_train (caller pre-filters to the desired training condition,
    e.g. data[data['condition']=='B50']), predict every row of data_test -- both drawn from
    the SAME experiment (same imaging session/calibration). Unlike run_vib_cross_experiment,
    signals are NOT independently re-aligned: data_test is fed directly into VIB's own
    predict(), relying entirely on the scaler VIB itself fit on data_train. This is
    deliberate: data_test's condition (e.g. B10, B200) is expected to have genuinely
    different signal statistics than data_train's B50 for real biological reasons (a
    different BMP dose driving a different signaling response), and independently
    re-standardizing data_test's own signals (as fit_zscore_alignment would) would erase
    exactly that biological difference instead of preserving it for the model to learn
    from. Only appropriate within one experiment, where there's no calibration gap to
    correct -- for genuinely different experiments/imaging sessions, use
    run_vib_cross_experiment instead, which does perform that alignment."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tag = f'[{label}] ' if label else ''
    runs = {}
    for it in range(N_run):
        out_path = f'{out_dir}/{out_prefix}_run{it}.csv' if out_dir else None
        if out_path is not None and not recompute and os.path.exists(out_path):
            print(f'{tag}Run {it}/{N_run} -- already exists, loading: {out_path}')
            runs[it] = pd.read_csv(out_path, index_col=0)
            continue
        print(f'{tag}Run {it}/{N_run}')
        start = time.time()
        torch.manual_seed(seed_offset + it)
        np.random.seed(seed_offset + it)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed_offset + it)
        print(f'{tag}  training on {len(data_train)} cells')
        vib = VIB(data_train[feat_names], data_train[gene_names], hyperparam)
        _ = vib.train(verbose=False)
        print(f'{tag}  predicting {len(data_test)} cells')
        tar_predict = vib.predict(data_test[feat_names])
        pred_df = data_test.copy()
        pred_df.loc[tar_predict.index, tar_predict.columns] = tar_predict
        if out_path is not None:
            pred_df.to_csv(out_path)
        runs[it] = pred_df
        print(f'{tag}Elapsed time: {time.time() - start:.1f} seconds')
    avg_pred = runs[0].copy()
    avg_pred[gene_names] = pd.concat([df[gene_names] for df in runs.values()]).groupby(level=0).mean()
    runs['avg'] = avg_pred
    return runs


def run_vib_cross_experiment(data_train_exp, data_test_exp, feat_names, gene_names, hyperparam,
                              N_run=3, out_dir=None, out_prefix='', recompute=False, seed_offset=0,
                              label=''):
    """Train on ALL of data_train_exp -- the caller is responsible for restricting this to
    whichever training rows are intended (e.g. data[data['condition']=='B50']); this
    function trusts data_train_exp as given rather than filtering internally, so a caller
    that wants to pool training data from multiple sources/conditions can just concatenate
    them beforehand and pass the combined dataframe (see run_vib_joint_experiments).
    Predicts every row of data_test_exp. Input signals are aligned first via
    fit_zscore_alignment (data_test_exp z-scored on its own B50 stats, then rescaled to
    data_train_exp's). Gene targets are NOT scaled -- predictions come out directly in
    data_train_exp's own gene-value scale; align data_test_exp's own measured gene columns
    onto that scale separately (fit_gene_alignment/apply_alignment)."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tag = f'[{label}] ' if label else ''
    signal_align_params = fit_zscore_alignment(data_test_exp, data_train_exp, feat_names)
    data_test_exp_aligned = apply_alignment(data_test_exp, signal_align_params, feat_names)
    runs = {}
    for it in range(N_run):
        out_path = f'{out_dir}/{out_prefix}_run{it}.csv' if out_dir else None
        if out_path is not None and not recompute and os.path.exists(out_path):
            print(f'{tag}Run {it}/{N_run} -- already exists, loading: {out_path}')
            runs[it] = pd.read_csv(out_path, index_col=0)
            continue
        print(f'{tag}Run {it}/{N_run}')
        start = time.time()
        torch.manual_seed(seed_offset + it)
        np.random.seed(seed_offset + it)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed_offset + it)
        print(f'{tag}  training on {len(data_train_exp)} cells')
        vib = VIB(data_train_exp[feat_names], data_train_exp[gene_names], hyperparam)
        _ = vib.train(verbose=False)
        print(f'{tag}  predicting {len(data_test_exp)} cells')
        tar_predict = vib.predict(data_test_exp_aligned[feat_names])
        pred_df = data_test_exp.copy()
        pred_df.loc[tar_predict.index, tar_predict.columns] = tar_predict
        if out_path is not None:
            pred_df.to_csv(out_path)
        runs[it] = pred_df
        print(f'{tag}Elapsed time: {time.time() - start:.1f} seconds')
    avg_pred = runs[0].copy()
    avg_pred[gene_names] = pd.concat([df[gene_names] for df in runs.values()]).groupby(level=0).mean()
    runs['avg'] = avg_pred
    return runs

def run_vib_joint_experiments(exp_results_list, train_configs, test_configs, signal_names,
                               gene_names, reference_exp, hyperparam, N_run=3,
                               out_dir=None, out_prefix='', recompute=False, seed_offset=0,
                               label='', fits_ref=None):
    """Cross-experiment VIB training/prediction across more than two experiments, with
    optional JOINT training pooled across multiple (experiment, condition) pairs at once --
    complements run_vib_cross_experiment (one train experiment -> one test experiment) for
    fig5-style setups (e.g. train on Exp20 B50 alone, or jointly on Exp20 B50 + a real Z2
    perturbation, then predict several other real perturbation conditions).

    exp_results_list: {exp_name: object with .data (raw per-cell df, 'condition'/'Colony'
    columns)}.

    Alignment: every experiment other than reference_exp has its signals and genes aligned
    onto reference_exp's raw-unit scale via fit_zscore_alignment/fit_gene_alignment, fit
    ONCE from that experiment's own B50 control condition against reference_exp's B50 (only
    B50 controls are assumed directly comparable across experiments -- never fit from a
    perturbed condition), then applied to that experiment's FULL dataframe (every
    condition, not just B50). fits_ref (if given) is reference_exp's precomputed
    fit_marker_distributions result, reused for every non-reference experiment aligned here
    instead of being refit each time, and can be passed in again on a repeat call (e.g. an
    ablation sweep over signal_names) to skip recomputing it there too.

    Training/prediction reuses run_vib_cross_experiment directly for the actual full-pool
    step (both functions share the same alignment mechanism and conventions; this one only
    adds multi-experiment pooling and self-CV around it). Before being returned, each
    (exp_name, cond) prediction is converted back to exp_name's OWN native units (the
    inverse of its gene alignment) so it lines up with that experiment's own manually-
    calibrated fate thresholds -- only reference_exp's own predictions need no conversion.

    train_configs / test_configs: lists of (exp_name, condition) pairs, e.g.
    [('20', 'B50'), ('Z2', 'B50M0p5')]. train_configs are pooled together for training
    (jointly, if more than one given); test_configs are pure holdout, predicted only by the
    final full-pool model, never included in any training pool.

    Two prediction passes happen per run, matching run_vib_self's self/cross split:
      (1) Leave-one-colony-out CV *within the training configs only*, fold ranks aligned
          across experiments so joint training can still hold out one colony per
          experiment per fold -- a self-style estimate for the training data itself (this
          is what a train_config with an empty test_configs list reads back, e.g. the
          joint-training variant's own B50M0p5 predictions). Implemented directly here, not
          via run_vib_cross_experiment, since nothing existing does rank-aligned multi-
          experiment colony holdout. Not disk-cached (unlike phase 2) -- always recomputed.
      (2) A model trained on ALL of every train_config's data (no colonies held out)
          predicts every test_config, via run_vib_cross_experiment (which handles its own
          disk caching, respecting recompute) -- the genuine cross-experiment/cross-
          condition holdout result.

    Returns (avg_preds, predictions): avg_preds is {(exp_name, cond): averaged-over-runs
    prediction df, in exp_name's own raw gene units}; predictions is
    {(exp_name, cond): {it: pred_df, 'avg': avg_pred_df}} for inspecting individual runs."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tag = f'[{label}] ' if label else ''

    # --- Align every non-reference experiment's signals + genes onto reference_exp.
    ref_data = exp_results_list[reference_exp].data
    fits_ref = fits_ref or fit_marker_distributions(ref_data[ref_data['condition'] == 'B50'], gene_names)
    aligned_data = {}
    gene_align_to_ref = {}
    for exp_name, exp_res in exp_results_list.items():
        if exp_name == reference_exp:
            aligned_data[exp_name] = exp_res.data
            continue
        fits_self = fit_marker_distributions(exp_res.data[exp_res.data['condition'] == 'B50'], gene_names)
        signal_align = fit_zscore_alignment(exp_res.data, ref_data, signal_names)
        gene_align = fit_gene_alignment(exp_res.data, ref_data, gene_names,
                                         fits_self=fits_self, fits_ref=fits_ref)
        data_aligned = apply_alignment(exp_res.data, signal_align, signal_names)
        data_aligned = apply_alignment(data_aligned, gene_align, gene_names)
        aligned_data[exp_name] = data_aligned
        gene_align_to_ref[exp_name] = gene_align
        print(f'{tag}aligned {exp_name} -> {reference_exp}')

    all_configs = train_configs + test_configs

    def get_cond_data(exp_name, cond):
        d = aligned_data[exp_name]
        return d[d['condition'] == cond]

    def to_native_units(exp_name, pred_df):
        """Convert a prediction (currently in reference_exp's aligned gene units) back to
        exp_name's own raw gene units -- the inverse of fit_gene_alignment's (scale, shift)."""
        if exp_name == reference_exp:
            return pred_df
        inverse_params = {g: (1 / s, -sh / s) for g, (s, sh) in gene_align_to_ref[exp_name].items()}
        return apply_alignment(pred_df, inverse_params, gene_names)

    # --- Rank-aligned colony folds, one list per training config, for phase 1's CV.
    colony_folds = {}
    for exp_name, cond in train_configs:
        colonies = np.sort(get_cond_data(exp_name, cond)['Colony'].unique())
        colony_folds[(exp_name, cond)] = {rank: col for rank, col in enumerate(colonies)}
    n_folds = max(len(cf) for cf in colony_folds.values())

    predictions = {key: {} for key in all_configs}
    test_exp_names = sorted({exp_name for exp_name, _ in test_configs})

    for it in range(N_run):
        print(f'{tag}Run {it}/{N_run}')
        start = time.time()
        torch.manual_seed(seed_offset + it)
        np.random.seed(seed_offset + it)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed_offset + it)

        pred_dfs = {key: get_cond_data(*key).copy() for key in all_configs}

        # --- Phase 1: leave-one-colony-out CV within train_configs only, cached to disk
        # separately from phase 2 (which run_vib_cross_experiment already caches on its own).
        selfcv_out_path = f'{out_dir}/{out_prefix}_selfcv_run{it}.csv' if out_dir else None
        if selfcv_out_path is not None and not recompute and os.path.exists(selfcv_out_path):
            print(f'{tag}  self-CV -- already exists, loading: {selfcv_out_path}')
            cached = pd.read_csv(selfcv_out_path, index_col=0, dtype={'_source_exp': str, '_source_cond': str})
            for exp_name, cond in train_configs:
                subset = cached[(cached['_source_exp'] == exp_name) & (cached['_source_cond'] == cond)]
                pred_dfs[(exp_name, cond)] = subset.drop(columns=['_source_exp', '_source_cond'])
        else:
            for test_rank in range(n_folds):
                print(f'{tag}  fold {test_rank}/{n_folds}')
                train_feat, train_tar, test_sets = [], [], {}
                for exp_name, cond in train_configs:
                    data_cond = get_cond_data(exp_name, cond)
                    colony_fold = colony_folds[(exp_name, cond)]
                    if test_rank in colony_fold:
                        test_col = colony_fold[test_rank]
                        test_mask = data_cond['Colony'] == test_col
                        train_mask = ~test_mask
                    else:
                        test_mask = pd.Series(False, index=data_cond.index)
                        train_mask = pd.Series(True, index=data_cond.index)
                    train_feat.append(data_cond[train_mask][signal_names])
                    train_tar.append(data_cond[train_mask][gene_names])
                    if test_mask.any():
                        test_sets[(exp_name, cond)] = data_cond[test_mask]

                feat_train = pd.concat(train_feat)
                tar_train = pd.concat(train_tar)
                vib = VIB(feat_train, tar_train, hyperparam)
                _ = vib.train(verbose=False)

                for key, data_test in test_sets.items():
                    tar_predict = vib.predict(data_test[signal_names])
                    pred_dfs[key].loc[tar_predict.index, tar_predict.columns] = tar_predict

            if selfcv_out_path is not None:
                tagged = []
                for exp_name, cond in train_configs:
                    df = pred_dfs[(exp_name, cond)].copy()
                    df['_source_exp'] = exp_name
                    df['_source_cond'] = cond
                    tagged.append(df)
                pd.concat(tagged).to_csv(selfcv_out_path)

        # --- Phase 2: pool ALL train_configs into one combined training dataframe, then
        # reuse run_vib_cross_experiment for the actual full-pool training/prediction
        # (once per distinct test experiment). Pass the FULL aligned dataframe (not just
        # the test conditions) as data_test, since run_vib_cross_experiment's own internal
        # signal re-alignment needs B50 rows present to fit from -- it comes out ~identity
        # here since the data's already aligned upstream, but needs B50 rows to compute
        # that from at all.
        combined_train_df = pd.concat([get_cond_data(*key) for key in train_configs])

        for exp_name in test_exp_names:
            test_conds = [c for e, c in test_configs if e == exp_name]
            data_test_full = aligned_data[exp_name]
            cross_out_prefix = f'{out_prefix}_{exp_name}_run{it}' if out_dir else ''
            cross_runs = run_vib_cross_experiment(
                combined_train_df, data_test_full, signal_names, gene_names, hyperparam,
                N_run=1, out_dir=out_dir, out_prefix=cross_out_prefix, recompute=recompute,
                seed_offset=seed_offset + it, label=f'{label} (joint, run {it})')
            pred_for_exp_full = cross_runs[0]
            for cond in test_conds:
                pred_dfs[(exp_name, cond)] = pred_for_exp_full[pred_for_exp_full['condition'] == cond].copy()

        # Convert every prediction back to its own experiment's native units.
        for exp_name, cond in all_configs:
            pred_dfs[(exp_name, cond)] = to_native_units(exp_name, pred_dfs[(exp_name, cond)])

        for key in all_configs:
            predictions[key][it] = pred_dfs[key]
        print(f'{tag}Elapsed time: {time.time() - start:.1f} seconds')

    avg_preds = {}
    for key in all_configs:
        avg = predictions[key][0].copy()
        avg[gene_names] = pd.concat(
            [predictions[key][it][gene_names] for it in range(N_run)]
        ).groupby(level=0).mean()
        predictions[key]['avg'] = avg
        avg_preds[key] = avg

    return avg_preds, predictions


def compute_latent_position_entropy(vib, data_z, signal_names, gene_names,
                                     thresh, n_samples=1000, device='cpu'):
    
    X_tensor = torch.FloatTensor(data_z[signal_names].values).to(device)
    
    vib.model.eval()
    with torch.no_grad():
        mu, logvar = vib.model.encode(X_tensor)
    
    all_fates = []
    for _ in range(n_samples):
        with torch.no_grad():
            std = torch.exp(0.5 * logvar)
            z   = mu + torch.randn_like(std) * std
            Y_z = vib.model.decode(z).cpu().numpy()
        
        Y = vib.scaler_Y_run.inverse_transform(Y_z)
        Y = pd.DataFrame(Y, columns=gene_names, index=data_z.index)
        
        fates, _ = return_fates(Y, thresh=thresh)
        all_fates.append(fates)
    
    fate_matrix = pd.DataFrame(
        {i: f for i, f in enumerate(all_fates)},
        index=data_z.index
    )
    per_cell = fate_matrix.apply(
        lambda row: stats.entropy(row.value_counts(), base=2), axis=1
    )
    
    return per_cell, mu.cpu().numpy()


def plot_colony_pie_triplet(colony_list, exp_result_list, labels,
                             ms=7, figsize=(5, 5), start_angle=90,
                             boundary_color='white', boundary_lw=2, dpi=150):
    """Arrange three colony scatter plots as pie sectors."""
    import io
    from matplotlib.patches import Wedge

    sector_starts = [start_angle + i * 120 for i in range(3)]
    fig, ax       = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_aspect('equal')
    ax.axis('off')

    for colony, exp_result, s_start in zip(colony_list, exp_result_list, sector_starts):

        # Render colony to buffer
        fig_tmp, ax_tmp = plt.subplots(figsize=(5, 5), constrained_layout=True)
        colony.scatter_fates(ms=ms, ax=ax_tmp, legend=False, thresh=exp_result.thresh)
        buf = io.BytesIO()
        fig_tmp.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=True)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig_tmp)

        # Crop whitespace
        mask         = img[:, :, 3] > 0.01 if img.shape[2] == 4 else ~np.all(img > 0.99, axis=2)
        rows, cols   = np.any(mask, axis=1), np.any(mask, axis=0)
        r0, r1       = np.where(rows)[0][[0, -1]]
        c0, c1       = np.where(cols)[0][[0, -1]]
        pad          = 5
        img_cropped  = img[max(0, r0-pad):r1+pad, max(0, c0-pad):c1+pad]

        # Place image clipped to sector wedge
        wedge = Wedge((0, 0), r=1.0, theta1=s_start, theta2=s_start+120,
                      transform=ax.transData)
        im    = ax.imshow(img_cropped, extent=[-1, 1, -1, 1],
                          aspect='equal', zorder=2, origin='upper')
        im.set_clip_path(wedge)

    # Sector dividers
    for angle in sector_starts:
        rad = np.radians(angle)
        ax.plot([0, np.cos(rad)], [0, np.sin(rad)],
                color=boundary_color, linewidth=boundary_lw, zorder=5)

    return fig, ax

def plot_latent_entropy_comparison(latent_entropy_results, condition_data_dict,
                                    figsize=(8, 5), use_ci=True, n_bootstrap=1000):
    """Bar plot of latent position entropy with colony-level bootstrap."""

    def get_ci(label):
        per_cell  = latent_entropy_results[label]['per_cell']
        data_cond = condition_data_dict.get(label)
        
        # Mean entropy per colony
        colony_means = np.array([
            per_cell.reindex(data_cond[data_cond['Colony'] == col].index).dropna().mean()
            for col in data_cond['Colony'].unique()
        ])
        colony_means = colony_means[~np.isnan(colony_means)]
        
        bootstraps = [np.mean(np.random.choice(colony_means, len(colony_means), replace=True))
                      for _ in range(n_bootstrap)]
        
        return {'mean':     colony_means.mean(),
                'ci_lower': np.percentile(bootstraps, 2.5),
                'ci_upper': np.percentile(bootstraps, 97.5)}

    def get_yerr(r):
        return [[r['mean'] - r['ci_lower']], [r['ci_upper'] - r['mean']]] if use_ci \
               else [[r['mean'] - r['ci_lower']]]

    def bar(ax, x, label, color, **kwargs):
        r = ci[label]
        ax.bar(x, r['mean'], width=bar_width, color=color, alpha=0.8,
               edgecolor='black', linewidth=1.5,
               yerr=get_yerr(r), capsize=5,
               error_kw={'linewidth': 1.5}, **kwargs)

    ci        = {label: get_ci(label) for label in latent_entropy_results}
    bar_width = 0.25
    x_pos     = np.arange(4)
    colors    = {'meas': 'coral', 'naive': '#aa5435', 'knn': '#6b2e1a'}

    fig, ax = plt.subplots(figsize=figsize)

    # B50 ctrl
    bar(ax, x_pos[0], 'B50', colors['meas'])

    # Perturbation groups
    for i, (key, x) in enumerate(zip(['Z2 MEKi', 'Z2 IWP2', 'X7 TRULI'], x_pos[1:])):
        bar(ax, x - bar_width, key,              colors['meas'])
        bar(ax, x,             f'{key} (naive)', colors['naive'])
        bar(ax, x + bar_width, f'{key} (knn)',   colors['knn'])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['B50', 'MEKi', 'WNTSeci', 'LATSi'], ha='center', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('prediction entropy', fontsize=16)
    ax.legend(handles=[
        mpatches.Patch(facecolor=c, edgecolor='black', linewidth=1.5, label=l)
        for c, l in zip(colors.values(), ['measured', 'naive', 'projected'])
    ], fontsize=15)

    plt.tight_layout()
    return fig, ax
#------------------------------------------------------------------------------------------------------------
# ANALYSIS: INFORMATION
#------------------------------------------------------------------------------------------------------------

import matplotlib.patheffects as pe
from matplotlib.patches import Arc

def getPreds(data, cond, dataDir, signal_chains, fatemarker, gene_names, hyperparam, N_run, save=True):

    mean_preds = {}
    mean_train_preds = {} # predictions on training data to check for overfitting

    data_cond = data[data['condition']==cond]
    
    for feat_names in signal_chains:

        feat_name_str = '_'.join(sorted(feat_names, key=str.lower))
        fname = dataDir + '/251115_VIB_' + str(len(feat_names)) + 'D_'+cond+'_' + feat_name_str + '_avg.csv'
        fname_train = dataDir + '/251115_VIB_' + str(len(feat_names)) + 'D_'+cond+'_' + feat_name_str + '_train_avg.csv'
        
        print(fname)
        if os.path.exists(fname) and os.path.exists(fname_train):
            #print('reading:'+fname)
            mean_preds[tuple(feat_names)] = pd.read_csv(fname, index_col=0)
            mean_train_preds[tuple(feat_names)] = pd.read_csv(fname_train, index_col=0)
        else:
            print('running: '+str(feat_names))
            mean_preds[tuple(feat_names)], mean_train_preds[tuple(feat_names)] = sig2fate(data_cond, list(feat_names),  gene_names, N_run, hyperparam)
            if save:
                mean_preds[tuple(feat_names)].to_csv(fname)
                mean_train_preds[tuple(feat_names)].to_csv(fname_train)

    return mean_preds, mean_train_preds

# i can probably merge this with getPreds - do later when there is time

def getPreds2(data, dataDir, signal_combinations, gene_names, hyperparam, N_run=3, save=True):

    pred_subs = {}
    conditions = np.unique(data['condition'])
    data_B50 = data[data['condition']=='B50']
    
    for signals in signal_combinations:

        sig_str = '_'.join(sorted(signals))

        fname = dataDir + '/Fig3_VIB/Fig3_VIB_' + sig_str + '_B50predonly.csv'
        
        if os.path.exists(fname):
            print(f'loading: {fname}')
            pred_mean_df = pd.read_csv(fname, index_col=0)
            pred_mean_df['condition'] = data['condition']
            
        else:
            print(f'calculating: {fname}')
            pred_df = data.copy()   
            mean_pred,_ = sig2fate(data_B50, signals, gene_names, N_run, hyperparam)
            pred_df.loc[mean_pred.index, mean_pred.columns] = mean_pred
            
            # then predict other conditions based on all B50 colonies
            start = time.time()
            preds = {}
            for it in range(N_run):

                # set seeds for reproducibility
                torch.manual_seed(it)
                np.random.seed(it)
                if torch.backends.mps.is_available():
                    torch.mps.manual_seed(it)
                    
                preds[it] = pred_df.copy()
                
                data_train = data_B50
                feat_train = data_train[signals]
                tar_train = data_train[gene_names] 
                vib = VIB(feat_train, tar_train, hyperparam)
                _ = vib.train(verbose=False) 
                
                for cond in [c for c in conditions if c != 'B50']:
            
                    data_cond = data[data['condition']==cond]
                    feat_test = data_cond[signals]
                    tar_predict = vib.predict(feat_test)
                    preds[it].loc[tar_predict.index, tar_predict.columns] = tar_predict

            pred_mean_df = pd.concat(list(preds.values())).groupby(level=0).mean(numeric_only=True)
            
            # add back non-numeric columns from original data
            non_numeric_cols = data.select_dtypes(exclude='number').columns
            for col in non_numeric_cols:
                if col in data.columns:
                    pred_mean_df[col] = data[col]

            if save:
                pred_mean_df.to_csv(fname)
            end = time.time()
            print(f"Elapsed time: {end - start} seconds")

        pred_subs[sig_str] = {'avg':pred_mean_df}

    return pred_subs
    

def col_meanstd(MI_dec):

    genelist = list(MI_dec.keys())
    signals = MI_dec[genelist[0]].index
    colonies = MI_dec[genelist[0]].columns

    # Initialize output DataFrames with signals as rows, genes as columns
    MI_dec_mean = pd.DataFrame(np.zeros((len(signals), len(genelist))), index=signals, columns=genelist)
    MI_dec_std = pd.DataFrame(np.zeros((len(signals), len(genelist))), index=signals, columns=genelist)

    # Calculate mean and std across colonies for each gene
    for gene in genelist:
        MI_dec_mean[gene] = MI_dec[gene].mean(axis=1)  # Mean across colonies
        MI_dec_std[gene] = MI_dec[gene].std(axis=1, ddof=1)  # Std across colonies

    return MI_dec_mean, MI_dec_std
    
def plotCumulativeMI(data, cond, dataDir, markergenes, signals, signames_simple, N_run, hyperparam, plotparam=None, uniqueMI=None, checkoverfit=False, title=True):

    # signals must be subset of global signal_names
    # also get the simplified names corresponding to this subset
    # index = [signal_names.index(item) for item in signals]
    # if not signames_simple: signames_simple = [signal_names_simplified[i] for i in index]

    if not plotparam: plotparam = {'fs':15, 'fs2':19, 'fs3':15, 'xlabel': 'cumulative MI (bits)','labelpad':-10, 'x':0.46,'round':2, 'marg':0.01}
    
    for fatemarker in markergenes:

        print('=========='+fatemarker+'====================')
        
        data_cond = data[data['condition']==cond]
        maxMI_dict = {0: {fatemarker:((), 0)}}
        remaining_sigs = signals
        MI_dec = {}
        rd = 1
    
        while len(remaining_sigs) > 0:
        
            print('rd: '+str(rd))
            signal_chains = [maxMI_dict[rd-1][fatemarker][0] + (s,) for s in remaining_sigs]

            mean_preds, mean_train_preds = getPreds(data, cond, dataDir, signal_chains, fatemarker, markergenes, hyperparam, N_run)
            if checkoverfit:
                maxMI_dict[rd], MI_dec[rd] = getmaxDecoderMI([fatemarker], signals, mean_train_preds, data_cond, debug=False)
            else:
                maxMI_dict[rd], MI_dec[rd] = getmaxDecoderMI([fatemarker], signals, mean_preds, data_cond, debug=False)
                
            maxMIsigs = maxMI_dict[rd][fatemarker][0]
            remaining_sigs = [s for s in signals if s not in list(maxMIsigs)]
            rd += 1
    
        # MAKE THE PLOT
        #------------------------------------------------------------------------------------------------

        N_signals = len(signals)
        signals_ordered = list(maxMI_dict[N_signals][fatemarker][0])
        perm = [signals.index(item) for item in signals_ordered]
        labels = [signames_simple[i] for i in perm]
        
        fig,ax = plt.subplots(1,1, figsize=(4,5))
        
        if title:
            plt.title(fatemarker,fontsize=26, pad=10, fontweight='bold',color = [0, 0.8, 0.8], path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        maxMItotal = 0
        xlim = 0
        #color = 'cornflowerblue'
        color = [0.6,0.6,0.6]
        uniquecolor = [0.8,0.8,0.8]
        kcutoff = 10

        # find the signaling combination for which the MI is maximal
        meanMI = [float(maxMI_dict[i][fatemarker][1].mean()) for i in range(1,len(maxMI_dict))]
        maxMIidx = meanMI.index(max(meanMI)) + 1 # offset because list above starts at 1 
        maxMIall = maxMI_dict[maxMIidx][fatemarker][1]

        # MI for all signals combined
        #maxMIall = maxMI_dict[N_signals][fatemarker][1]
    
        for k, s in enumerate(signals_ordered):
        
            MI_mean, MI_std = col_meanstd(MI_dec[1])
            w = MI_mean.loc[s, fatemarker].iloc[0]
            if uniqueMI:
                u = uniqueMI[fatemarker][s]
            else:
                u = 0
            std = MI_std.loc[s, fatemarker].iloc[0]/np.sqrt(5)
            y = k+1
    
            # test if total MI is still significantly different from total
            maxMIthisrd = maxMI_dict[k+1][fatemarker][1]
            t_stat, p_value = sp.stats.ttest_rel(maxMIthisrd, maxMIall, alternative='two-sided')
            print('p value ' + str(p_value))
            if p_value > 0.05:
                kcutoff = min(kcutoff, k+1)
                
            maxMIthisrdavg = np.mean(maxMI_dict[k+1][fatemarker][1])
            maxMIthisrdstd = np.std(maxMI_dict[k+1][fatemarker][1])
            sem = maxMIthisrdstd/np.sqrt(5)
            maxMItotal = max(maxMIthisrdavg, maxMItotal)
            
            if k>0:
                left = maxMIthisrdavg - w
                ax.barh(y=y,width=w-u,left=left,color=color) 
                ax.barh(y=y,width=u,left=left+w-u,color=uniquecolor) 
                ax.errorbar(left+w,y,xerr=sem,capsize=3,color='k')
                ax.text(left - plotparam['marg'], y, labels[k], va='center', ha='right', fontsize=plotparam['fs3'])
                #ax.text(left + marg, y, labels[k], va='center', ha='left', fontsize=fs, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            else:
                ax.barh(y=y, width=w-u, color=color)
                ax.barh(y=y,width=u,left=w-u,color=uniquecolor) 
                ax.errorbar(w,y,xerr=sem,capsize=3,color='k')
                ax.text(float(plotparam['marg']), y, labels[k], va='center', ha='left', fontsize=plotparam['fs3'], path_effects=[pe.withStroke(linewidth=5, foreground="white")])
        
        ax.set_yticks([])
    
        totalmean = np.mean(maxMIall)
        totalsem = np.std(maxMIall)/np.sqrt(5)
        xlim = totalmean*1.1 
        
        ax.axhline(y=kcutoff + 0.5, xmin=0, xmax=1, color='k', linestyle=':', linewidth=2, zorder=10)
        ax.axvspan(xmin=totalmean-totalsem, xmax=totalmean+totalsem, color='k', alpha=0.1, zorder=0)
        ax.axvline(x=np.mean(maxMIall), ymin=0, ymax=k+1, color='k', linestyle='-', linewidth=1, zorder=0, alpha=0.5)
        
        #ax.set_yticks(range(1,1+N_signals),labels=labels)
        plt.xlim([0,xlim])
        plt.ylim([1/4, N_signals+3/4])
        plt.xticks([0,np.round(np.mean(maxMIall),plotparam['round'])],labels=['0',str(np.round(np.mean(maxMIall),plotparam['round']))], fontsize=plotparam['fs'])
        ax.set_xlabel(plotparam['xlabel'], fontsize=plotparam['fs2'], labelpad=plotparam['labelpad'], x=plotparam['x'])
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        ax.set_box_aspect(1) #ax.set_aspect('equal', adjustable='box')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15) 
        #plt.tight_layout()
        
        #plt.tight_layout(pad=0)

        # Save core fate markers in Fig3, and the rest in FigS4
        
        if fatemarker in ['ISL1', 'TFAP2C', 'SOX17', 'TBXT', 'TBX6', 'NANOG', 'SOX2']:
            print(fatemarker + " core")
            file_prefix = 'Fig3'

        elif fatemarker in ['L', 'g1', 'g2']:
            file_prefix = 'FigS6'
        else:
            file_prefix = 'FigS4'
            
        if checkoverfit:
            fname = file_prefix + '/cumulativeMI_' + fatemarker + '_' + cond + '_' + str(len(signals)) + 'D' + '_train.png'
        else:
            fname = file_prefix  + '/cumulativeMI_' + fatemarker + '_' + cond + '_' + str(len(signals)) + 'D' + '.png'
        plt.savefig(fname)
        
def getmaxDecoderMI(genelist, signal_names, pred, data, debug=False):
    # for genes in genelist, return the signal or signaling combination (keys in pred) that provides the highest decoder-based MI
    # 
    # list: list of marker gene names
    # pred: dictionary: keys are tuples of signals, values are predictions of fate markers based on the signals 
    
    colonies = [int(n) for n in np.unique(data['Colony'])]
    MI_dec = {} 
    MI_dec_mean = pd.DataFrame(np.zeros((len(pred.keys()),len(genelist))), index=pred.keys(), columns=genelist)
    maxMI_signal_for_gene = {}

    for f in genelist:

        # calculate MI for each combination of signals in pred
        #-----------------------------------------------------
        MI_dec[f] = pd.DataFrame(np.zeros((len(pred.keys()),len(colonies))), index=pred.keys(), columns=colonies)

        for ci in colonies:
            idx = data['Colony']==ci

            for s in pred.keys():
                MI_dec[f].at[s,ci] = sklearn.feature_selection.mutual_info_regression(pred[s].loc[idx,f].to_numpy().reshape(-1, 1), data.loc[idx,f].to_numpy())/np.log(2)

        MI_dec_mean[f] = MI_dec[f].mean(axis=1)  # Mean across colonies (columns)
        if debug: print(MI_dec_mean)

        # test for significant differences in MI
        #------------------------------------------------

        # significance threshold 
        alpha = 0.1
        
        best_idx = MI_dec_mean[f].idxmax()
        best_mi = MI_dec[f].loc[best_idx]
        print(f'best mean: {best_idx}, {MI_dec[f].loc[best_idx].mean():.2f}({MI_dec[f].loc[best_idx].std():.2f})')
        
        p_values = []
        rejected = []
        for s in MI_dec_mean.index:
            if s != best_idx:
                current_mi = MI_dec[f].loc[s]
                
                # Check if data is valid (not identical, not NaN)
                if not current_mi.equals(best_mi) and len(current_mi) > 1:
                    # paired ttest: MIs for each colony for different signals
                    _, p = sp.stats.ttest_rel(best_mi, current_mi) 
                    p_values.append((s, p))
                    if debug: print(f'{s}, {current_mi.mean():.2f}({current_mi.std():.2f}), p:{p:.3f}')

        if p_values:  # Only if there are comparisons to make
            # Apply multiple testing correction
            indices, pvals = zip(*p_values)
            rejected, corrected_p, _, _ = statsmodels.stats.multitest.multipletests(pvals, alpha=alpha, method='fdr_bh') # , method='fdr_bh'
            if debug: print(corrected_p)

        # Check if best is significantly better than all others
        if all(rejected):  # All comparisons significant
            selected = best_idx
            maxMI_signal_for_gene[f] = (best_idx, MI_dec[f].loc[best_idx])
            print(f'selected best: {best_idx}, {MI_dec[f].loc[best_idx].mean():.2f}({MI_dec[f].loc[best_idx].std():.2f})')
        else:
            # Identify which variables are NOT significantly different from best
            equivalent_vars = [best_idx] + [indices[i] for i, rej in enumerate(rejected) if not rej]

            def get_sort_key(signal_tuple):
                # Make sure it's a tuple
                if not isinstance(signal_tuple, tuple):
                    signal_tuple = (signal_tuple,)
                
                # For each signal in the tuple, get its position in default_order
                # Signals not in default_order go to the end
                priorities = []
                for signal in signal_tuple:
                    priorities.append(signal_names.index(signal))
                return priorities
            
            # Sort and take the first one
            equivalent_vars_sorted = sorted(equivalent_vars, key=get_sort_key)
            print('selected best: '+str(equivalent_vars_sorted))
            
            #maxMI_signal_for_gene[f] = (equivalent_vars_sorted[0], MI_dec_mean.loc[equivalent_vars_sorted[0]].iloc[0])
            maxMI_signal_for_gene[f] = (equivalent_vars_sorted[0], MI_dec[f].loc[equivalent_vars_sorted[0]])

    # maxMI signal for gene is array with values for each colony
    return maxMI_signal_for_gene, MI_dec


def plotRedundantMI(dataDir, data, cond, markergenes, signals, N_run, hyperparam):

    maxMI = {}
    sumMI = {}
    
    for fatemarker in markergenes:

        print('=========='+fatemarker+'====================')
        
        data_cond = data[data['condition']==cond]
        maxMI_dict = {0: {fatemarker:((), 0)}}
        remaining_sigs = signals
        MI_dec = {}
        rd = 1
    
        while len(remaining_sigs) > 0:
        
            print('rd: '+str(rd))
            signal_chains = [maxMI_dict[rd-1][fatemarker][0] + (s,) for s in remaining_sigs]

            mean_preds, mean_train_preds = getPreds(data, cond, dataDir, signal_chains, fatemarker, markergenes, hyperparam, N_run)
            maxMI_dict[rd], MI_dec[rd] = getmaxDecoderMI([fatemarker], signals, mean_preds, data_cond, debug=False)
            maxMIsigs = maxMI_dict[rd][fatemarker][0]
            remaining_sigs = [s for s in signals if s not in list(maxMIsigs)]
            rd += 1

        maxMI[fatemarker] = max([MI_dec[i][fatemarker].mean(axis=1).max() for i in range(1,len(MI_dec))])
        sumMI[fatemarker] = sum(MI_dec[1][fatemarker].mean(axis=1))

    # MAKE THE PLOT
    #------------------------------------------------------------------------------------------------
    
    labels = list(sumMI.keys())
    fs = 18
    ms = 10
    
    fig, ax = plt.subplots(figsize=(6, 0.3*len(labels) + 0.8))
    xlim = 3
    
    for ct, fatemarker in enumerate(labels):
    
        ax.plot([sumMI[fatemarker], maxMI[fatemarker]], [ct, ct], color='k',zorder=1)
        ax.scatter(sumMI[fatemarker],ct,color='blue',zorder=2)
        ax.scatter(maxMI[fatemarker],ct,color='red',zorder=2)
        print(fatemarker + ': ' + str(sumMI[fatemarker]) + ', ' + str(maxMI[fatemarker]))
    
    labels_tick = labels.copy()
    ax.set_xticks([0,1,2,3])
    ax.set_yticks(range(len(labels)), labels_tick)  # Set y-ticks to gene names
    ax.tick_params(axis='both', labelsize=fs)
    
    ax.set_ylim([-0.5, len(labels) -0.5])
    ax.set_xlim([0, xlim])
    ax.set_xlabel('MI (bits)', fontsize=fs)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    fname = 'FigS4/MI_redundancy_' + str(len(signals)) + 'D.png'
    plt.savefig(fname)
    
    
    fig, ax = plt.subplots(figsize=(6, 0.3*len(labels) + 0.8))
    
    meanratio = np.mean([maxMI[fatemarker]/sumMI[fatemarker] for fatemarker in labels])
    print('mean ratio: ' + str(meanratio))
    ax.axvline(x=meanratio, ymin=-0.5, ymax=len(labels) -0.5, color='k', linestyle='--', linewidth=1)
    
    for ct, fatemarker in enumerate(labels):
        ax.scatter(maxMI[fatemarker]/sumMI[fatemarker],ct,color='k',zorder=2)
        
    labels_tick = labels.copy()
    ax.set_xticks([0,1,2,3])
    ax.set_yticks(range(len(labels)), labels_tick)  # Set y-ticks to gene names
    ax.tick_params(axis='both', labelsize=fs)
    
    ax.set_ylim([-0.5, len(labels) -0.5])
    ax.set_xlim([0, 1])
    ax.set_xlabel('MI (bits)', fontsize=fs)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    fname = 'FigS4/MI_redundancyratio_' + str(len(signals)) + 'D.png'
    plt.savefig(fname)
    #plt.legend()

    
def plotMIgraph(dataDir, data, cond, signames, markernames, signames_clean=None, markernames_clean=None, rotate=True, ax=None, fs=12, file_suffix='.png', sig2sig=True):

    if not signames_clean: signames_clean = signames
    if not markernames_clean: markernames_clean = markernames
    
    data_cond = data[data['condition']==cond]
    
    N_signals = len(signames)
    N_genes = len(markernames)
    
    #========================================
    # calculate MI
    #========================================
    
    # mutual information between signals
    MIsigs = np.zeros((N_signals, N_signals))
    for si in range(N_signals):
        
        MIsigs[si,si] = np.nan
        
        for sj in range(si+1, N_signals):
            MIsigs[si,sj] = fns_plot.calc_MI_sklearn(data_cond[signames[si]].to_numpy(), data_cond[signames[sj]].to_numpy())
            MIsigs[sj,si] = MIsigs[si,sj]
    MIsigs = pd.DataFrame(MIsigs, index=signames, columns=signames)
    
    # mutual information between signals and fate markers
    MI = np.zeros((N_signals, N_genes))
    for si in range(N_signals):
        for fj in range(N_genes):
            MI[si,fj] = fns_plot.calc_MI_sklearn(data_cond[signames[si]].to_numpy(), data_cond[markernames[fj]].to_numpy())
    MI = pd.DataFrame(MI, index=signames, columns=markernames)
    
    #========================================
    # PLOT GRAPH
    #========================================
        
    labels_top = signames.copy()
    labels_top_clean = signames_clean.copy()
    n = len(labels_top)
    
    labels_bottom = markernames.copy()
    labels_bottom_clean = markernames_clean.copy()
    m = len(labels_bottom) 
    
    if rotate:
        labels_top.reverse()
        labels_top_clean.reverse()
        labels_bottom.reverse()
        labels_bottom_clean.reverse()
        offset = 0.2
    else:
        offset = 0.3
    
    conn = MI.loc[labels_top, labels_bottom]
    sigs = data_cond[labels_top]
    corr = sigs.corr().to_numpy()
    allcorr = data_cond[labels_top + labels_bottom].corr()
    
    # ---- Plotting ----
    if not ax:
        fig, ax = plt.subplots(figsize=(4,3))
        
    ax.set_xlim(-1, max(n, m))
    ax.set_ylim(-4, 2)
    ax.axis('off')
    
    x_pos_top = np.linspace(0, max(n, m)-1, n)
    x_pos_bot = np.linspace(0, max(n, m)-1, m)
    y_top = 0
    y_bot = -3
    
    # Draw top-layer nodes
    for i, label in enumerate(labels_top_clean):
        if rotate:
            if sig2sig:
                ax.text(x_pos_top[i], y_top-offset, label, ha='center', va='top', fontsize=fs, color='k',path_effects=[pe.withStroke(linewidth=3, foreground="white")], rotation=-90)
            else:
                offset = 0.15
                ax.text(x_pos_top[i], y_top+offset, label, ha='center', va='bottom', fontsize=fs, color='k',path_effects=[pe.withStroke(linewidth=3, foreground="white")], rotation=-90, fontfamily='Arial Narrow')
        else:
            ax.text(x_pos_top[i], y_top-offset, label, ha='center', va='center', fontsize=fs, color='k',path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.plot(x_pos_top[i], y_top, 'o', color='k', markersize=6)
    
    # Draw bottom-layer nodes
    for j, label in enumerate(labels_bottom_clean):
        if rotate:
            if sig2sig:
                ax.text(x_pos_bot[j]  - 0.05, y_bot-offset, label, ha='center', va='top', fontsize=fs, color='k', rotation=-90)
            else:
                ax.text(x_pos_bot[j]  - 0.05, y_bot-offset, label, ha='center', va='top', fontsize=fs, color='k', rotation=-90, fontfamily='Arial Narrow')
        else:
            ax.text(x_pos_bot[j], y_bot-offset, label, ha='center', va='center', fontsize=fs, color='k')
        ax.plot(x_pos_bot[j], y_bot, 'o', color='k', markersize=6)
    
    sc = 8
    
    MIcutoff = 0.05
    if sig2sig:
        # Draw arches signal MI (top layer)
        for i, s in enumerate(labels_top):
            for j, t in enumerate(labels_top):
                strength = MIsigs.loc[s, t]#abs(corr[i, j])
                if strength < MIcutoff:
                    continue
                mid = (x_pos_top[i] + x_pos_top[j]) / 2
                width = abs(x_pos_top[j] - x_pos_top[i])
                height = width / 2
                color = 'darkturquoise' if corr[i, j] > 0 else 'firebrick' #'darkmagenta'
                linewidth = sc * strength
                arc = Arc((mid, y_top), width=width, height=height, angle=0, theta1=0, theta2=180, color=color, linewidth=linewidth, alpha=0.6)
                ax.add_patch(arc)
    
    # Draw lines between top and bottom layers, with thickness from conn
    for i,s in enumerate(labels_top):
        for j,f in enumerate(labels_bottom):
            strength = MI.loc[s, f]
            if strength < MIcutoff:  # skip very weak
                continue
            color = 'darkturquoise' if allcorr.loc[s, f] > 0 else 'firebrick' #'darkmagenta'
            linewidth = sc * strength
            ax.plot([x_pos_top[i], x_pos_bot[j]], [y_top, y_bot],color=color, lw=linewidth, alpha=0.6, zorder=1)
    
    plt.tight_layout()
    prefix = '_sig2fate_'
    if sig2sig:
        prefix = '_sig2sig_and' + prefix
    if rotate:
        prefix = prefix + 'rotated_'
    if signames == ['s1','s2']:
        plt.savefig(dataDir+ 'MI' + prefix + cond + file_suffix, bbox_inches='tight')
    else:
        plt.savefig(dataDir+ 'MI' + prefix + cond + file_suffix, bbox_inches='tight')
    
    return MI, MIsigs

#------------------------------------------------------------------------------------------------------------
# ANALYSIS: general
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


    # PGCLC = TFAP2C & SOX17 
    # #AMLC = (ISL1 | TFAP2C) & ~PGCLC
    # #endo = SOX17 & ~PGCLC
    # meso = TBXT & TBX6 & ~PGCLC #& ~endo
    # AMLC = ISL1 & ~PGCLC & ~meso # & ~endo  
    # PSLC = TBXT & ~PGCLC & ~meso & ~AMLC  #& ~endo # (data['TBXT'] > 100)
    # pluri = NANOG & SOX2 & ~PGCLC & ~meso & ~AMLC & ~PSLC  # & ~endo 
    # ecto = ~NANOG & SOX2 & ~PGCLC & ~meso & ~AMLC &~pluri &~PSLC #  & ~endo 
    # other = ~(ecto | pluri | PSLC | AMLC | meso | PGCLC) #  | endo

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

def compute_category_performance_perfold(true_labels, pred_labels, colonies, category_list, aggregate_name='all'):
    records = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
        for colID in np.unique(colonies):
            colidx = (colonies == colID)

            bal_acc = balanced_accuracy_score(true_labels[colidx], pred_labels[colidx])
            f1_val = f1_score(true_labels[colidx], pred_labels[colidx], labels=category_list, average='macro', zero_division=0)
            f1_rand = 1 / len(category_list)
            records.append({'fold': colID, 'category': aggregate_name, 'balanced_accuracy': bal_acc, 'f1': f1_val, 'f1_rand': f1_rand})

            for cat in category_list:
                is_cat = true_labels[colidx] == cat
                is_cat_pred = pred_labels[colidx] == cat
                if is_cat.sum() == 0:
                    continue
                bal_acc = balanced_accuracy_score(is_cat, is_cat_pred)
                f1_val = f1_score(is_cat, is_cat_pred, average='binary', pos_label=True, zero_division=0)
                f1_rand = is_cat.mean()
                records.append({'fold': colID, 'category': cat, 'balanced_accuracy': bal_acc, 'f1': f1_val, 'f1_rand': f1_rand})
    return records


def compute_mse_perfold(true_values, pred_values, colonies, columns, aggregate_name='marker_avg'):
    """Per-fold (colony), per-column MSE between true_values and pred_values (DataFrames
    sharing `columns`), plus an aggregate row averaging MSE across columns. Mirrors
    compute_category_performance_perfold's tidy long-format/per-colony structure."""
    records = []
    for colID in np.unique(colonies):
        colidx = (colonies == colID)

        col_mses = []
        for col in columns:
            mse = np.mean((true_values[col].to_numpy()[colidx] - pred_values[col].to_numpy()[colidx])**2)
            col_mses.append(mse)
            records.append({'fold': colID, 'category': col, 'mse': mse})

        records.append({'fold': colID, 'category': aggregate_name, 'mse': np.mean(col_mses)})
    return records

def compute_marker_performance_perfold(data_cond, pred_sub, colonies, gene_names, thresh):
    """Per-fold (colony), per-marker binary positivity (data > thresh) vs predicted positivity
    -- balanced accuracy and F1 for the threshold-based call."""
    records = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
        for colID in np.unique(colonies):
            colidx = colonies == colID
            for marker in gene_names:
                markerpos = data_cond[colidx][marker].to_numpy() > thresh[marker]
                markerpos_pred = pred_sub[colidx][marker].to_numpy() > thresh[marker]
                bal_acc = balanced_accuracy_score(markerpos, markerpos_pred)
                f1_val = f1_score(markerpos, markerpos_pred, average='binary', pos_label=1, zero_division=0)
                f1_rand = markerpos.mean()
                records.append({'fold': colID, 'gene': marker, 'balanced_accuracy': bal_acc, 'f1': f1_val, 'f1_rand': f1_rand})
    return pd.DataFrame(records)


def plot_category_performance_perfold(perfold_df, labels, fname, fs=20, ms=10, group_col='category', tick_labels=None):
    """Per-fold dot plot for a SINGLE dataset/prediction set: compares its three built-in
    metrics (f1_rand, f1, balanced_accuracy) against each other, one row per category/gene
    (whichever column group_col names). `labels` must match the actual values in group_col
    for filtering; pass tick_labels separately if the y-axis display text should differ. To
    compare the SAME metric across multiple datasets/experiments instead (e.g. exp20 vs
    exp28, self vs cross), use plot_perfold_comparison."""
    tick_labels = tick_labels if tick_labels is not None else labels
    fig, ax = plt.subplots(figsize=(7, 0.31*len(labels) + 0.8))

    metric_colors = {'f1_rand': [0.7,0.7,0.7], 'f1': 'dodgerblue', 'balanced_accuracy': 'k'}
    offsets = {'f1_rand': 0, 'f1': -0.1, 'balanced_accuracy': 0.1}

    for row_i, cat in enumerate(labels):
        for metric, c in metric_colors.items():
            fold_vals = perfold_df[perfold_df[group_col] == cat].set_index('fold')[metric]

            if cat == labels[-1]:
                print(f'{cat} | {metric} = {fold_vals.mean():.3f}')

            y = row_i + offsets[metric]
            ax.scatter(fold_vals, [y]*len(fold_vals), color=c, s=(ms/2)**2, alpha=0.5, edgecolors='none', zorder=2)
            ax.scatter(fold_vals.mean(), y, color=c, s=ms**2, alpha=1.0, edgecolors='black', linewidths=0.6, zorder=3)

    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_yticks(range(len(labels)), tick_labels)
    ax.tick_params(axis='both', labelsize=fs)

    ax.set_ylim([-0.5, len(labels) - 0.5])
    ax.set_xlim([0, 1])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fname)



def plot_perfold_comparison(series, group_col, labels, fname, fs=18, ms=10, xlim=1, legend_labels=None,
                             row_spacing=1.0, row_height=0.3, offset_range=0.15, alpha=0.4, tick_labels=None):
    """Per-fold dot plot comparing multiple datasets/prediction sets (e.g. exp20 vs exp28,
    self vs cross, or several regression methods) at a SINGLE chosen metric, one row per
    category/gene. `series` is a list of (df, score_col, color) triples, each df sharing
    the same `group_col`/`score_col` schema. `labels` must match the actual values in
    group_col for filtering; pass `tick_labels` separately if the y-axis display text
    should differ (e.g. 'marker avg' shown for a 'marker_avg' grouping value). row_height
    sets the figure-height coefficient per label; row_spacing sets the vertical spread of
    row positions -- independent knobs. To compare several metrics of one dataset against
    each other instead, use plot_category_performance_perfold."""

    tick_labels = tick_labels if tick_labels is not None else labels
    fig, ax = plt.subplots(figsize=(7, row_height*len(labels) + 0.8))
    n = len(series)
    offsets = np.linspace(-offset_range, offset_range, n) if n > 1 else [0]
    row_positions = np.arange(len(labels)) * row_spacing

    for (df, score_col, color), offset in zip(series, offsets):
        for row_i, label in zip(row_positions, labels):
            fold_vals = df[df[group_col] == label][score_col]
            y = row_i + offset
            ax.scatter(fold_vals, [y]*len(fold_vals), color=color, s=(ms/2)**2, alpha=alpha, edgecolors='none', zorder=2)
            ax.scatter(fold_vals.mean(), y, color=color, s=ms**2, alpha=1.0, edgecolors='black', linewidths=0.6, zorder=3)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks(row_positions, tick_labels)
    ax.tick_params(axis='both', labelsize=fs)
    ax.set_ylim([-row_spacing/2, row_positions[-1] + row_spacing/2])
    ax.set_xlim([0, xlim])
    plt.gca().invert_yaxis()
    if legend_labels:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                      markersize=ms, label=name)
                           for (df, score_col, color), name in zip(series, legend_labels)]
        plt.legend(handles=legend_handles, fontsize=fs - 2)
    plt.tight_layout()
    plt.savefig(fname)



def add_group_avg(df, score_cols, group_col='gene', avg_label='marker_avg'):
    """Add an aggregate row per fold, averaging score_cols across all existing rows,
    labeled avg_label under group_col. Used to give plot_perfold_comparison a combined
    summary row (e.g. 'marker_avg') alongside the per-item rows, for dataframes that don't
    already have one built in (unlike compute_category_performance_perfold's aggregate_name,
    compute_marker_performance_perfold has no equivalent)."""
    if isinstance(score_cols, str):
        score_cols = [score_cols]
    avg = df.groupby('fold')[score_cols].mean().reset_index()
    avg[group_col] = avg_label
    return pd.concat([df, avg], ignore_index=True)


def save_df_as_image(df, filepath, col_width=1.7, row_height=0.4, fontsize=10,
                      header_color='#e8e8e8', row_alt_color='#f5f5f5'):
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(col_width * (n_cols + 1), row_height * (n_rows + 1)))
    ax.axis('off')

    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                      cellLoc='right', rowLoc='right', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('none')
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor(row_alt_color if row % 2 == 0 else 'white')

    plt.tight_layout()

    # position the header-separator line in FIGURE coordinates, not axes coordinates --
    # the table can overflow its axes' bbox (table.scale stretches rows independently of
    # the axes size), so ax.transAxes doesn't reliably map to the table's true footprint
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    header_cell = table[0, 0]
    y_fig = header_cell.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
    x0_fig = table[0, 0].get_window_extent(renderer).x0
    x1_fig = table[0, n_cols - 1].get_window_extent(renderer).x1
    x0_fig, _ = fig.transFigure.inverted().transform((x0_fig, 0))
    x1_fig, _ = fig.transFigure.inverted().transform((x1_fig, 0))

    import matplotlib.lines as mlines
    line = mlines.Line2D([x0_fig, x1_fig], [y_fig, y_fig], color='black', linewidth=1.2,
                          transform=fig.transFigure, clip_on=False)
    fig.add_artist(line)

    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    plt.close(fig)
    
def sig2fate(data, signals, gene_names, N_run, hyperparam):
    # returns average prediction for N_runs from list of signals
    
    preds = {}
    
    for it in range(N_run):
        
        # set seeds for reproducibility
        torch.manual_seed(it)
        np.random.seed(it)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(it)
            
        print(f"  Run {it}/{N_run}")
        start = time.time()
    
        pred_df = data[signals + gene_names].copy()
    
        # predict each colony in B50 based on the other colonies
        colony_idx = np.unique(data['Colony'])
        for test_colony in colony_idx:
    
            #print('B50 col: ' + str(test_colony))
            
            # split data into test and train by colonies
            train_colonies = np.setdiff1d(colony_idx, test_colony)
            data_test = data[data['Colony'].isin([test_colony])]
            data_train = data[data['Colony'].isin(train_colonies)]
    
            feat_train = data[signals] 
            feat_test = data[signals]
            tar_train = data[gene_names]
    
            # run VIB
            vib = VIB(feat_train, tar_train, hyperparam)
            _ = vib.train(verbose=False) 
            tar_predict = vib.predict(feat_test)
            pred_df.loc[tar_predict.index, tar_predict.columns] = tar_predict

        preds[it] = pred_df
    
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
    
    # average runs
    pred_mean_df = pd.concat(list(preds.values())).groupby(level=0).mean()

    # WARNING: THIS IS A PLACEHOLDER, I SOMEHOW DELETED A VERSION OF THIS CODE THAT RETURNS PREDICTION ON TRAINING DATA TO TEST OVERFITTING, CAN RESTORE LATER
    pred_mean_train_df = pred_mean_df
    
    return pred_mean_df, pred_mean_train_df

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
        self.hyperparam = {**defaults, **hyperparam} # merge dictionaries second overrides first
 
        N_DIM_INPUT = feat_train.shape[1]
        N_DIM_OUTPUT = tar_train.shape[1]

        # Standardize
        self.scaler_X_run = sklearn.preprocessing.StandardScaler()
        self.scaler_Y_run = sklearn.preprocessing.StandardScaler()

        self.tar_train = tar_train
        self.feat_train_z = self.scaler_X_run.fit_transform(feat_train)
        self.tar_train_z = self.scaler_Y_run.fit_transform(tar_train)

        # Create new VIB model (fresh random initialization each run)
        self.model = fns_NN.FlexibleVIB(
            input_dim=N_DIM_INPUT,
            output_dim=N_DIM_OUTPUT,
            latent_dim=self.hyperparam['LATENT_DIM'],
            hidden_dim=self.hyperparam['HIDDEN_DIM'],
            n_layers=self.hyperparam['N_LAYERS'],
            encoder_type='nonlinear',
            decoder_type='nonlinear'
        ).to(device)
        
        # Train
        #_ = self.train(verbose=False)

    def predict(self, feat_test):
        
        feat_test_z = self.scaler_X_run.transform(feat_test)
        X_test_run = torch.FloatTensor(feat_test_z).to(device)

        self.model.eval()
        with torch.no_grad():
            target_predict_z = self.model(X_test_run)[0].cpu().numpy()

        # Inverse transform
        target_predict = self.scaler_Y_run.inverse_transform(target_predict_z)

        # convert back to dataframe if appropriate
        if type(feat_test) == pd.core.frame.DataFrame and type(self.tar_train) == pd.core.frame.DataFrame:
            target_predict = pd.DataFrame(target_predict, index=feat_test.index, columns=self.tar_train.columns)
    
        return target_predict

    def train(self, patience=10, min_delta=1e-4, verbose=False, print_every=200):

        # Convert to torch
        X_train_run = torch.FloatTensor(self.feat_train_z).to(device)
        Y_train_run = torch.FloatTensor(self.tar_train_z).to(device)

        recon_losses = fns_NN.train_model(
            self.model, X_train_run, Y_train_run, is_vae=False,
            epochs=self.hyperparam['EPOCHS'],
            lr=self.hyperparam['LEARNING_RATE'],
            beta=self.hyperparam['BETA'],
            verbose=verbose, print_every=print_every,
        )

        return recon_losses

from sklearn.preprocessing import FunctionTransformer

class IdentityScaler:
    """Scaler that does nothing - for use when data is already normalized"""
    def __init__(self, n_features):
        self.mean_   = np.zeros(n_features)
        self.scale_  = np.ones(n_features)
        self.var_    = np.ones(n_features)
        self.n_features_in_ = n_features
    
    def transform(self, X):
        return np.array(X)
    
    def inverse_transform(self, X):
        return np.array(X)
    
    def fit(self, X):
        return self
    
    def fit_transform(self, X):
        return np.array(X)

class Metadata: 

    def __init__(self, data=None):

        self.xres = np.nan;
        self.yres = np.nan;
        self.channels = [];
        self.conditions = [];
    
        if data is not None:
            self.conditionPositions = data.groupby('condition')['Colony'].unique()

    def conditionStartPos(self, condition):
        # provide the first position for given condition
        return min(self.conditionPositions[condition])

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

    def scatter_fates(self, ms=1, legend=True, ax=None, thresh=1, fate='all', imageCoordinates=False):
    
        data = self.cellData['intensities']
        fates, fate_names = return_fates(data, thresh)
        colors = fns_plot.return_colmaps('fates')
    
        if imageCoordinates:
            X = self.cellData['XY']['X']
            Y = self.cellData['XY']['Y']
        else:
            X = (self.cellData['XY']['X'] - self.center[0]) * self.resolution
            Y = (self.cellData['XY']['Y'] - self.center[1]) * self.resolution
        
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        for i, f in enumerate(fate_names):
    
            idx = fates == f
            
            if (fate == 'all') or (fate == f):
                scatter = ax.scatter(X[idx], Y[idx], color=colmap_fates[f], s=ms, edgecolors='none')
            else:
                scatter = ax.scatter(X[idx], Y[idx], color='lightgray', s=ms, edgecolors='none')
                
        if legend:
            ax.legend(fate_names)
        ax.set_aspect('equal')
        ax.axis('off')
        
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
            
    def plotRadialProfiles(self, channel, condition, mode='cells', sigma=0, color = 'blue', ax=None, normalize=False, errorbars=True):

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
            if errorbars:
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
            if errorbars:
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
        if not errorbars:
            ax.set_ylim(top=max(y)*1.2)
    
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


def plot_radial_profiles_comparison(reference_data, figure_configs, columns, align_params_by_key,
                                     out_dir, out_prefix, normalize='background_mode', pred_variants=None,
                                     reference_label='reference', measured_label='measured'):
    """Radial profiles: a reference population (e.g. Exp20 B50), each figure_config's own
    B50 control (skipped when its own condition IS B50, to avoid duplicating the measured
    series), measured, and any predicted variants -- one figure per figure_configs entry,
    all columns (genes or signals) laid out in a single tightly-spaced row of square
    panels, each with a shaded std-between-colonies error band (matching
    Colony/MPexperiment.calcRadialProfiles's convention: mean and std of each colony's own
    interpolated profile, not pooled per-cell variance within a bin). Y-axis normalized
    per column so a data-grounded floor = 0 and the reference's peak = 1 (only 0/1 shown
    as ticks): normalize='background_mode' anchors 0 to the column's KDE-fitted background
    mode (fit_marker_distributions) -- appropriate for genes' bimodal background/
    foreground structure; normalize='minmax' anchors 0 to the reference profile's own
    observed minimum instead -- appropriate for signals, which don't have that same
    bimodal structure to fit a background mode from, but still benefits from a
    data-grounded (not literal-zero) floor. X-axis fixed to [0, 350] (colony radius, only
    endpoints shown as ticks). Y-limits shared ACROSS figure_configs per column (based on
    the normalized plotted means, extended to include the error bands) so the figures are
    directly comparable, and allowed to extend below 0 since some regions can genuinely
    average below the floor. reference_data: the reference population's raw dataframe
    (e.g. exp_result_20.data filtered to B50). figure_configs: list of (data, key, cond,
    label) -- one entry per output figure; data is that population's raw per-cell
    dataframe (e.g. exp_result_Z2.data or fig2n's data28), key indexes into
    align_params_by_key. columns: gene or signal names to plot, one per subplot panel.
    align_params_by_key: {key: {column: (scale, shift)}}, already fit (e.g. via
    fit_gene_alignment for genes, fit_zscore_alignment for signals) onto reference_data's
    own units. pred_variants: optional list of (pred_source, pred_label, color) triples,
    pred_source a {(key, cond): pred_df} dict keyed like avg_preds -- omit for an
    alignment-only comparison with no predictions. reference_label/measured_label:
    legend names for the reference and each figure_config's own-condition series (e.g.
    'Exp20'/'Exp28' for a two-experiment alignment check, vs. the default
    'reference'/'measured' for fig5-style perturbation-vs-control comparisons)."""
    pred_variants = pred_variants or []
    r_max = 350
    N_bins_x = 20
    n_cols = len(columns)
    n_rows = 1
    sw = 2  # spine linewidth

    def profile(vals, rdist):
        bins_x, mean_v, _, _ = fns_plot.calc_profile_meanvar(
            vals[np.newaxis, :, np.newaxis], rdist[np.newaxis, :, np.newaxis], N_bins_x, r_max)
        bins_x = bins_x.copy()
        bins_x[0] = 0             # first bin plotted at the true edge (CircleEdgeDist=0),
                                   # not its empirical midpoint
        bins_x[-1] = rdist.max()  # last bin plotted at the actual maximum observed
                                   # CircleEdgeDist (nearest the true center), not its
                                   # midpoint -- same logic as the first bin, applied
                                   # symmetrically to the other end
        return bins_x, mean_v[:, 0]

    def colony_std(df, col, bins_x):
        """Std across colonies of each colony's own radial profile (profile(), edge-
        anchored the same way), each interpolated onto bins_x so colonies with different
        cell counts/bin edges can be compared at the same x-positions -- the
        std-between-colonies convention, not pooled per-cell variance within a bin."""
        colony_profiles = []
        for colony_id in df['Colony'].unique():
            df_col = df[df['Colony'] == colony_id]
            if len(df_col) < 5:
                continue
            col_bins_x, col_mean_v = profile(df_col[col].to_numpy(), df_col['CircleEdgeDist'].to_numpy())
            interpolator = interp.interp1d(col_bins_x, col_mean_v, kind='linear',
                                            fill_value='extrapolate', bounds_error=False)
            colony_profiles.append(interpolator(bins_x))
        if len(colony_profiles) < 2:
            return np.full(len(bins_x), np.nan)
        return np.nanstd(np.array(colony_profiles), axis=0)

    # --- Per-column normalization anchors: floor (background mode or profile minimum) and
    # peak, both from reference_data. Maps floor -> 0, peak -> 1 below.
    floor = {}
    peak = {}
    if normalize == 'background_mode':
        fits_ref = fit_marker_distributions(reference_data, columns)
    for col in columns:
        _, mean_v = profile(reference_data[col].to_numpy(), reference_data['CircleEdgeDist'].to_numpy())
        peak[col] = np.nanmax(mean_v)
        if normalize == 'background_mode':
            fit = fits_ref[col]
            floor[col] = 10**fit['m'] - fit['offset']
        elif normalize == 'minmax':
            floor[col] = np.nanmin(mean_v)
        else:
            raise ValueError(f"normalize must be 'background_mode' or 'minmax', got {normalize!r}")

    # --- Pass 1: compute every series' radial profile (and colony-std error band) for
    # every figure_config up front, normalized so the floor -> 0 and reference_data's
    # peak -> 1 (the std is scaled but not floor-shifted, since it's a width, not a
    # position).
    profiles_by_figure = {}
    for data, key, cond, label in figure_configs:
        series_data = {reference_label: (reference_data, 'black')}

        if cond != 'B50':
            data_own_B50 = data[data['condition'] == 'B50']
            data_own_B50_aligned = apply_alignment(data_own_B50, align_params_by_key[key], columns)
            series_data['B50'] = (data_own_B50_aligned, 'gray')

        data_cond = data[data['condition'] == cond]
        data_cond_aligned = apply_alignment(data_cond, align_params_by_key[key], columns)
        series_data[measured_label] = (data_cond_aligned, 'steelblue')

        for pred_source, pred_label, pred_color in pred_variants:
            pred_cond_aligned = apply_alignment(pred_source[(key, cond)], align_params_by_key[key], columns)
            series_data[pred_label] = (pred_cond_aligned, pred_color)

        profiles_by_figure[label] = {}
        for col in columns:
            profiles_by_figure[label][col] = {}
            bg, pk = floor[col], peak[col]
            for name, (df, color) in series_data.items():
                bins_x, mean_v = profile(df[col].to_numpy(), df['CircleEdgeDist'].to_numpy())
                std_v = colony_std(df, col, bins_x)
                mean_v_norm = (mean_v - bg) / (pk - bg)
                std_v_norm = std_v / (pk - bg)
                profiles_by_figure[label][col][name] = (bins_x, mean_v_norm, std_v_norm, color)

    # --- Shared per-column y-limits, based on the normalized plotted means +/- error
    # bands (always spans at least [0, 1]; allowed to extend below 0 since some regions
    # can genuinely average below the floor).
    ylims = {}
    for col in columns:
        all_bounds = []
        for fig_profiles in profiles_by_figure.values():
            for (_, mean_v, std_v, _) in fig_profiles[col].values():
                all_bounds.append(mean_v - np.nan_to_num(std_v))
                all_bounds.append(mean_v + np.nan_to_num(std_v))
        combined = np.concatenate(all_bounds)
        lo, hi = np.nanmin(combined), np.nanmax(combined)
        pad = 0.05 * (hi - lo)
        ylims[col] = (min(0, lo - pad), max(1, hi + pad))

    # --- Pass 2: plot, one figure per figure_configs entry, all columns in a single
    # tightly-spaced row of square panels.
    for data, key, cond, label in figure_configs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8*n_cols, 2.5*n_rows),
                                  constrained_layout=True)
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)
        axes = np.atleast_1d(axes).ravel()

        for i, col in enumerate(columns):
            ax = axes[i]
            for name, (bins_x, mean_v, std_v, color) in profiles_by_figure[label][col].items():
                ax.fill_between(bins_x, mean_v - std_v, mean_v + std_v, alpha=0.2, color=color, edgecolor='none')
                ax.plot(bins_x, mean_v, color=color, label=name)
            ax.set_title(col, fontsize=10)
            ax.set_xlim(0, 350)
            ax.set_xticks([0, 350])
            ax.set_ylim(ylims[col])
            ax.set_yticks([0, 1])
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_linewidth(sw)
            ax.tick_params(width=sw, labelsize=10)

        axes[0].legend(fontsize=8, loc='upper right')
        fig.suptitle(label, fontsize=14)
        fig.supxlabel('radial position')
        fig.supylabel('intensity (ref units)')
        plt.savefig(f'{out_dir}/{out_prefix}_{label}.png', bbox_inches='tight')
        plt.show()



#----------------------------------------------------------------------------------------------------------------------------------
# ANALYSIS: FIT INTENSITY DISTRIBUTIONS AND ALIGN ACROSS EXPERIMENTS
#----------------------------------------------------------------------------------------------------------------------------------

from scipy.stats import gaussian_kde, norm
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment  # already imported elsewhere if not here

GAUSSIAN_HWHM_TO_SIGMA = np.sqrt(2 * np.log(2))

def log_transform(values):
    min_val = np.min(values)
    offset = 0.0 if min_val > 0 else (1.0 - min_val)
    return np.log10(values + offset), offset

def background_mode(log_vals, bandwidth_scale=1, peak_prominence_frac=0.00, peak_height_frac=0.01, n_grid=1000):
    kde_default = gaussian_kde(log_vals)
    bw = kde_default.scotts_factor() * bandwidth_scale
    kde = gaussian_kde(log_vals, bw_method=bw)

    grid = np.linspace(log_vals.min(), log_vals.max(), n_grid)
    density = kde(grid)

    peak_idx, _ = find_peaks(density,
                              prominence=peak_prominence_frac * density.max(),
                              height=peak_height_frac * density.max())
    if len(peak_idx) == 0:
        peak_idx = np.array([np.argmax(density)])

    peak_x = grid[peak_idx]
    m_lowest = peak_x.min()
    return m_lowest, peak_x, grid, density

def left_hwhm_sigma(grid, density, m):
    peak_idx = np.argmin(np.abs(grid - m))
    peak_height = density[peak_idx]
    half_height = 0.5 * peak_height
    idx = peak_idx
    while idx > 0 and density[idx] >= half_height:
        idx -= 1
    if idx == peak_idx or (idx == 0 and density[0] >= half_height):
        return np.nan
    x0, x1 = grid[idx], grid[idx+1]
    y0, y1 = density[idx], density[idx+1]
    x_cross = x0 + (half_height - y0) * (x1 - x0) / (y1 - y0)
    return (m - x_cross) / GAUSSIAN_HWHM_TO_SIGMA

def right_hwhm_sigma(grid, density, m):
    peak_idx = np.argmin(np.abs(grid - m))
    peak_height = density[peak_idx]
    half_height = 0.5 * peak_height
    idx = peak_idx
    while idx < len(grid) - 1 and density[idx] >= half_height:
        idx += 1
    if idx == peak_idx or (idx == len(grid) - 1 and density[-1] >= half_height):
        return np.nan
    x0, x1 = grid[idx-1], grid[idx]
    y0, y1 = density[idx-1], density[idx]
    x_cross = x0 + (half_height - y0) * (x1 - x0) / (y1 - y0)
    return (x_cross - m) / GAUSSIAN_HWHM_TO_SIGMA

def fit_marker_distributions(data_cond, markers, foreground_markers=None, bandwidth_scale=1,
                              peak_prominence_frac=0.00, peak_height_frac=0.01):
    """Per-marker background/foreground KDE-mode + HWHM-sigma fit. Same logic as
    fig2n.ipynb's discretization pipeline, generalized to any DataFrame/marker list so it
    can be run on a different experiment (e.g. a replicate) the same way. bandwidth_scale/
    peak_prominence_frac/peak_height_frac are forwarded to background_mode."""
    foreground_markers = foreground_markers or set()
    fits = {}
    for marker in markers:
        values = data_cond[marker].to_numpy()
        values = values[~np.isnan(values)]

        log_vals, offset = log_transform(values)
        m_lowest, all_modes, grid, density = background_mode(
            log_vals, bandwidth_scale=bandwidth_scale,
            peak_prominence_frac=peak_prominence_frac, peak_height_frac=peak_height_frac)
        n_modes = len(all_modes)

        if marker in foreground_markers and n_modes == 1:
            m, sigma_hwhm = np.nan, np.nan
            m_high = all_modes.max()
            sigma_hwhm_high = right_hwhm_sigma(grid, density, m_high)
        else:
            m = m_lowest
            sigma_hwhm = left_hwhm_sigma(grid, density, m)
            has_foreground = n_modes >= 2
            m_high = all_modes.max() if has_foreground else np.nan
            sigma_hwhm_high = right_hwhm_sigma(grid, density, m_high) if has_foreground else np.nan

        fits[marker] = dict(m=m, all_modes=all_modes, offset=offset, sigma_hwhm=sigma_hwhm,
                             m_high=m_high, sigma_hwhm_high=sigma_hwhm_high,
                             n_modes=n_modes, log_vals=log_vals, grid=grid, density=density)
    return fits


def posterior_crossing(grid, density, m, sigma, w, cutoff, direction):
    model_density = w * norm.pdf(grid, m, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(density > 0, model_density / density, np.nan)
    frac = np.clip(frac, 0, 1)

    peak_idx = np.argmin(np.abs(grid - m))
    idx = peak_idx

    if direction > 0:
        while idx < len(grid) - 1 and frac[idx] >= cutoff:
            idx += 1
        if idx == peak_idx or frac[idx] >= cutoff:
            return np.nan
        x0, x1 = grid[idx-1], grid[idx]
        y0, y1 = frac[idx-1], frac[idx]
    else:
        while idx > 0 and frac[idx] >= cutoff:
            idx -= 1
        if idx == peak_idx or frac[idx] >= cutoff:
            return np.nan
        x0, x1 = grid[idx], grid[idx+1]
        y0, y1 = frac[idx], frac[idx+1]

    return x0 + (cutoff - y0) * (x1 - x0) / (y1 - y0)

def equal_posterior_crossing(grid, m, sigma_bg, w_bg, m_high, sigma_fg, w_fg):
    mask = (grid >= m) & (grid <= m_high)
    sub_grid = grid[mask]
    bg = w_bg * norm.pdf(sub_grid, m, sigma_bg)
    fg = w_fg * norm.pdf(sub_grid, m_high, sigma_fg)
    diff = bg - fg

    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return np.nan
    idx = sign_change[0]
    x0, x1 = sub_grid[idx], sub_grid[idx+1]
    y0, y1 = diff[idx], diff[idx+1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)

def fit_posterior_thresholds(fits, markers, thresh_manual=None, strict_foreground_markers=None,
                              strict_foreground_cutoff=0.9, foreground_fallback_cutoff=0.5,
                              background_fallback_cutoff=0.05):
    """Per-marker posterior-crossing threshold, plus a record of which rule fired and
    the fitted background/foreground parameters -- same logic as fig2n.ipynb's threshold
    cell, generalized so it can be run on any experiment's `fits` (see fit_marker_distributions)."""
    strict_foreground_markers = strict_foreground_markers or set()
    thresh_posterior = {}
    records = []

    for marker in markers:
        fit = fits[marker]
        has_bg = not np.isnan(fit['sigma_hwhm'])
        has_fg = not np.isnan(fit['sigma_hwhm_high'])

        w_bg = 2 * np.mean(fit['log_vals'] < fit['m']) if has_bg else np.nan
        w_fg = 2 * np.mean(fit['log_vals'] > fit['m_high']) if has_fg else np.nan

        if marker in strict_foreground_markers and has_fg:
            method = 'strict_foreground_cutoff'
            x_cross = posterior_crossing(fit['grid'], fit['density'], fit['m_high'], fit['sigma_hwhm_high'],
                                          w_fg, strict_foreground_cutoff, direction=-1)
        elif has_bg and has_fg:
            method = 'equal_posterior_crossing'
            x_cross = equal_posterior_crossing(fit['grid'], fit['m'], fit['sigma_hwhm'], w_bg,
                                                fit['m_high'], fit['sigma_hwhm_high'], w_fg)
        elif has_fg:
            method = 'foreground_fallback'
            x_cross = posterior_crossing(fit['grid'], fit['density'], fit['m_high'], fit['sigma_hwhm_high'],
                                          w_fg, foreground_fallback_cutoff, direction=-1)
        elif has_bg:
            method = 'background_fallback'
            x_cross = posterior_crossing(fit['grid'], fit['density'], fit['m'], fit['sigma_hwhm'],
                                          w_bg, background_fallback_cutoff, direction=1)
        else:
            method = 'none'
            x_cross = np.nan

        thresh_posterior[marker] = 10**x_cross - fit['offset'] if not np.isnan(x_cross) else np.nan

        frac_below_mode = np.mean(fit['log_vals'] < fit['m']) if not np.isnan(fit['m']) else np.nan
        records.append({
            'marker': marker, 'offset': fit['offset'],
            'm_bg': fit['m'], 'sigma_bg': fit['sigma_hwhm'], 'w_bg': w_bg,
            'm_fg': fit['m_high'], 'sigma_fg': fit['sigma_hwhm_high'], 'w_fg': w_fg,
            'n_modes': fit['n_modes'],
            'mode_suspicious': frac_below_mode < 0.02 if not np.isnan(frac_below_mode) else False,
            'method': method, 'x_cross_log': x_cross, 'threshold_posterior': thresh_posterior[marker],
            'threshold_manual': thresh_manual[marker] if thresh_manual is not None else np.nan,
        })

    return thresh_posterior, records


def robust_range(*value_arrays, lo_pct=0.5, hi_pct=99.5):
    """Percentile-based (lo, hi) range spanning all given value arrays -- robust to a single
    isolated outlier point dragging a shared plotting/fitting range out unnecessarily."""
    lo = min(np.percentile(v, lo_pct) for v in value_arrays)
    hi = max(np.percentile(v, hi_pct) for v in value_arrays)
    return lo, hi

def optimize_intensity_scale(vals_self, vals_ref, m_self, m_ref, n_grid=200, n_coarse=25, n_fine=40,
                              scale_bounds=(0.2, 5.0), eps=1e-6, log_density=True, bandwidth_scale=1,
                              reg_strength=0.0):
    """Grid-search scale (shift fixed by peak-anchoring m_self->m_ref) minimizing log-density
    mismatch between KDE-smoothed linear-intensity distributions, optionally regularized
    toward the raw std ratio (vals_ref.std()/vals_self.std()) in log-space via reg_strength.
    Added because the unregularized fit was found to systematically over-scale relative to
    the raw std ratio (~15% on average across genes); a KDE-bandwidth sweep ruled out
    smoothing as the cause, so the fix constrains the fit directly rather than chasing the
    exact mechanism. reg_strength=0 (default) recovers the original, unregularized behavior."""
    lo, hi = robust_range(vals_ref)
    grid = np.linspace(lo, hi, n_grid)

    kde_ref_default = gaussian_kde(vals_ref)
    kde_ref = gaussian_kde(vals_ref, bw_method=kde_ref_default.scotts_factor() * bandwidth_scale)
    ref_density = kde_ref(grid)
    if log_density:
        ref_density = np.log(ref_density + eps)

    kde_self_default = gaussian_kde(vals_self)
    kde_self = gaussian_kde(vals_self, bw_method=kde_self_default.scotts_factor() * bandwidth_scale)

    log_scale_prior = np.log(np.std(vals_ref) / np.std(vals_self))

    def cost(scale):
        shift = m_ref - scale * m_self
        self_density = kde_self((grid - shift) / scale) / scale
        if log_density:
            self_density = np.log(self_density + eps)
        density_cost = np.mean((ref_density - self_density) ** 2)
        reg_cost = reg_strength * (np.log(scale) - log_scale_prior) ** 2
        return density_cost + reg_cost

    def grid_search(scale_range, n):
        scales = np.linspace(*scale_range, n)
        costs = np.array([cost(s) for s in scales])
        best_idx = np.argmin(costs)
        return scales[best_idx], scales[1] - scales[0]

    scale, step = grid_search(scale_bounds, n_coarse)
    fine_range = (max(scale_bounds[0], scale - 2*step), min(scale_bounds[1], scale + 2*step))
    scale, _ = grid_search(fine_range, n_fine)
    shift = m_ref - scale * m_self
    return scale, shift


def dominant_mode_linear(fit):
    """Linear-intensity position of whichever KDE mode (background or foreground) has more
    cells under it -- the peak-anchoring point. Reuses an already-computed fits[marker]
    (see fit_marker_distributions) instead of refitting."""
    log_vals, all_modes, offset = fit['log_vals'], fit['all_modes'], fit['offset']
    if len(all_modes) == 1:
        m_log = all_modes[0]
    else:
        m_lowest, m_high = all_modes.min(), all_modes.max()
        midpoint = (m_lowest + m_high) / 2
        m_log = m_lowest if np.mean(log_vals < midpoint) >= 0.5 else m_high
    return 10**m_log - offset


def fit_gene_alignment(data_self, data_ref, markers, fits_self=None, fits_ref=None, condition='B50',
                        bandwidth_scale=1, reg_strength=0.0):
    """Peak-anchored shift + log-density-matched scale, per marker, mapping data_self's raw
    units onto data_ref's. Pass fits_self/fits_ref (from fit_marker_distributions) to reuse
    an already-computed KDE fit instead of refitting. bandwidth_scale/reg_strength are
    forwarded to optimize_intensity_scale."""
    data_self_cond = data_self[data_self['condition'] == condition]
    data_ref_cond = data_ref[data_ref['condition'] == condition]

    fits_self = fits_self or fit_marker_distributions(data_self_cond, markers)
    fits_ref = fits_ref or fit_marker_distributions(data_ref_cond, markers)

    align_params = {}
    for marker in markers:
        vals_self = data_self_cond[marker].to_numpy()
        vals_self = vals_self[~np.isnan(vals_self)]
        vals_ref = data_ref_cond[marker].to_numpy()
        vals_ref = vals_ref[~np.isnan(vals_ref)]

        m_self = dominant_mode_linear(fits_self[marker])
        m_ref = dominant_mode_linear(fits_ref[marker])
        align_params[marker] = optimize_intensity_scale(vals_self, vals_ref, m_self, m_ref,
                                                          bandwidth_scale=bandwidth_scale,
                                                          reg_strength=reg_strength)
    return align_params


def apply_alignment(data, align_params, columns):
    """Apply a fitted {column: (scale, shift)} mapping to a DataFrame's columns."""
    data_aligned = data.copy()
    for col in columns:
        scale, shift = align_params[col]
        data_aligned[col] = data[col] * scale + shift
    return data_aligned

def cutoff_bounds(cutoff, factor=10):
    """Given a posterior-probability cutoff (e.g. STRICT_FOREGROUND_CUTOFF), return
    (low, high) cutoffs corresponding to the underlying odds shifted by `factor` in each
    direction -- used to bracket a threshold-sensitivity range around a chosen cutoff."""
    odds = cutoff / (1 - cutoff)
    odds_low, odds_high = odds / factor, odds * factor
    return odds_low / (1 + odds_low), odds_high / (1 + odds_high)


def odds_crossing(grid, m, sigma_bg, w_bg, m_high, sigma_fg, w_fg, ratio):
    """Position where the posterior odds of foreground vs background (fg(x)/bg(x)) first
    crosses `ratio`, searched between the background and foreground peaks. Odds increase
    monotonically from ~0 near m to very large near m_high and can span many orders of
    magnitude, so solved in log space rather than as a raw ratio."""
    mask = (grid >= m) & (grid <= m_high)
    sub_grid = grid[mask]

    log_bg = np.log(w_bg) + norm.logpdf(sub_grid, m, sigma_bg)
    log_fg = np.log(w_fg) + norm.logpdf(sub_grid, m_high, sigma_fg)
    log_diff = (log_fg - log_bg) - np.log(ratio)

    sign_change = np.where(np.diff(np.sign(log_diff)) != 0)[0]
    if len(sign_change) == 0:
        return sub_grid[0] if log_diff[0] > 0 else sub_grid[-1]

    idx = sign_change[0]
    x0, x1 = sub_grid[idx], sub_grid[idx+1]
    y0, y1 = log_diff[idx], log_diff[idx+1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)
