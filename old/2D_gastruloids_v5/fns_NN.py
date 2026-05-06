#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Classes and Training Functions
Contains VAE and VIB models with flexible architectures
"""

import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#print("Using device:", device)

import torch.nn as nn
import torch.optim as optim
import numpy as np


class FlexibleVIB(nn.Module):
    """Variational Information Bottleneck: X -> Z -> Y"""
    def __init__(self, input_dim, output_dim, latent_dim, hidden_dim=128, 
                 n_layers=2, encoder_type='nonlinear', decoder_type='nonlinear'):
        super(FlexibleVIB, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        
        # Encoder
        if encoder_type == 'nonlinear':
            encoder_layers = []
            prev_dim = input_dim
            for i in range(n_layers):
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            self.encoder_layers = nn.ModuleList(encoder_layers)
            self.enc_mu = nn.Linear(hidden_dim, latent_dim)
            self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            self.enc_mu = nn.Linear(input_dim, latent_dim)
            self.enc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder
        if decoder_type == 'nonlinear':
            decoder_layers = []
            prev_dim = latent_dim
            for i in range(n_layers):
                decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            decoder_layers.append(nn.Linear(hidden_dim, output_dim))
            self.decoder_layers = nn.ModuleList(decoder_layers)
        else:
            self.dec_out = nn.Linear(latent_dim, output_dim)
        
    def encode(self, x):
        if self.encoder_type == 'nonlinear':
            h = x
            for layer in self.encoder_layers:
                h = torch.relu(layer(h))
            return self.enc_mu(h), self.enc_logvar(h)
        else:
            return self.enc_mu(x), self.enc_logvar(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        if self.decoder_type == 'nonlinear':
            h = z
            for i, layer in enumerate(self.decoder_layers):
                h = layer(h)
                if i < len(self.decoder_layers) - 1:
                    h = torch.relu(h)
            return h
        else:
            return self.dec_out(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class FlexibleVAE(nn.Module):
    """Variational Autoencoder: X -> Z -> X"""
    def __init__(self, input_dim, latent_dim, hidden_dim=128, 
                 n_layers=2, encoder_type='nonlinear', decoder_type='nonlinear'):
        super(FlexibleVAE, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        
        # Encoder
        if encoder_type == 'nonlinear':
            encoder_layers = []
            prev_dim = input_dim
            for i in range(n_layers):
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            self.encoder_layers = nn.ModuleList(encoder_layers)
            self.enc_mu = nn.Linear(hidden_dim, latent_dim)
            self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            self.enc_mu = nn.Linear(input_dim, latent_dim)
            self.enc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder
        if decoder_type == 'nonlinear':
            decoder_layers = []
            prev_dim = latent_dim
            for i in range(n_layers):
                decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            decoder_layers.append(nn.Linear(hidden_dim, input_dim))
            self.decoder_layers = nn.ModuleList(decoder_layers)
        else:
            self.dec_out = nn.Linear(latent_dim, input_dim)
        
    def encode(self, x):
        if self.encoder_type == 'nonlinear':
            h = x
            for layer in self.encoder_layers:
                h = torch.relu(layer(h))
            return self.enc_mu(h), self.enc_logvar(h)
        else:
            return self.enc_mu(x), self.enc_logvar(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        if self.decoder_type == 'nonlinear':
            h = z
            for i, layer in enumerate(self.decoder_layers):
                h = layer(h)
                if i < len(self.decoder_layers) - 1:
                    h = torch.relu(h)
            return h
        else:
            return self.dec_out(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def compute_loss(recon, target, mu, logvar, beta=1.0):
    """
    Compute VAE/VIB loss
    
    Parameters:
    -----------
    recon : torch.Tensor
        Reconstructed output
    target : torch.Tensor
        Target output
    mu : torch.Tensor
        Mean of latent distribution
    logvar : torch.Tensor
        Log variance of latent distribution
    beta : float
        Weight for KL divergence term
        
    Returns:
    --------
    total_loss : torch.Tensor
        Total loss (reconstruction + beta * KL)
    recon_loss : torch.Tensor
        Reconstruction loss (MSE)
    kl_loss : torch.Tensor
        KL divergence loss
    """
    recon_loss = nn.MSELoss()(recon, target)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_model(model, X_data, Y_data, is_vae=False, epochs=800, lr=1e-3, 
                beta=1.0, verbose=False, print_every=200):
    """
    Train VAE or VIB model
    
    Parameters:
    -----------
    model : nn.Module
        VAE or VIB model
    X_data : torch.Tensor
        Input data
    Y_data : torch.Tensor
        Output data (used for VIB, ignored for VAE)
    is_vae : bool
        If True, train as VAE (X->X), else train as VIB (X->Y)
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    beta : float
        Weight for KL divergence
    verbose : bool
        If True, print training progress
    print_every : int
        Print progress every N epochs
        
    Returns:
    --------
    recon_losses : list
        List of reconstruction losses per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if is_vae:
            recon, mu, logvar = model(X_data)
            target = X_data
        else:
            recon, mu, logvar = model(X_data)
            target = Y_data
        
        loss, recon_loss, kl_loss = compute_loss(recon, target, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        
        recon_losses.append(recon_loss.item())
        
        if verbose and (epoch + 1) % print_every == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
    return recon_losses


def evaluate_model(model, X_data, Y_data=None, is_vae=False):
    """
    Evaluate trained model
    
    Parameters:
    -----------
    model : nn.Module
        Trained VAE or VIB model
    X_data : torch.Tensor
        Input data
    Y_data : torch.Tensor, optional
        Output data (for VIB evaluation)
    is_vae : bool
        If True, evaluate as VAE (X->X), else as VIB (X->Y)
        
    Returns:
    --------
    final_loss : float
        Final reconstruction loss
    """
    model.eval()
    with torch.no_grad():
        if is_vae:
            recon, _, _ = model(X_data)
            final_loss = nn.MSELoss()(recon, X_data).item()
        else:
            recon, _, _ = model(X_data)
            final_loss = nn.MSELoss()(recon, Y_data).item()
    
    return final_loss


def clean_data(feature, target):
    """
    Remove NaN entries from feature and target arrays
    
    Parameters:
    -----------
    feature : np.ndarray
        Feature array (may contain NaNs)
    target : np.ndarray
        Target array (may contain NaNs)
        
    Returns:
    --------
    feature_clean : np.ndarray
        Feature array with NaN rows removed
    target_clean : np.ndarray
        Target array with NaN rows removed
    indices : np.ndarray
        Indices of non-NaN rows
    """
    
    # Reshape to 2D if needed
    feature_flat = feature.reshape(-1, feature.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    
    # Find rows without NaN in either feature or target
    feature_mask = ~np.isnan(feature_flat).any(axis=1)
    target_mask = ~np.isnan(target_flat).any(axis=1)
    valid_mask = feature_mask & target_mask
    
    indices = np.where(valid_mask)[0]
    
    feature_clean = feature_flat[valid_mask]
    target_clean = target_flat[valid_mask]
    
    return feature_clean, target_clean, indices


def clean_data_full_v2(feature, target, metricdist, markers):
    """
    Clean data and corresponding metadata by removing NaN entries
    
    Parameters:
    -----------
    feature : np.ndarray
        Feature array with shape (N_sys, N_part_max, N_features)
    target : np.ndarray
        Target array with shape (N_sys, N_part_max, N_targets)
    metricdist : np.ndarray
        Metric distance array with shape (N_sys, N_part_max)
    markers : np.ndarray
        Marker array with shape (N_sys, N_part_max)
        
    Returns:
    --------
    feature_clean : np.ndarray
        Cleaned feature array
    target_clean : np.ndarray
        Cleaned target array
    metricdist_clean : np.ndarray
        Cleaned metric distance array
    markers_clean : np.ndarray
        Cleaned markers array
    """

    N_sys = feature.shape[0]
    N_part_max = feature.shape[1]
    
    feature_clean, target_clean, indices = clean_data(feature, target)

    metricdist_clean = metricdist.reshape((int(N_sys * N_part_max)))[indices]
    markers_clean = markers.reshape((int(N_sys * N_part_max)))[indices]
    
    return feature_clean, target_clean, metricdist_clean, markers_clean


def test_train_split_colonies(data, feature, target, train_size=3):
    """
    Split data into train and test sets by colony (not by individual cells)
    
    Parameters:
    -----------
    data : object
        Data object containing metricdist and markers attributes
    feature : np.ndarray
        Feature array with shape (N_sys, N_part_max, N_features)
    target : np.ndarray
        Target array with shape (N_sys, N_part_max, N_targets)
    train_size : int
        Number of colonies to use for training
        
    Returns:
    --------
    feat_train : np.ndarray
        Training features
    feat_test : np.ndarray
        Testing features
    tar_train : np.ndarray
        Training targets
    tar_test : np.ndarray
        Testing targets
    metricdist_train : np.ndarray
        Training metric distances
    metricdist_test : np.ndarray
        Testing metric distances
    markers_train : np.ndarray
        Training markers
    markers_test : np.ndarray
        Testing markers
    """
    
    # Split test/train by colony
    N_sys = feature.shape[0]
    N_tar = target.shape[2]
    
    test_size = N_sys - train_size
    
    # Use first N=test_size colonies for testing, so colony 1 can be used for plotting
    feat_test = feature[:test_size, :, :]
    tar_test = target[:test_size, :, :]
    metricdist_test = data.metricdist[:test_size, :]
    markers_test = data.markers[:test_size, :]
    
    # Use last colonies for training
    feat_train = feature[-train_size:, :, :] 
    tar_train = target[-train_size:, :, :]
    metricdist_train = data.metricdist[-train_size:, :]
    markers_train = data.markers[-train_size:, :]

    # Clean train data
    feat_train, tar_train, metricdist_train, markers_train = clean_data_full_v2(
        feat_train, tar_train, metricdist_train, markers_train
    )
    
    # Clean test data
    feat_test, tar_test, metricdist_test, markers_test = clean_data_full_v2(
        feat_test, tar_test, metricdist_test, markers_test
    )
    
    if N_tar == 1:
        tar_train = tar_train.ravel()
        tar_test = tar_test.ravel()
    
    return (feat_train, feat_test, tar_train, tar_test, 
            metricdist_train, metricdist_test, markers_train, markers_test)