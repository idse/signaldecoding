import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
import os
import pickle

# ============================================
# Configuration
# ============================================

MODE_RUN = 0   # Run analysis, save to data_toy_model/data/
MODE_PLOT = True  # Load data, save plots to data_toy_model/plots_vib/

output_dir = 'data_toy_model'
data_dir = f'{output_dir}/data'
plot_dir = f'{output_dir}/plots_vib'

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ============================================
# Colormap for fates
# ============================================

colmap_fates = {'AMLC':[90/255,166/255,71/255,1],'PGCLC':[227/255,143/255,52/255,1],
                 'PSLC':[211/255,62/255,43/255,1], 'meso':[140/255,40/255,93/255,1],
                 'pluri':[75/255,167/255,158/255,1], 'ecto':[49/255,118/255,181/255,1], 
                'endo':[227/255,179/255,61/255,1],'other':[0.8,0.8,0.8,1]}

fate_colors = [colmap_fates['AMLC'], colmap_fates['PGCLC'], colmap_fates['PSLC']]

# ============================================
# VIB Model
# ============================================

class FlexibleVIB(nn.Module):
    """Variational Information Bottleneck: X -> Z -> Y"""
    def __init__(self, input_dim, output_dim, latent_dim, hidden_dim=128, 
                 n_layers=2, encoder_type='nonlinear', decoder_type='nonlinear'):
        super(FlexibleVIB, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        
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


def train_vib(model, X_train, Y_train, epochs=1000, lr=1e-3, beta=0.01, verbose=True):
    """Train VIB model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(X_train)
        
        recon_loss = nn.MSELoss()(recon, Y_train)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X_train.size(0)
        
        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 200 == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
    return model


# ============================================
# Generative model functions
# ============================================

def compute_L(s1, s2):
    return s1 - s2

def hill_increasing(L, K=0, n=4):
    L_shifted = L + 1
    K_shifted = K + 1
    return L_shifted**n / (K_shifted**n + L_shifted**n)

def hill_decreasing(L, K=0, n=4):
    return 1 - hill_increasing(L, K, n)

def compute_genes(L):
    g1 = hill_increasing(L, K=0, n=4)
    g2 = hill_decreasing(L, K=0, n=4)
    return g1, g2

def assign_fate(L):
    fate = np.zeros_like(L, dtype=int)
    fate[L < -0.33] = 0
    fate[(L >= -0.33) & (L <= 0.33)] = 1
    fate[L > 0.33] = 2
    return fate

def assign_fate_from_genes(g1, g2):
    K_shifted = 1.0
    n = 4
    g1_clipped = np.clip(g1, 1e-6, 1 - 1e-6)
    L_shifted = K_shifted * (g1_clipped / (1 - g1_clipped)) ** (1/n)
    L_pred = L_shifted - 1
    return assign_fate(L_pred)

def sample_gaussian_mixture(means, cov, n_samples, weights=None):
    n_components = len(means)
    if weights is None:
        weights = np.ones(n_components) / n_components
    
    component_idx = np.random.choice(n_components, size=n_samples, p=weights)
    
    samples = np.zeros((n_samples, 2))
    for i in range(n_components):
        mask = component_idx == i
        n_comp = mask.sum()
        if n_comp > 0:
            samples[mask] = np.random.multivariate_normal(means[i], cov, n_comp)
    
    samples = np.clip(samples, 0, 1)
    return samples[:, 0], samples[:, 1]

def discretize_signal(s, n_bins=10):
    return np.digitize(s, bins=np.linspace(0, 1, n_bins + 1)[1:-1])


# ============================================
# MODE_RUN: Generate data and train model
# ============================================

if MODE_RUN:
    np.random.seed(40)
    torch.manual_seed(40)
    
    n_display = 500    # Number of points to display
    n_train = 10000    # Number of points for training
    
    # Generate display data for conditions
    # Condition 1: Uniform (Standard) - for display
    s1_cond1 = np.random.uniform(0, 1, n_display)
    s2_cond1 = np.random.uniform(0, 1, n_display)
    
    # Condition 2: GMM peaked on Fates 0 & 1
    means_cond2 = [[0.2, 0.65], [0.35, 0.45]]
    cov_cond2 = [[0.02, 0.0], [0.0, 0.02]]
    s1_cond2, s2_cond2 = sample_gaussian_mixture(means_cond2, cov_cond2, n_display)
    
    # Condition 3: GMM peaked on Fates 1 & 2
    means_cond3 = [[0.45, 0.35], [0.65, 0.2]]
    cov_cond3 = [[0.02, 0.0], [0.0, 0.02]]
    s1_cond3, s2_cond3 = sample_gaussian_mixture(means_cond3, cov_cond3, n_display)

    conditions = [
        {'name': 'Standard condition', 's1': s1_cond1, 's2': s2_cond1},
        {'name': 'Condition 1', 's1': s1_cond2, 's2': s2_cond2},
        {'name': 'Condition 2', 's1': s1_cond3, 's2': s2_cond3},
    ]
    
    for cond in conditions:
        cond['L'] = compute_L(cond['s1'], cond['s2'])
        cond['g1'], cond['g2'] = compute_genes(cond['L'])
        cond['fate'] = assign_fate(cond['L'])
    
    # Generate large training dataset (uniform distribution)
    s1_train_full = np.random.uniform(0, 1, n_train)
    s2_train_full = np.random.uniform(0, 1, n_train)
    L_train_full = compute_L(s1_train_full, s2_train_full)
    g1_train_full, g2_train_full = compute_genes(L_train_full)
    
    X_train = torch.tensor(np.column_stack([s1_train_full, s2_train_full]), dtype=torch.float32)
    Y_train = torch.tensor(np.column_stack([g1_train_full, g2_train_full]), dtype=torch.float32)
    
    # ========== Train VIB with latent_dim=1 ==========
    print(f"Training VIB model (latent_dim=1) on {n_train} samples...")
    model_1d = FlexibleVIB(input_dim=2, output_dim=2, latent_dim=1, hidden_dim=64, n_layers=2)
    model_1d = train_vib(model_1d, X_train, Y_train, epochs=1000, lr=1e-3, beta=0.01)
    
    # ========== Train VIB with latent_dim=2 ==========
    print(f"\nTraining VIB model (latent_dim=2) on {n_train} samples...")
    torch.manual_seed(42)  # Reset seed for reproducibility
    model_2d = FlexibleVIB(input_dim=2, output_dim=2, latent_dim=2, hidden_dim=64, n_layers=2)
    model_2d = train_vib(model_2d, X_train, Y_train, epochs=1000, lr=1e-3, beta=0.01)
    
    # Apply trained models to all conditions
    model_1d.eval()
    model_2d.eval()
    
    for cond in conditions:
        X = torch.tensor(np.column_stack([cond['s1'], cond['s2']]), dtype=torch.float32)
        
        # 1D model
        with torch.no_grad():
            Y_pred, mu_1d, _ = model_1d(X)
        cond['g1_pred'] = Y_pred[:, 0].numpy()
        cond['g2_pred'] = Y_pred[:, 1].numpy()
        cond['L_pred'] = mu_1d[:, 0].numpy()
        cond['fate_pred'] = assign_fate_from_genes(cond['g1_pred'], cond['g2_pred'])
        
        # 2D model
        with torch.no_grad():
            _, mu_2d, _ = model_2d(X)
        cond['L1'] = mu_2d[:, 0].numpy()
        cond['L2'] = mu_2d[:, 1].numpy()
    
    # Compute mutual information
    for cond in conditions:
        s1_disc = discretize_signal(cond['s1'])
        s2_disc = discretize_signal(cond['s2'])
        cond['MI_s1_fate'] = mutual_info_score(s1_disc, cond['fate'])
        cond['MI_s2_fate'] = mutual_info_score(s2_disc, cond['fate'])
        cond['MI_fate_pred'] = mutual_info_score(cond['fate_pred'], cond['fate'])
    
    # Evaluate 1D latent on high-res grid
    grid_res = 100
    s1_grid = np.linspace(0, 1, grid_res)
    s2_grid = np.linspace(0, 1, grid_res)
    S1_grid, S2_grid = np.meshgrid(s1_grid, s2_grid)
    X_grid = torch.tensor(np.column_stack([S1_grid.ravel(), S2_grid.ravel()]), dtype=torch.float32)
    
    with torch.no_grad():
        _, mu_grid, _ = model_1d(X_grid)
    L_grid = mu_grid[:, 0].numpy().reshape(grid_res, grid_res)
    
    # Compute L vs s1 for different s2 values (only where data exists)
    s2_values = [0.2, 0.5, 0.8]
    s2_tolerance = 0.1
    s1_line_full = np.linspace(0, 1, 100)
    L_vs_s1 = {}
    s1_ranges = {}
    
    # Use standard condition data to find valid s1 ranges for each s2 slice
    std_s1 = conditions[0]['s1']
    std_s2 = conditions[0]['s2']
    
    for s2_val in s2_values:
        # Find data points near this s2 value
        mask = np.abs(std_s2 - s2_val) < s2_tolerance
        if mask.sum() > 0:
            s1_min = std_s1[mask].min()
            s1_max = std_s1[mask].max()
        else:
            s1_min, s1_max = 0, 1
        
        s1_ranges[s2_val] = (s1_min, s1_max)
        
        # Create s1 line only in the valid range
        s1_line = np.linspace(s1_min, s1_max, 100)
        X_line = torch.tensor(np.column_stack([s1_line, np.full_like(s1_line, s2_val)]), dtype=torch.float32)
        with torch.no_grad():
            _, mu_line, _ = model_1d(X_line)
        L_vs_s1[s2_val] = (s1_line, mu_line[:, 0].numpy())
    
    # Linear regression on uniform condition
    X_reg = np.column_stack([conditions[0]['s1'], conditions[0]['s2']])
    y_reg = conditions[0]['L_pred']
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    coefs = reg.coef_
    intercept = reg.intercept_
    r2 = reg.score(X_reg, y_reg)
    
    print(f"\nLinear regression: L_pred = {coefs[0]:.3f}*s1 + {coefs[1]:.3f}*s2 + {intercept:.3f}")
    print(f"R² = {r2:.4f}")
    
    # Save data
    data = {
        'conditions': conditions,
        'model_1d_state': model_1d.state_dict(),
        'model_2d_state': model_2d.state_dict(),
        'S1_grid': S1_grid,
        'S2_grid': S2_grid,
        'L_grid': L_grid,
        's2_values': s2_values,
        's1_ranges': s1_ranges,
        'L_vs_s1': L_vs_s1,
        'coefs': coefs,
        'intercept': intercept,
        'r2': r2,
    }
    with open(f'{data_dir}/vib_results.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved data to {data_dir}/vib_results.pkl")


# ============================================
# MODE_PLOT: Load data and create figures
# ============================================

if MODE_PLOT:
    # Load data
    with open(f'{data_dir}/vib_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    conditions = data['conditions']
    S1_grid = data['S1_grid']
    S2_grid = data['S2_grid']
    L_grid = data['L_grid']
    s2_values = data['s2_values']
    s1_ranges = data['s1_ranges']
    L_vs_s1 = data['L_vs_s1']
    coefs = data['coefs']
    intercept = data['intercept']
    r2 = data['r2']
    
    # ============================================
    # Figure 1: Scatter plots and MI bar plots
    # ============================================
    
    fig, axes = plt.subplots(2, 3, figsize=(6, 4.5))
    
    # Row 1: Scatter plots
    for ax, cond in zip(axes[0], conditions):
        for fate_idx in [0, 1, 2]:
            mask = cond['fate'] == fate_idx
            ax.scatter(cond['s1'][mask], cond['s2'][mask], 
                       c=[fate_colors[fate_idx]], s=8, alpha=0.6)
        
        ax.set_xlabel('$s_1$')
        ax.set_ylabel('$s_2$')
        ax.set_title(cond['name'], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Row 2: MI bar plots
    std_mi = [conditions[0]['MI_s1_fate'], conditions[0]['MI_s2_fate'], conditions[0]['MI_fate_pred']]
    mi_ylim = 1.2 * max(std_mi)
    
    bar_colors = ['#636363', '#969696', '#31a354']
    bar_labels = ['$I(s_1, F)$', '$I(s_2, F)$', '$I(F^*, F)$']
    
    for ax, cond in zip(axes[1], conditions):
        mi_values = [cond['MI_s1_fate'], cond['MI_s2_fate'], cond['MI_fate_pred']]
        bars = ax.bar(range(3), mi_values, color=bar_colors)
        ax.set_xticks(range(3))
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel('MI')
        ax.set_ylim(0, mi_ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/fig1_conditions_mi.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{plot_dir}/fig1_conditions_mi.pdf', bbox_inches='tight')
    
    # ============================================
    # Figure 2: 2x2 panel layout
    # ============================================
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(6, 5.5))
    
    # Get L range for consistent colorbar
    std_cond = conditions[0]
    L_min = min(std_cond['L_pred'].min(), L_grid.min())
    L_max = max(std_cond['L_pred'].max(), L_grid.max())
    
    # Panel (0,0): Data points in (L1, L2) space, colored by L (1D latent)
    ax = axes2[0, 0]
    sc = ax.scatter(std_cond['L1'], std_cond['L2'], c=std_cond['L_pred'], 
                    cmap='viridis', s=10, alpha=0.7, vmin=L_min, vmax=L_max)
    ax.set_xlabel('$L_1$')
    ax.set_ylabel('$L_2$')
    ax.set_title('2D latent space')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel (0,1): Learned latent L on high-res grid (viridis)
    ax = axes2[0, 1]
    im = ax.pcolormesh(S1_grid, S2_grid, L_grid, cmap='viridis', shading='auto',
                        vmin=L_min, vmax=L_max)
    ax.set_xlabel('$s_1$')
    ax.set_ylabel('$s_2$')
    ax.set_title('Learned latent $L$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_aspect('equal')
    cbar = fig2.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('1D latent $L$')
    
    # Panel (1,0): Coefficient colorplot (narrow, 2 pixels)
    ax = axes2[1, 0]
    coef_img = np.array([[coefs[0]], [coefs[1]]])  # Shape (2, 1)
    im_coef = ax.imshow(coef_img, cmap='coolwarm', vmin=-1.5, vmax=1.5, aspect='auto')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['$s_1$', '$s_2$'])
    ax.set_xticks([])
    ax.set_xlabel('$w_i$')
    ax.set_title('$L \\approx \\sum_i w_i s_i$', fontsize=10)
    cbar2 = fig2.colorbar(im_coef, ax=ax, shrink=0.8)
    
    # Panel (1,1): L vs s1 for different s2 values
    ax = axes2[1, 1]
    colors_s2 = ['#1f77b4', '#2ca02c', '#d62728']
    for i, s2_val in enumerate(s2_values):
        s1_line, L_line = L_vs_s1[s2_val]
        ax.plot(s1_line, L_line, color=colors_s2[i], label=f'$s_2={s2_val}$')
    ax.set_xlabel('$s_1$')
    ax.set_ylabel('$L$')
    ax.set_title('$L$ vs $s_1$')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/fig2_latent_regression.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{plot_dir}/fig2_latent_regression.pdf', bbox_inches='tight')
    
    print(f"\nSaved plots to {plot_dir}/")
    
    # Print summary
    print("\n=== Summary ===")
    for cond in conditions:
        print(f"\n{cond['name']}:")
        print(f"  I(s1, F)   = {cond['MI_s1_fate']:.3f}")
        print(f"  I(s2, F)   = {cond['MI_s2_fate']:.3f}")
        print(f"  I(F*, F)   = {cond['MI_fate_pred']:.3f}")
        acc = np.mean(cond['fate'] == cond['fate_pred'])
        print(f"  Accuracy   = {acc:.1%}")