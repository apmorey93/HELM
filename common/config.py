"""
Configuration system for HELM experiments.
"""

from dataclasses import dataclass


@dataclass
class Exp1Config:
    """Configuration for Experiment 1: Spectral Cliff"""
    # Lorenz system parameters
    dt: float = 0.01
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    control_coupling: float = 0.1  # coupling strength for control input
    
    # Training parameters
    num_trajectories: int = 10000
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    
    # HELM-specific parameters
    lambda_spectral: float = 1e-3  # spectral regularization weight
    
    # Analysis parameters
    horizons: list = None  # [5, 10, 20, 50, 100, 200]
    num_test_samples: int = 64
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 10, 20, 50, 100, 200]


@dataclass
class Exp2Config:
    """Configuration for Experiment 2: Funnel Heatmap"""
    # Energy function parameters
    a_optimum: list = None  # [0.0, 0.0]
    Q_eigenvalues: list = None  # [1.0, 1.0] for identity
    sinusoid_freqs: list = None  # [[5.0, 0.0], [0.0, 5.0], [7.0, 7.0]]
    sinusoid_amplitudes: list = None  # [1.0, 1.0, 0.5]
    
    # Training parameters
    num_training_steps: int = 5000
    batch_size: int = 256
    learning_rate: float = 1e-3
    domain_range: float = 3.0  # sample from [-3, 3]^2
    
    # HELM-specific parameters
    lambda_hf: float = 1.0  # high-frequency invariance weight
    hf_noise_sigma: float = 0.05  # perturbation scale for HF invariance
    
    # Visualization parameters
    grid_resolution: int = 200  # for contour plots
    basin_grid_resolution: int = 50  # for basin analysis
    gd_steps_basin: int = 100  # GD steps for basin analysis
    gd_lr_basin: float = 0.1
    
    def __post_init__(self):
        if self.a_optimum is None:
            self.a_optimum = [0.0, 0.0]
        if self.Q_eigenvalues is None:
            self.Q_eigenvalues = [1.0, 1.0]
        if self.sinusoid_freqs is None:
            self.sinusoid_freqs = [[5.0, 0.0], [0.0, 5.0], [7.0, 7.0]]
        if self.sinusoid_amplitudes is None:
            self.sinusoid_amplitudes = [1.0, 1.0, 0.5]


@dataclass
class Exp3Config:
    """Configuration for Experiment 3: Planning Efficiency (MiniGrid)"""
    # Environment parameters
    env_name: str = 'MiniGrid-KeyCorridorS3R3-v0'
    max_steps_per_episode: int = 100
    
    # World model parameters
    d_latent: int = 64  # latent state dimension
    hidden_dim: int = 128
    encoder_channels: list = None  # [32, 64, 64]
    
    # Training parameters
    num_training_episodes: int = 500
    learning_rate: float = 1e-3
    
    # Planning parameters
    planning_horizons: list = None  # [10, 20, 30, 50, 80]
    
    # Planner-specific parameters
    # Flat/HELM gradient descent
    gd_steps: int = 50
    gd_lr: float = 0.1
    gumbel_tau: float = 1.0  # temperature for Gumbel-Softmax
    
    # HELM coarse-to-fine
    coarse_stride: int = 4  # N in the spec
    gd_steps_coarse: int = 20
    gd_steps_fine: int = 30
    gd_lr_coarse: float = 0.1
    gd_lr_fine: float = 0.05
    
    # CEM parameters
    cem_num_samples: int = 256
    cem_num_iters: int = 5
    cem_elite_frac: float = 0.1
    
    # HELM regularization (for world model training)
    lambda_spectral: float = 1e-3
    lambda_hf: float = 0.1
    hf_noise_sigma: float = 0.05
    
    # Evaluation parameters
    num_eval_episodes: int = 50
    num_seeds: int = 10
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64, 64]
        if self.planning_horizons is None:
            self.planning_horizons = [10, 20, 30, 50, 80]


# Device configuration
def get_device():
    """Get default device (cuda if available)"""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'
