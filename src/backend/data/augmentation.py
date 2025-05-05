import numpy as np
from scipy.ndimage import rotate, zoom
import torch
from typing import Tuple, List, Optional
import random
from scipy.interpolate import interp1d


class FMRIAugmentor:
    """Class for performing data augmentation on fMRI data."""
    
    def __init__(
        self,
        temporal_noise_std: float = 0.01,
        spatial_noise_std: float = 0.01,
        rotation_range: Tuple[float, float] = (-5, 5),
        scaling_range: Tuple[float, float] = (0.95, 1.05),
        dropout_prob: float = 0.1,
        time_warping_sigma: float = 0.2
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            temporal_noise_std: Standard deviation for temporal noise
            spatial_noise_std: Standard deviation for spatial noise
            rotation_range: Range for random rotation in degrees
            scaling_range: Range for random scaling
            dropout_prob: Probability of dropping out a region
            time_warping_sigma: Sigma for temporal warping
        """
        self.temporal_noise_std = temporal_noise_std
        self.spatial_noise_std = spatial_noise_std
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.dropout_prob = dropout_prob
        self.time_warping_sigma = time_warping_sigma
    
    def add_temporal_noise(self, data: np.ndarray) -> np.ndarray:
        """Add temporal noise to the time series."""
        noise = np.random.normal(0, self.temporal_noise_std, data.shape)
        return data + noise
    
    def add_spatial_noise(self, data: np.ndarray) -> np.ndarray:
        """Add spatial noise to the connectivity patterns."""
        noise = np.random.normal(0, self.spatial_noise_std, data.shape)
        return data + noise
    
    def rotate_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        """Apply random rotation to connectivity matrix."""
        angle = np.random.uniform(*self.rotation_range)
        rotated = rotate(connectivity, angle, axes=(0, 1), reshape=False)
        return np.clip(rotated, -1, 1)  # Maintain valid correlation values
    
    def scale_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        """Apply random scaling to connectivity matrix."""
        scale = np.random.uniform(*self.scaling_range)
        return np.clip(connectivity * scale, -1, 1)
    
    def region_dropout(self, data: np.ndarray) -> np.ndarray:
        """Randomly drop out brain regions."""
        mask = np.random.binomial(1, 1-self.dropout_prob, data.shape[1])
        return data * mask
    
    def time_warping(self, data: np.ndarray) -> np.ndarray:
        """Apply random temporal warping."""
        time_steps = np.arange(data.shape[0])
        
        # Generate random warping function
        warped_time = time_steps + np.random.normal(
            0, self.time_warping_sigma, time_steps.shape
        )
        warped_time = np.sort(warped_time)  # Ensure monotonicity
        
        # Interpolate
        warped_data = np.zeros_like(data)
        for region in range(data.shape[1]):
            f = interp1d(time_steps, data[:, region], kind='cubic', fill_value='extrapolate')
            warped_data[:, region] = f(warped_time)
        
        return warped_data
    
    def mix_regions(
        self,
        data: np.ndarray,
        mix_ratio: float = 0.2,
        n_regions: Optional[int] = None
    ) -> np.ndarray:
        """Mix signals between random pairs of regions."""
        if n_regions is None:
            n_regions = max(2, int(data.shape[1] * mix_ratio))
        
        mixed_data = data.copy()
        regions = np.random.choice(data.shape[1], n_regions, replace=False)
        
        for i in range(0, len(regions), 2):
            if i + 1 < len(regions):
                r1, r2 = regions[i], regions[i + 1]
                ratio = np.random.uniform(0.2, 0.8)
                mixed_data[:, r1] = ratio * data[:, r1] + (1 - ratio) * data[:, r2]
                mixed_data[:, r2] = (1 - ratio) * data[:, r1] + ratio * data[:, r2]
        
        return mixed_data
    
    def augment(
        self,
        data: np.ndarray,
        connectivity: np.ndarray,
        augmentation_strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all augmentations with controlled strength."""
        # Adjust parameters based on augmentation strength
        temp_noise_std = self.temporal_noise_std * augmentation_strength
        spat_noise_std = self.spatial_noise_std * augmentation_strength
        dropout_prob = self.dropout_prob * augmentation_strength
        
        # Apply augmentations to time series data
        aug_data = data.copy()
        
        if np.random.random() < 0.5:
            aug_data = self.add_temporal_noise(aug_data)
        if np.random.random() < 0.5:
            aug_data = self.time_warping(aug_data)
        if np.random.random() < 0.3:
            aug_data = self.region_dropout(aug_data)
        if np.random.random() < 0.3:
            aug_data = self.mix_regions(aug_data)
        
        # Apply augmentations to connectivity matrix
        aug_connectivity = connectivity.copy()
        
        if np.random.random() < 0.5:
            aug_connectivity = self.add_spatial_noise(aug_connectivity)
        if np.random.random() < 0.3:
            aug_connectivity = self.rotate_connectivity(aug_connectivity)
        if np.random.random() < 0.3:
            aug_connectivity = self.scale_connectivity(aug_connectivity)
        
        return aug_data, aug_connectivity


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies augmentation on the fly."""
    
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        augmentor: FMRIAugmentor,
        augment_prob: float = 0.5,
        augmentation_strength: float = 1.0
    ):
        """
        Initialize augmented dataset.
        
        Args:
            base_dataset: Original dataset
            augmentor: FMRIAugmentor instance
            augment_prob: Probability of applying augmentation
            augmentation_strength: Strength of augmentation (0 to 1)
        """
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.augment_prob = augment_prob
        self.augmentation_strength = augmentation_strength
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, connectivity, label = self.base_dataset[idx]
        
        # Convert to numpy for augmentation
        data_np = data.numpy()
        connectivity_np = connectivity.numpy()
        
        # Apply augmentation with probability
        if np.random.random() < self.augment_prob:
            data_np, connectivity_np = self.augmentor.augment(
                data_np,
                connectivity_np,
                self.augmentation_strength
            )
        
        # Convert back to torch tensors
        return (
            torch.from_numpy(data_np).float(),
            torch.from_numpy(connectivity_np).float(),
            label
        )
