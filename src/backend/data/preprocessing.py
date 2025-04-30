import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_and_preprocess_fmri(fmri_path):
    """Load and preprocess fMRI data."""
    # Load NIfTI file
    img = nib.load(fmri_path)
    data = img.get_fdata()
    
    # Basic preprocessing
    # 1. Standardize the time series for each voxel
    data = StandardScaler().fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    
    return data


def create_connectivity_matrix(fmri_data, atlas_labels):
    """Create connectivity matrix from fMRI data using atlas regions."""
    num_regions = len(np.unique(atlas_labels))
    connectivity_matrix = np.zeros((num_regions, num_regions))
    
    # Extract time series for each region
    region_time_series = []
    for i in range(num_regions):
        region_mask = atlas_labels == i
        region_mean = fmri_data[region_mask].mean(axis=0)
        region_time_series.append(region_mean)
    
    # Calculate correlations between regions
    for i in range(num_regions):
        for j in range(num_regions):
            correlation = np.corrcoef(region_time_series[i], region_time_series[j])[0, 1]
            connectivity_matrix[i, j] = correlation
    
    # Ensure the matrix is symmetric
    connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
    
    return connectivity_matrix


class FMRIDataset(Dataset):
    """Dataset class for fMRI connectivity matrices."""
    def __init__(self, connectivity_matrices, labels):
        self.connectivity_matrices = torch.FloatTensor(connectivity_matrices)
        self.labels = torch.FloatTensor(labels)
        
        # Create node features (using degree centrality as initial features)
        self.node_features = torch.sum(self.connectivity_matrices, dim=2)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.node_features[idx],
            self.connectivity_matrices[idx],
            self.labels[idx].unsqueeze(0)
        )


def prepare_data_loaders(connectivity_matrices, labels, batch_size=32, train_split=0.8):
    """Prepare train and validation data loaders."""
    # Split indices
    num_samples = len(labels)
    indices = np.random.permutation(num_samples)
    split_idx = int(train_split * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create datasets
    train_dataset = FMRIDataset(
        connectivity_matrices[train_indices],
        labels[train_indices]
    )
    val_dataset = FMRIDataset(
        connectivity_matrices[val_indices],
        labels[val_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader
