import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from nilearn import image
from nilearn.masking import compute_epi_mask
import warnings
from .qc import generate_qc_report


def load_and_preprocess_fmri(
    fmri_path,
    tr=2.0,
    high_pass=0.01,
    low_pass=0.1,
    motion_threshold=0.5,
    spike_thresh=5.0,
    smooth_fwhm=6.0
):
    """Load and preprocess fMRI data.
    
    Args:
        fmri_path (str): Path to the NIfTI file
        tr (float): Repetition time in seconds
        high_pass (float): High-pass filter frequency in Hz
        low_pass (float): Low-pass filter frequency in Hz
        motion_threshold (float): Threshold for motion correction in mm
        spike_thresh (float): Z-score threshold for spike detection
        smooth_fwhm (float): FWHM for spatial smoothing in mm
    
    Returns:
        tuple: (preprocessed_data, qc_report)
            - preprocessed_data (np.ndarray): Preprocessed fMRI data
            - qc_report (dict): Comprehensive quality control metrics
    """
    # Load NIfTI file
    img = nib.load(fmri_path)
    
    # Generate initial QC report
    initial_qc = generate_qc_report(img)
    
    # Create brain mask
    mask_img = compute_epi_mask(img)
    mask = mask_img.get_fdata()
    
    # Motion correction with configurable threshold
    motion_corrected = image.clean_img(
        img,
        detrend=True,
        standardize=True,
        confounds=None,
        t_r=tr,
        high_pass=high_pass,
        low_pass=low_pass,
        motion_threshold=motion_threshold
    )
    
    # Get motion parameters
    motion_params = motion_corrected.header.get_qform()
    motion_params = motion_params[:3, :].flatten()
    
    # Apply spatial smoothing
    smoothed = image.smooth_img(
        motion_corrected,
        fwhm=smooth_fwhm
    )
    data = smoothed.get_fdata()
    
    # Temporal filtering
    nyquist = 1 / (2 * tr)
    b, a = signal.butter(3, [high_pass/nyquist, low_pass/nyquist], btype='band')
    
    # Apply filtering to each voxel's time series
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if mask[i, j, k]:
                    ts = data[i, j, k]
                    # Spike detection and interpolation
                    z_scores = (ts - np.mean(ts)) / np.std(ts)
                    spikes = np.abs(z_scores) > spike_thresh
                    if np.any(spikes):
                        ts[spikes] = np.interp(
                            np.where(spikes)[0],
                            np.where(~spikes)[0],
                            ts[~spikes]
                        )
                    filtered_data[i, j, k] = signal.filtfilt(b, a, ts)
    
    # Final standardization
    filtered_data = StandardScaler().fit_transform(
        filtered_data.reshape(-1, filtered_data.shape[-1])
    ).reshape(filtered_data.shape)
    
    # Generate final QC report
    final_qc = generate_qc_report(
        nib.Nifti1Image(filtered_data, img.affine),
        motion_params
    )
    
    # Combine QC reports with parameter information
    qc_report = {
        'pre_processing': initial_qc,
        'post_processing': final_qc,
        'processing_impact': {
            'tsnr_improvement': (
                final_qc['signal_metrics']['temporal_snr'] -
                initial_qc['signal_metrics']['temporal_snr']
            ),
            'artifact_reduction': (
                initial_qc['artifact_metrics']['ghost_to_signal_ratio'] -
                final_qc['artifact_metrics']['ghost_to_signal_ratio']
            ),
            'quality_change': (
                final_qc['quality_summary']['overall_quality'] !=
                initial_qc['quality_summary']['overall_quality']
            )
        },
        'parameters_used': {
            'high_pass': high_pass,
            'low_pass': low_pass,
            'motion_threshold': motion_threshold,
            'spike_thresh': spike_thresh,
            'smooth_fwhm': smooth_fwhm
        }
    }
    
    return filtered_data, qc_report


def optimize_and_preprocess(fmri_path: str, n_iterations: int = 20):
    """Optimize parameters and preprocess fMRI data.
    
    Args:
        fmri_path (str): Path to the fMRI data
        n_iterations (int): Number of optimization iterations
    
    Returns:
        tuple: (preprocessed_data, qc_report, optimization_history)
    """
    from .param_optimizer import optimize_preprocessing_parameters
    
    # Run parameter optimization
    best_params, history = optimize_preprocessing_parameters(
        fmri_path,
        n_iterations=n_iterations
    )
    
    # Preprocess with optimized parameters
    preprocessed_data, qc_report = load_and_preprocess_fmri(
        fmri_path,
        **best_params
    )
    
    # Add optimization history to QC report
    qc_report['optimization_history'] = history
    
    return preprocessed_data, qc_report, history


def calculate_framewise_displacement(img):
    """Calculate framewise displacement for motion assessment."""
    motion_params = image.load_img(img).header.get_zooms()[:3]  # Get voxel dimensions
    fd = np.abs(np.diff(motion_params, axis=0))
    return np.mean(fd)


def create_connectivity_matrix(fmri_data, atlas_labels, atlas_type='aal'):
    """Create connectivity matrix from fMRI data using atlas regions.
    
    Args:
        fmri_data (np.ndarray): Preprocessed fMRI data
        atlas_labels (np.ndarray): Atlas labels for each voxel
        atlas_type (str): Type of atlas being used ('aal', 'harvard_oxford', etc.)
    """
    num_regions = len(np.unique(atlas_labels))
    connectivity_matrix = np.zeros((num_regions, num_regions))
    
    # Extract time series for each region
    region_time_series = []
    for i in range(num_regions):
        region_mask = atlas_labels == i
        if np.sum(region_mask) == 0:
            warnings.warn(f"Empty region found in {atlas_type} atlas at index {i}")
            region_mean = np.zeros(fmri_data.shape[-1])
        else:
            region_mean = fmri_data[region_mask].mean(axis=0)
        region_time_series.append(region_mean)
    
    # Calculate correlations between regions
    for i in range(num_regions):
        for j in range(i, num_regions):  # Optimize by using symmetry
            correlation = np.corrcoef(region_time_series[i], region_time_series[j])[0, 1]
            connectivity_matrix[i, j] = correlation
            connectivity_matrix[j, i] = correlation  # Matrix is symmetric
    
    # Apply Fisher z-transformation to normalize correlation values
    connectivity_matrix = np.arctanh(connectivity_matrix)
    
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
