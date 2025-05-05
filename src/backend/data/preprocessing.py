import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from nilearn import image
from .qc import calculate_qc_metrics
from .augmentation import FMRIAugmentor, AugmentedDataset
from .parallel_loader import ParallelFMRILoader
import logging
from tqdm import tqdm


def preprocess_fmri_data(
    raw_data: np.ndarray,
    tr: float = 2.0,
    high_pass: float = 0.01,
    low_pass: float = 0.1,
    smoothing_fwhm: float = 6.0,
    motion_threshold: float = 0.5,
    detrend: bool = True,
    standardize: bool = True
) -> np.ndarray:
    """
    Preprocess fMRI data with configurable parameters.
    
    Args:
        raw_data: Raw fMRI data
        tr: Repetition time in seconds
        high_pass: High-pass filter frequency in Hz
        low_pass: Low-pass filter frequency in Hz
        smoothing_fwhm: Full width at half maximum for spatial smoothing
        motion_threshold: Threshold for motion correction
        detrend: Whether to perform linear detrending
        standardize: Whether to standardize the data
    
    Returns:
        Preprocessed fMRI data
    """
    # Create frequency filter
    nyquist = 1 / (2 * tr)
    b, a = signal.butter(
        3,
        [high_pass / nyquist, low_pass / nyquist],
        btype='band'
    )
    
    # Initialize preprocessed data
    processed_data = raw_data.copy()
    
    # Apply temporal filtering
    for x in range(processed_data.shape[0]):
        for y in range(processed_data.shape[1]):
            for z in range(processed_data.shape[2]):
                timeseries = processed_data[x, y, z, :]
                if detrend:
                    timeseries = signal.detrend(timeseries)
                processed_data[x, y, z, :] = signal.filtfilt(b, a, timeseries)
    
    # Apply spatial smoothing
    if smoothing_fwhm > 0:
        for t in range(processed_data.shape[-1]):
            processed_data[..., t] = image.smooth_img(
                nib.Nifti1Image(processed_data[..., t], np.eye(4)),
                smoothing_fwhm
            ).get_fdata()
    
    # Motion correction
    if motion_threshold > 0:
        processed_data = _apply_motion_correction(
            processed_data,
            motion_threshold
        )
    
    # Standardization
    if standardize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def _apply_motion_correction(
    data: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Apply motion correction to fMRI data."""
    # Calculate frame-wise displacement
    motion = np.zeros(data.shape[-1])
    for t in range(1, data.shape[-1]):
        motion[t] = np.mean(np.abs(data[..., t] - data[..., t-1]))
    
    # Identify high-motion frames
    high_motion = motion > threshold
    
    # Interpolate high-motion frames
    corrected_data = data.copy()
    for t in range(data.shape[-1]):
        if high_motion[t]:
            if t > 0 and t < data.shape[-1] - 1:
                corrected_data[..., t] = (data[..., t-1] + data[..., t+1]) / 2
            elif t == 0:
                corrected_data[..., t] = data[..., t+1]
            else:
                corrected_data[..., t] = data[..., t-1]
    
    return corrected_data


class FMRIDataset(Dataset):
    """Dataset class for fMRI data."""
    
    def __init__(
        self,
        data_dir: Path,
        labels: Dict[str, int],
        preprocessing_params: Optional[Dict[str, Any]] = None,
        augment: bool = False,
        augment_params: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing fMRI data
            labels: Dictionary mapping subject IDs to labels
            preprocessing_params: Parameters for preprocessing
            augment: Whether to apply data augmentation
            augment_params: Parameters for augmentation
            cache_dir: Directory to cache preprocessed data
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.preprocessing_params = preprocessing_params or {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize parallel loader
        self.loader = ParallelFMRILoader(cache_dir=cache_dir)
        
        # Load all data
        self.file_paths = list(self.data_dir.glob("*.nii.gz"))
        self.subject_ids = [f.stem for f in self.file_paths]
        self.label_list = [labels[sid] for sid in self.subject_ids]
        
        # Load and preprocess data
        self.logger.info("Loading and preprocessing data...")
        self.data_loader = self.loader.create_data_loader(
            self.file_paths,
            self.label_list,
            self.preprocessing_params
        )
        
        # Set up augmentation if requested
        if augment:
            augmentor = FMRIAugmentor(**(augment_params or {}))
            self.dataset = AugmentedDataset(
                self.data_loader.dataset,
                augmentor
            )
        else:
            self.dataset = self.data_loader.dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        return self.dataset[idx]


def load_and_preprocess_fmri(
    data_dir: str,
    preprocessing_params: Optional[Dict[str, Any]] = None,
    augment: bool = False,
    augment_params: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    max_cache_size_gb: float = 50.0,
    batch_size: int = 32,
    num_workers: int = 4,
    validate_cache: bool = True
) -> DataLoader:
    """
    Load and preprocess fMRI data with optional augmentation and caching.
    
    Args:
        data_dir: Directory containing fMRI data
        preprocessing_params: Parameters for preprocessing
        augment: Whether to apply data augmentation
        augment_params: Parameters for augmentation
        cache_dir: Directory to cache preprocessed data
        max_cache_size_gb: Maximum cache size in GB
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        validate_cache: Whether to validate cache before loading
    
    Returns:
        DataLoader for the preprocessed dataset
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Initialize preprocessing parameters
    default_params = {
        'tr': 2.0,
        'high_pass': 0.01,
        'low_pass': 0.1,
        'smoothing_fwhm': 6.0,
        'motion_threshold': 0.5,
        'detrend': True,
        'standardize': True
    }
    preprocessing_params = {**default_params, **(preprocessing_params or {})}
    
    # Initialize cache directory
    if cache_dir:
        cache_dir = Path(cache_dir) / 'fmri_cache'
        logger.info(f"Using cache directory: {cache_dir}")
    
    # Create parallel loader with caching
    loader = ParallelFMRILoader(
        num_workers=num_workers,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_cache_size_gb=max_cache_size_gb
    )
    
    # Validate cache if requested
    if validate_cache and cache_dir:
        logger.info("Validating cache...")
        issues = loader.validate_cache()
        if issues:
            logger.warning(
                f"Found {len(issues)} cache issues. Consider running clean_cache()"
            )
    
    # Load labels
    logger.info("Loading subject labels...")
    labels = _load_subject_labels(data_dir)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = FMRIDataset(
        Path(data_dir),
        labels,
        preprocessing_params=preprocessing_params,
        augment=augment,
        augment_params=augment_params,
        cache_dir=cache_dir
    )
    
    # Create data loader
    logger.info("Creating data loader...")
    data_loader = loader.create_data_loader(
        dataset.file_paths,
        dataset.label_list,
        preprocessing_params
    )
    
    return data_loader


def clean_cache(cache_dir: str):
    """
    Clean the preprocessing cache.
    
    Args:
        cache_dir: Cache directory to clean
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    cache_dir = Path(cache_dir) / 'fmri_cache'
    if not cache_dir.exists():
        logger.info("No cache directory found.")
        return
    
    logger.info(f"Cleaning cache directory: {cache_dir}")
    loader = ParallelFMRILoader(cache_dir=cache_dir)
    loader.clean_cache()
    logger.info("Cache cleaned successfully")


def _load_subject_labels(data_dir: str) -> Dict[str, int]:
    # Implement your label loading logic here
    return {}
