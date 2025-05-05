import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading
from .preprocessing import preprocess_fmri_data
from .qc import calculate_qc_metrics
from .cache_manager import CacheManager
import logging
from tqdm import tqdm


class ParallelFMRILoader:
    """Parallel loader for fMRI data with preprocessing and quality control."""
    
    def __init__(
        self,
        num_workers: int = mp.cpu_count(),
        batch_size: int = 32,
        prefetch_factor: int = 2,
        cache_dir: Optional[str] = None,
        max_cache_size_gb: float = 50.0
    ):
        """
        Initialize parallel loader.
        
        Args:
            num_workers: Number of worker processes
            batch_size: Batch size for loading
            prefetch_factor: Number of batches to prefetch per worker
            cache_dir: Directory to cache preprocessed data
            max_cache_size_gb: Maximum cache size in GB
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        
        # Initialize cache manager if cache directory is provided
        self.cache_manager = CacheManager(
            cache_dir,
            max_cache_size_gb=max_cache_size_gb
        ) if cache_dir else None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def _load_single_subject(
        self,
        file_path: Path,
        preprocessing_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Load and preprocess data for a single subject."""
        # Generate a hash for the file and preprocessing params
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Try to get from cache first
        if self.cache_manager:
            data_hash = self.cache_manager._compute_hash(
                np.frombuffer(file_content, dtype=np.uint8),
                preprocessing_params
            )
            cached_data, metadata = self.cache_manager.get(data_hash)
            if cached_data is not None:
                self.logger.debug(f"Cache hit for {file_path.name}")
                return cached_data, metadata.qc_metrics
        
        # Load raw data
        img = nib.load(str(file_path))
        raw_data = img.get_fdata()
        
        # Preprocess data
        processed_data = preprocess_fmri_data(raw_data, **preprocessing_params)
        
        # Calculate QC metrics
        qc_metrics = calculate_qc_metrics(processed_data)
        
        # Cache results if cache manager is available
        if self.cache_manager:
            self.logger.debug(f"Caching data for {file_path.name}")
            self.cache_manager.put(
                processed_data,
                preprocessing_params,
                qc_metrics
            )
        
        return processed_data, qc_metrics
    
    def _load_batch(
        self,
        file_paths: List[Path],
        preprocessing_params: Dict[str, Any]
    ) -> List[Tuple[np.ndarray, Dict[str, float]]]:
        """Load and preprocess a batch of subjects in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._load_single_subject, fp, preprocessing_params)
                for fp in file_paths
            ]
            results = [future.result() for future in futures]
        return results
    
    def create_data_loader(
        self,
        file_paths: List[Path],
        labels: List[int],
        preprocessing_params: Dict[str, Any]
    ) -> DataLoader:
        """Create a DataLoader with parallel loading and preprocessing."""
        # First, load all data in parallel batches
        all_data = []
        all_qc = []
        
        total_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(
            range(0, len(file_paths), self.batch_size),
            total=total_batches,
            desc="Loading and preprocessing data"
        ):
            batch_files = file_paths[i:i + self.batch_size]
            batch_results = self._load_batch(batch_files, preprocessing_params)
            
            batch_data, batch_qc = zip(*batch_results)
            all_data.extend(batch_data)
            all_qc.extend(batch_qc)
            
            # Log progress
            self.logger.info(
                f"Processed {i + len(batch_files)}/{len(file_paths)} files"
            )
        
        # Log cache stats if available
        if self.cache_manager:
            stats = self.cache_manager.get_stats()
            self.logger.info("Cache statistics:")
            self.logger.info(f"Total entries: {stats['total_entries']}")
            self.logger.info(
                f"Cache size: {stats['total_size_bytes'] / 1024**3:.2f} GB"
            )
            self.logger.info(f"Cache usage: {stats['usage_percent']:.1f}%")
        
        # Create dataset
        dataset = ParallelFMRIDataset(all_data, all_qc, labels)
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
        
        return loader
    
    def clean_cache(self):
        """Clean the cache."""
        if self.cache_manager:
            self.logger.info("Cleaning cache...")
            self.cache_manager.clear()
            self.logger.info("Cache cleaned successfully")
    
    def validate_cache(self) -> List[str]:
        """Validate cache integrity."""
        if self.cache_manager:
            issues = self.cache_manager.validate()
            if issues:
                self.logger.warning(f"Found {len(issues)} cache issues:")
                for issue in issues:
                    self.logger.warning(f"  - {issue}")
            else:
                self.logger.info("Cache validation successful")
            return issues
        return ["Cache manager not initialized"]


class ParallelFMRIDataset(Dataset):
    """Dataset class for parallel loading of fMRI data."""
    
    def __init__(
        self,
        data: List[np.ndarray],
        qc_metrics: List[Dict[str, float]],
        labels: List[int]
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of preprocessed fMRI data
            qc_metrics: List of QC metrics for each subject
            labels: List of labels (0 for control, 1 for autism)
        """
        self.data = data
        self.qc_metrics = qc_metrics
        self.labels = labels
        
        # Convert data to tensors
        self.tensors = [torch.from_numpy(d).float() for d in data]
        self.label_tensors = torch.tensor(labels).float()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """Get a single item with its QC metrics and label."""
        return self.tensors[idx], self.qc_metrics[idx], self.label_tensors[idx]


class AsyncPreprocessor:
    """Asynchronous preprocessor for real-time data loading."""
    
    def __init__(
        self,
        preprocessing_params: Dict[str, Any],
        queue_size: int = 10
    ):
        """
        Initialize async preprocessor.
        
        Args:
            preprocessing_params: Parameters for preprocessing
            queue_size: Size of the preprocessing queue
        """
        self.preprocessing_params = preprocessing_params
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._preprocessing_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def _preprocessing_worker(self):
        """Worker function for preprocessing thread."""
        while not self.stop_event.is_set():
            try:
                # Get raw data from queue
                raw_data = self.queue.get(timeout=1)
                
                # Preprocess data
                processed_data = preprocess_fmri_data(
                    raw_data,
                    **self.preprocessing_params
                )
                
                # Calculate QC metrics
                qc_metrics = calculate_qc_metrics(processed_data)
                
                # Store results (implement storage logic here)
                self._store_results(processed_data, qc_metrics)
                
                self.queue.task_done()
            
            except queue.Empty:
                continue
    
    def _store_results(
        self,
        processed_data: np.ndarray,
        qc_metrics: Dict[str, float]
    ):
        """Store preprocessing results."""
        # Implement storage logic (e.g., to disk or memory)
        pass
    
    def submit(self, raw_data: np.ndarray):
        """Submit raw data for preprocessing."""
        self.queue.put(raw_data)
    
    def stop(self):
        """Stop the preprocessing worker."""
        self.stop_event.set()
        self.worker_thread.join()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
