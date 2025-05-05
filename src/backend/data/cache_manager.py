import os
import json
import hashlib
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import shutil
import logging
import lz4.frame
from datetime import datetime
import fcntl
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class CacheMetadata:
    """Metadata for cached items."""
    version: str
    created_at: str
    preprocessing_params: Dict[str, Any]
    data_hash: str
    qc_metrics: Dict[str, float]
    compression: str
    size_bytes: int


class CacheManager:
    """Manages caching of preprocessed fMRI data with versioning and compression."""
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        cache_dir: str,
        max_cache_size_gb: float = 50.0,
        compression_level: int = 3,
        max_threads: int = 4
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_cache_size_gb: Maximum cache size in GB
            compression_level: LZ4 compression level (1-9)
            max_threads: Maximum number of threads for parallel operations
        """
        self.cache_dir = Path(cache_dir)
        self.data_dir = self.cache_dir / 'data'
        self.metadata_dir = self.cache_dir / 'metadata'
        self.lock_file = self.cache_dir / 'cache.lock'
        
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.compression_level = compression_level
        self.max_threads = max_threads
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize cache structure
        self._init_cache_dirs()
    
    def _init_cache_dirs(self):
        """Initialize cache directory structure."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Create lock file if it doesn't exist
        self.lock_file.touch(exist_ok=True)
        
        # Load cache index
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / 'index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {
                'version': self.VERSION,
                'entries': {},
                'size_bytes': 0
            }
            self._save_cache_index()
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        index_path = self.cache_dir / 'index.json'
        with self._file_lock():
            with open(index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
    
    def _file_lock(self):
        """Context manager for file locking."""
        class FileLock:
            def __init__(self, lock_file):
                self.lock_file = lock_file
                self.lock_fd = None
            
            def __enter__(self):
                self.lock_fd = open(self.lock_file, 'r')
                while True:
                    try:
                        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.1)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.lock_fd:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                    self.lock_fd.close()
        
        return FileLock(self.lock_file)
    
    def _compute_hash(self, data: np.ndarray, params: Dict[str, Any]) -> str:
        """Compute hash for data and parameters."""
        # Hash the data
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()
        
        # Hash the parameters
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()
        
        # Combine hashes
        combined = f"{data_hash}:{param_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _compress_data(self, data: np.ndarray) -> bytes:
        """Compress data using LZ4."""
        return lz4.frame.compress(
            data.tobytes(),
            compression_level=self.compression_level
        )
    
    def _decompress_data(self, compressed_data: bytes, shape: Tuple, dtype: str) -> np.ndarray:
        """Decompress data from LZ4."""
        decompressed = lz4.frame.decompress(compressed_data)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    
    def _enforce_cache_size(self):
        """Enforce maximum cache size by removing old entries."""
        with self._file_lock():
            current_size = self.cache_index['size_bytes']
            
            if current_size <= self.max_cache_size:
                return
            
            # Sort entries by last access time
            entries = list(self.cache_index['entries'].items())
            entries.sort(key=lambda x: x[1]['last_accessed'])
            
            # Remove entries until we're under the limit
            for key, entry in entries:
                if current_size <= self.max_cache_size:
                    break
                
                # Remove data and metadata files
                data_path = self.data_dir / f"{key}.npz"
                meta_path = self.metadata_dir / f"{key}.json"
                
                if data_path.exists():
                    current_size -= data_path.stat().st_size
                    data_path.unlink()
                if meta_path.exists():
                    current_size -= meta_path.stat().st_size
                    meta_path.unlink()
                
                del self.cache_index['entries'][key]
            
            self.cache_index['size_bytes'] = current_size
            self._save_cache_index()
    
    def get(
        self,
        data_hash: str
    ) -> Tuple[Optional[np.ndarray], Optional[CacheMetadata]]:
        """
        Retrieve data from cache.
        
        Args:
            data_hash: Hash of the data to retrieve
        
        Returns:
            Tuple of (data, metadata) if found, (None, None) if not found
        """
        if data_hash not in self.cache_index['entries']:
            return None, None
        
        # Update access time
        with self._file_lock():
            self.cache_index['entries'][data_hash]['last_accessed'] = \
                datetime.now().isoformat()
            self._save_cache_index()
        
        # Load metadata
        meta_path = self.metadata_dir / f"{data_hash}.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Load data
        data_path = self.data_dir / f"{data_hash}.npz"
        with open(data_path, 'rb') as f:
            compressed_data = f.read()
        
        # Decompress data
        data = self._decompress_data(
            compressed_data,
            tuple(metadata['shape']),
            metadata['dtype']
        )
        
        return data, CacheMetadata(**metadata)
    
    def put(
        self,
        data: np.ndarray,
        preprocessing_params: Dict[str, Any],
        qc_metrics: Dict[str, float]
    ) -> str:
        """
        Store data in cache.
        
        Args:
            data: Data to cache
            preprocessing_params: Parameters used for preprocessing
            qc_metrics: Quality control metrics
        
        Returns:
            Hash of the cached data
        """
        # Compute hash
        data_hash = self._compute_hash(data, preprocessing_params)
        
        # Check if already cached
        if data_hash in self.cache_index['entries']:
            return data_hash
        
        # Compress data
        compressed_data = self._compress_data(data)
        
        # Create metadata
        metadata = {
            'version': self.VERSION,
            'created_at': datetime.now().isoformat(),
            'preprocessing_params': preprocessing_params,
            'data_hash': data_hash,
            'qc_metrics': qc_metrics,
            'compression': 'lz4',
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'size_bytes': len(compressed_data)
        }
        
        # Save data and metadata
        with self._file_lock():
            # Write data
            data_path = self.data_dir / f"{data_hash}.npz"
            with open(data_path, 'wb') as f:
                f.write(compressed_data)
            
            # Write metadata
            meta_path = self.metadata_dir / f"{data_hash}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update index
            self.cache_index['entries'][data_hash] = {
                'created_at': metadata['created_at'],
                'last_accessed': metadata['created_at'],
                'size_bytes': metadata['size_bytes']
            }
            self.cache_index['size_bytes'] += metadata['size_bytes']
            
            self._save_cache_index()
        
        # Enforce cache size limits
        self._enforce_cache_size()
        
        return data_hash
    
    def clear(self):
        """Clear all cached data."""
        with self._file_lock():
            # Remove all files
            shutil.rmtree(self.data_dir)
            shutil.rmtree(self.metadata_dir)
            
            # Recreate directories
            self._init_cache_dirs()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'version': self.VERSION,
            'total_entries': len(self.cache_index['entries']),
            'total_size_bytes': self.cache_index['size_bytes'],
            'max_size_bytes': self.max_cache_size,
            'usage_percent': (self.cache_index['size_bytes'] / self.max_cache_size) * 100
        }
        return stats
    
    def validate(self) -> List[str]:
        """Validate cache integrity."""
        issues = []
        
        with self._file_lock():
            # Check for missing files
            for key in self.cache_index['entries']:
                data_path = self.data_dir / f"{key}.npz"
                meta_path = self.metadata_dir / f"{key}.json"
                
                if not data_path.exists():
                    issues.append(f"Missing data file for {key}")
                if not meta_path.exists():
                    issues.append(f"Missing metadata file for {key}")
            
            # Check for orphaned files
            for file_path in self.data_dir.glob("*.npz"):
                key = file_path.stem
                if key not in self.cache_index['entries']:
                    issues.append(f"Orphaned data file: {key}")
            
            for file_path in self.metadata_dir.glob("*.json"):
                key = file_path.stem
                if key not in self.cache_index['entries']:
                    issues.append(f"Orphaned metadata file: {key}")
        
        return issues
