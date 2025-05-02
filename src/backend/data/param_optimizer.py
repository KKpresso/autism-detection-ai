"""Parameter optimization for fMRI preprocessing using Bayesian optimization.

This module implements automatic parameter tuning for the preprocessing pipeline
using Bayesian optimization with Gaussian Processes.
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from concurrent.futures import ProcessPoolExecutor
import warnings
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from .preprocessing import load_and_preprocess_fmri


@dataclass
class ParameterSpace:
    """Defines the search space for preprocessing parameters."""
    name: str
    low: float
    high: float
    is_log: bool = False  # Whether to search in log space


class PreprocessingOptimizer:
    """Optimizer for fMRI preprocessing parameters using Bayesian optimization."""
    
    def __init__(
        self,
        fmri_path: str,
        n_iterations: int = 20,
        n_initial_points: int = 5,
        n_jobs: int = -1
    ):
        """Initialize the optimizer.
        
        Args:
            fmri_path: Path to the fMRI data
            n_iterations: Number of optimization iterations
            n_initial_points: Number of initial random points
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.fmri_path = fmri_path
        self.n_iterations = n_iterations
        self.n_initial_points = n_initial_points
        self.n_jobs = n_jobs
        
        # Define parameter space
        self.param_space = [
            ParameterSpace('high_pass', 0.001, 0.1, True),    # High-pass frequency (Hz)
            ParameterSpace('low_pass', 0.05, 0.5, True),      # Low-pass frequency (Hz)
            ParameterSpace('motion_threshold', 0.1, 2.0),     # Motion threshold (mm)
            ParameterSpace('spike_thresh', 3.0, 7.0),         # Z-score threshold for spike detection
            ParameterSpace('smooth_fwhm', 4.0, 8.0),         # Spatial smoothing FWHM (mm)
        ]
        
        # Initialize Gaussian Process
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42
        )
        
        self.X = []  # Parameter combinations
        self.y = []  # Corresponding scores
    
    def _evaluate_parameters(self, params: Dict[str, float]) -> float:
        """Evaluate a parameter combination using QC metrics.
        
        Args:
            params: Dictionary of parameter values
        
        Returns:
            float: Quality score (higher is better)
        """
        try:
            # Run preprocessing with current parameters
            _, qc_report = load_and_preprocess_fmri(
                self.fmri_path,
                high_pass=params['high_pass'],
                low_pass=params['low_pass']
            )
            
            # Extract relevant metrics
            signal_metrics = qc_report['post_processing']['signal_metrics']
            artifact_metrics = qc_report['post_processing']['artifact_metrics']
            motion_metrics = qc_report['post_processing']['motion_metrics']
            
            # Calculate weighted quality score
            score = (
                0.4 * signal_metrics['temporal_snr'] / 100 +  # Normalize TSNR
                0.2 * (1 - artifact_metrics['ghost_to_signal_ratio']) +
                0.2 * (1 - motion_metrics['percent_motion_outliers'] / 100) +
                0.1 * (1 - artifact_metrics['drift_percent'] / 100) +
                0.1 * (1 - len(qc_report['post_processing']['quality_summary']['issues_detected']) / 10)
            )
            
            return max(0, min(1, score))  # Clip to [0, 1]
            
        except Exception as e:
            warnings.warn(f"Parameter evaluation failed: {str(e)}")
            return 0.0
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function.
        
        Args:
            X: Points to evaluate
        
        Returns:
            Expected improvement values
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X, return_std=True)
        
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        best_y = np.max(self.y) if self.y else 0
        
        with np.errstate(divide='ignore'):
            imp = mu - best_y
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def _get_next_point(self) -> Dict[str, float]:
        """Get next point to evaluate using acquisition function."""
        n_params = len(self.param_space)
        
        # Generate random candidates
        n_candidates = 1000
        candidates = np.zeros((n_candidates, n_params))
        
        for i, param in enumerate(self.param_space):
            if param.is_log:
                candidates[:, i] = np.exp(np.random.uniform(
                    np.log(param.low),
                    np.log(param.high),
                    n_candidates
                ))
            else:
                candidates[:, i] = np.random.uniform(
                    param.low,
                    param.high,
                    n_candidates
                )
        
        # Calculate acquisition function values
        ei_values = self._acquisition_function(candidates)
        
        # Select best point
        best_idx = np.argmax(ei_values)
        best_candidate = candidates[best_idx]
        
        # Convert to dictionary
        return {
            param.name: best_candidate[i]
            for i, param in enumerate(self.param_space)
        }
    
    def optimize(self) -> Tuple[Dict[str, float], float]:
        """Run the optimization process.
        
        Returns:
            Tuple containing:
            - Best parameters found
            - Best score achieved
        """
        # Initial random points
        for _ in range(self.n_initial_points):
            params = {
                param.name: (
                    np.exp(np.random.uniform(np.log(param.low), np.log(param.high)))
                    if param.is_log
                    else np.random.uniform(param.low, param.high)
                )
                for param in self.param_space
            }
            
            score = self._evaluate_parameters(params)
            self.X.append([params[param.name] for param in self.param_space])
            self.y.append(score)
        
        # Main optimization loop
        for _ in range(self.n_iterations - self.n_initial_points):
            # Fit GP to current data
            self.gp.fit(np.array(self.X), np.array(self.y))
            
            # Get next point to evaluate
            next_params = self._get_next_point()
            
            # Evaluate point
            score = self._evaluate_parameters(next_params)
            self.X.append([next_params[param.name] for param in self.param_space])
            self.y.append(score)
        
        # Find best parameters
        best_idx = np.argmax(self.y)
        best_params = {
            param.name: self.X[best_idx][i]
            for i, param in enumerate(self.param_space)
        }
        
        return best_params, self.y[best_idx]
    
    def get_optimization_history(self) -> Dict:
        """Get the history of the optimization process.
        
        Returns:
            Dictionary containing optimization history
        """
        return {
            'parameters': [
                {param.name: x[i] for i, param in enumerate(self.param_space)}
                for x in self.X
            ],
            'scores': self.y,
            'best_score_evolution': [
                max(self.y[:i+1]) for i in range(len(self.y))
            ]
        }


def optimize_preprocessing_parameters(
    fmri_path: str,
    n_iterations: int = 20,
    n_initial_points: int = 5
) -> Tuple[Dict[str, float], Dict]:
    """Convenience function to optimize preprocessing parameters.
    
    Args:
        fmri_path: Path to the fMRI data
        n_iterations: Number of optimization iterations
        n_initial_points: Number of initial random points
    
    Returns:
        Tuple containing:
        - Best parameters found
        - Optimization history
    """
    optimizer = PreprocessingOptimizer(
        fmri_path=fmri_path,
        n_iterations=n_iterations,
        n_initial_points=n_initial_points
    )
    
    best_params, best_score = optimizer.optimize()
    history = optimizer.get_optimization_history()
    
    print(f"\nOptimization completed:")
    print(f"Best score achieved: {best_score:.4f}")
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value:.4f}")
    
    return best_params, history
