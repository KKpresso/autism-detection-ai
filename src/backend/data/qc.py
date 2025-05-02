"""Quality Control metrics for fMRI preprocessing.

This module provides comprehensive QC metrics for fMRI data analysis,
including motion parameters, signal quality, and artifact detection.
"""

import numpy as np
from scipy import stats
from nilearn.image import mean_img, load_img
import nibabel as nib
from sklearn.covariance import MinCovDet


def calculate_motion_metrics(motion_params):
    """Calculate comprehensive motion-related QC metrics.
    
    Args:
        motion_params: 6-parameter motion time series (3 translation, 3 rotation)
    
    Returns:
        dict: Motion-related QC metrics
    """
    # Calculate framewise displacement
    fd = np.abs(np.diff(motion_params, axis=0))
    mean_fd = np.mean(fd)
    max_fd = np.max(fd)
    
    # Calculate motion outliers (>0.5mm displacement)
    num_outliers = np.sum(fd > 0.5)
    percent_outliers = (num_outliers / len(fd)) * 100
    
    # Calculate absolute and relative motion
    abs_motion = np.max(np.abs(motion_params), axis=0)
    rel_motion = np.max(np.abs(np.diff(motion_params, axis=0)), axis=0)
    
    return {
        'mean_framewise_displacement': mean_fd,
        'max_framewise_displacement': max_fd,
        'num_motion_outliers': num_outliers,
        'percent_motion_outliers': percent_outliers,
        'max_absolute_motion': abs_motion.tolist(),
        'max_relative_motion': rel_motion.tolist()
    }


def calculate_signal_metrics(fmri_data, mask=None):
    """Calculate signal quality metrics for fMRI data.
    
    Args:
        fmri_data: 4D fMRI data array (x, y, z, time)
        mask: Brain mask (optional)
    
    Returns:
        dict: Signal quality metrics
    """
    if mask is None:
        mask = np.ones(fmri_data.shape[:3], dtype=bool)
    
    # Calculate temporal SNR
    mean_signal = np.mean(fmri_data, axis=-1)
    std_signal = np.std(fmri_data, axis=-1)
    tsnr = np.mean(mean_signal[mask] / (std_signal[mask] + 1e-6))
    
    # Calculate DVARS (temporal derivative of timecourses)
    squared_diff = np.diff(fmri_data, axis=-1) ** 2
    dvars = np.sqrt(np.mean(squared_diff[mask], axis=0))
    
    # Calculate outlier volumes using Robust Covariance
    voxel_ts = fmri_data[mask].T
    try:
        robust_cov = MinCovDet(random_state=42).fit(voxel_ts)
        mahal_dist = robust_cov.mahalanobis(voxel_ts)
        outlier_volumes = np.sum(stats.chi2.sf(mahal_dist, df=voxel_ts.shape[1]) < 0.01)
    except:
        outlier_volumes = 0
    
    return {
        'temporal_snr': tsnr,
        'mean_dvars': np.mean(dvars),
        'max_dvars': np.max(dvars),
        'outlier_volumes': outlier_volumes,
        'percent_outlier_volumes': (outlier_volumes / fmri_data.shape[-1]) * 100
    }


def calculate_artifact_metrics(fmri_data, mask=None):
    """Calculate metrics related to artifacts in fMRI data.
    
    Args:
        fmri_data: 4D fMRI data array (x, y, z, time)
        mask: Brain mask (optional)
    
    Returns:
        dict: Artifact-related metrics
    """
    if mask is None:
        mask = np.ones(fmri_data.shape[:3], dtype=bool)
    
    # Calculate ghost-to-signal ratio
    ghost_ratio = calculate_ghost_ratio(fmri_data)
    
    # Calculate spike statistics
    spike_stats = detect_spikes(fmri_data[mask])
    
    # Calculate drift metrics
    drift_metrics = calculate_drift(fmri_data[mask])
    
    return {
        'ghost_to_signal_ratio': ghost_ratio,
        'num_spikes': spike_stats['num_spikes'],
        'max_spike_amplitude': spike_stats['max_amplitude'],
        'drift_percent': drift_metrics['drift_percent'],
        'drift_rate': drift_metrics['drift_rate']
    }


def calculate_ghost_ratio(fmri_data):
    """Calculate the ghost-to-signal ratio in fMRI data."""
    # Calculate in central slice
    central_slice = fmri_data.shape[2] // 2
    slice_data = np.mean(fmri_data[:, :, central_slice, :], axis=-1)
    
    # Define ghost region (shifted by N/2 in phase-encode direction)
    ghost_region = np.roll(slice_data, slice_data.shape[0] // 2, axis=0)
    
    # Calculate ratio
    signal = np.mean(np.abs(slice_data))
    ghost = np.mean(np.abs(ghost_region))
    
    return ghost / (signal + 1e-6)


def detect_spikes(time_series):
    """Detect signal spikes in fMRI time series."""
    # Z-score the time series
    z_scores = stats.zscore(time_series, axis=-1)
    
    # Detect spikes (|Z| > 5)
    spikes = np.abs(z_scores) > 5
    num_spikes = np.sum(np.any(spikes, axis=0))
    max_amplitude = np.max(np.abs(z_scores))
    
    return {
        'num_spikes': num_spikes,
        'max_amplitude': max_amplitude
    }


def calculate_drift(time_series):
    """Calculate signal drift metrics."""
    # Fit linear trend to mean signal
    mean_signal = np.mean(time_series, axis=0)
    timepoints = np.arange(len(mean_signal))
    slope, _, _, _, _ = stats.linregress(timepoints, mean_signal)
    
    # Calculate drift percentage
    signal_range = np.ptp(mean_signal)
    total_drift = slope * len(mean_signal)
    drift_percent = (total_drift / signal_range) * 100
    
    return {
        'drift_percent': drift_percent,
        'drift_rate': slope
    }


def generate_qc_report(fmri_img, motion_params=None):
    """Generate a comprehensive QC report for fMRI data.
    
    Args:
        fmri_img: Path to fMRI NIfTI file or nibabel image object
        motion_params: Motion parameters if available (optional)
    
    Returns:
        dict: Comprehensive QC metrics
    """
    # Load data
    if isinstance(fmri_img, str):
        fmri_img = nib.load(fmri_img)
    fmri_data = fmri_img.get_fdata()
    
    # Create brain mask
    mean_data = np.mean(fmri_data, axis=-1)
    mask = mean_data > np.percentile(mean_data, 10)
    
    # Calculate all metrics
    signal_metrics = calculate_signal_metrics(fmri_data, mask)
    artifact_metrics = calculate_artifact_metrics(fmri_data, mask)
    
    # Add motion metrics if available
    motion_metrics = {}
    if motion_params is not None:
        motion_metrics = calculate_motion_metrics(motion_params)
    
    # Combine all metrics
    qc_report = {
        'signal_metrics': signal_metrics,
        'artifact_metrics': artifact_metrics,
        'motion_metrics': motion_metrics,
        'data_info': {
            'dimensions': fmri_data.shape,
            'voxel_size': fmri_img.header.get_zooms(),
            'num_timepoints': fmri_data.shape[-1],
            'percent_brain_voxels': (np.sum(mask) / mask.size) * 100
        }
    }
    
    # Add overall quality assessment
    qc_report['quality_summary'] = assess_overall_quality(qc_report)
    
    return qc_report


def assess_overall_quality(qc_report):
    """Assess overall data quality based on QC metrics."""
    issues = []
    severity = 'good'
    
    # Check signal quality
    if qc_report['signal_metrics']['temporal_snr'] < 50:
        issues.append('Low temporal SNR')
        severity = 'poor' if severity == 'good' else severity
    
    # Check motion
    if 'motion_metrics' in qc_report and qc_report['motion_metrics']:
        if qc_report['motion_metrics']['mean_framewise_displacement'] > 0.5:
            issues.append('High mean framewise displacement')
            severity = 'poor'
        if qc_report['motion_metrics']['percent_motion_outliers'] > 20:
            issues.append('High percentage of motion outliers')
            severity = 'poor'
    
    # Check artifacts
    if qc_report['artifact_metrics']['ghost_to_signal_ratio'] > 0.1:
        issues.append('High ghost-to-signal ratio')
        severity = 'poor' if severity == 'good' else severity
    
    if qc_report['artifact_metrics']['num_spikes'] > 5:
        issues.append('Multiple signal spikes detected')
        severity = 'poor' if severity == 'good' else severity
    
    if qc_report['artifact_metrics']['drift_percent'] > 10:
        issues.append('High signal drift')
        severity = 'poor' if severity == 'good' else severity
    
    return {
        'overall_quality': severity,
        'issues_detected': issues,
        'num_issues': len(issues)
    }
