import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Any
import torch
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
from scipy.spatial.distance import pdist, squareform


class NeuroMetrics:
    """Advanced neuroimaging-specific metrics for model evaluation."""
    
    @staticmethod
    def calculate_network_metrics(connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate graph theoretical metrics from connectivity matrix."""
        # Create graph from connectivity matrix
        G = nx.from_numpy_array(np.abs(connectivity_matrix))
        
        metrics = {
            'global_efficiency': nx.global_efficiency(G),
            'clustering_coefficient': nx.average_clustering(G),
            'characteristic_path_length': nx.average_shortest_path_length(G),
            'modularity': nx.community.modularity(
                G,
                nx.community.greedy_modularity_communities(G)
            ),
            'degree_centrality': np.mean(list(nx.degree_centrality(G).values())),
            'betweenness_centrality': np.mean(list(nx.betweenness_centrality(G).values()))
        }
        
        return metrics
    
    @staticmethod
    def calculate_regional_metrics(
        attention_weights: List[np.ndarray],
        connectivity_matrices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate region-specific metrics combining attention and connectivity."""
        n_regions = connectivity_matrices.shape[1]
        
        # Average attention weights across layers
        avg_attention = np.mean([attn for attn in attention_weights], axis=0)
        
        # Calculate regional metrics
        metrics = {
            'attention_strength': np.mean(avg_attention, axis=0),
            'connectivity_strength': np.mean(np.abs(connectivity_matrices), axis=(0, 2)),
            'regional_diversity': np.array([
                stats.entropy(avg_attention[i]) for i in range(n_regions)
            ]),
            'hub_score': np.array([
                np.mean(np.sort(connectivity_matrices[:, i, :])[:, -5:])
                for i in range(n_regions)
            ])
        }
        
        return metrics
    
    @staticmethod
    def calculate_temporal_metrics(
        temporal_attention: np.ndarray,
        timeseries_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate temporal dynamics metrics."""
        # Calculate temporal variability
        temporal_var = np.var(timeseries_data, axis=1)
        
        # Calculate temporal entropy
        temporal_entropy = np.array([
            stats.entropy(ts[~np.isnan(ts)])
            for ts in timeseries_data
        ])
        
        # Calculate attention-weighted temporal features
        weighted_timeseries = temporal_attention[:, :, None] * timeseries_data
        
        metrics = {
            'temporal_variability': temporal_var,
            'temporal_entropy': temporal_entropy,
            'attention_weighted_var': np.var(weighted_timeseries, axis=1),
            'temporal_attention_entropy': np.array([
                stats.entropy(attn) for attn in temporal_attention
            ])
        }
        
        return metrics
    
    @staticmethod
    def calculate_clinical_metrics(
        predictions: np.ndarray,
        true_labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate clinically relevant metrics."""
        # Convert predictions to binary
        binary_preds = (predictions > threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
        
        # Calculate clinical metrics
        sensitivity = tp / (tp + fn)  # True Positive Rate
        specificity = tn / (tn + fp)  # True Negative Rate
        ppv = tp / (tp + fp)  # Positive Predictive Value
        npv = tn / (tn + fn)  # Negative Predictive Value
        
        # Calculate likelihood ratios
        positive_lr = sensitivity / (1 - specificity)
        negative_lr = (1 - sensitivity) / specificity
        
        # Calculate diagnostic odds ratio
        diagnostic_odds_ratio = positive_lr / negative_lr
        
        # Calculate Youden's J statistic
        youdens_j = sensitivity + specificity - 1
        
        metrics = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'positive_likelihood_ratio': positive_lr,
            'negative_likelihood_ratio': negative_lr,
            'diagnostic_odds_ratio': diagnostic_odds_ratio,
            'youdens_j': youdens_j,
            'balanced_accuracy': (sensitivity + specificity) / 2
        }
        
        return metrics
    
    @staticmethod
    def calculate_reliability_metrics(
        predictions: List[np.ndarray],
        connectivity_matrices: List[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate reliability and reproducibility metrics."""
        # Calculate prediction consistency
        pred_consistency = np.mean([
            np.corrcoef(pred1, pred2)[0, 1]
            for i, pred1 in enumerate(predictions)
            for j, pred2 in enumerate(predictions)
            if i < j
        ])
        
        # Calculate pattern consistency
        pattern_consistency = np.mean([
            np.corrcoef(conn1.flatten(), conn2.flatten())[0, 1]
            for i, conn1 in enumerate(connectivity_matrices)
            for j, conn2 in enumerate(connectivity_matrices)
            if i < j
        ])
        
        # Calculate intraclass correlation
        predictions_array = np.array(predictions)
        icc = stats.f_oneway(*predictions_array)[0]
        
        metrics = {
            'prediction_consistency': pred_consistency,
            'pattern_consistency': pattern_consistency,
            'icc_score': icc
        }
        
        return metrics
    
    @staticmethod
    def calculate_group_difference_metrics(
        control_data: np.ndarray,
        autism_data: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate metrics related to group differences."""
        # Calculate effect sizes
        cohens_d = (np.mean(autism_data) - np.mean(control_data)) / np.sqrt(
            (np.var(autism_data) + np.var(control_data)) / 2
        )
        
        # Calculate statistical significance
        ttest_result = stats.ttest_ind(autism_data, control_data)
        
        # Calculate distribution overlap
        kde_control = stats.gaussian_kde(control_data.flatten())
        kde_autism = stats.gaussian_kde(autism_data.flatten())
        x_range = np.linspace(
            min(control_data.min(), autism_data.min()),
            max(control_data.max(), autism_data.max()),
            1000
        )
        overlap = np.minimum(kde_control(x_range), kde_autism(x_range)).sum()
        
        metrics = {
            'cohens_d': cohens_d,
            't_statistic': ttest_result.statistic,
            'p_value': ttest_result.pvalue,
            'distribution_overlap': overlap
        }
        
        return metrics
    
    @staticmethod
    def calculate_connectivity_metrics(
        connectivity_matrices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate advanced connectivity metrics."""
        # Calculate various connectivity measures
        correlation_measure = ConnectivityMeasure(kind='correlation')
        partial_correlation_measure = ConnectivityMeasure(kind='partial correlation')
        covariance_measure = ConnectivityMeasure(kind='covariance')
        
        metrics = {
            'correlation': correlation_measure.fit_transform(connectivity_matrices),
            'partial_correlation': partial_correlation_measure.fit_transform(connectivity_matrices),
            'covariance': covariance_measure.fit_transform(connectivity_matrices),
            'connectivity_variance': np.var(connectivity_matrices, axis=0),
            'connectivity_entropy': np.array([
                stats.entropy(cm[~np.isnan(cm)])
                for cm in connectivity_matrices
            ])
        }
        
        return metrics
