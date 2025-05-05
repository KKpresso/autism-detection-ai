import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Any
import nibabel as nib
from nilearn import plotting
import json
from .neuro_metrics import NeuroMetrics


class ModelEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'evaluation_results'
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def cross_validate(
        self,
        dataset: torch.utils.data.Dataset,
        n_splits: int = 5,
        batch_size: int = 32,
        num_epochs: int = 50
    ) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation."""
        from ..training.trainer import ModelTrainer
        
        # Prepare cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_auc': [], 'val_auc': [],
            'fold_attention_weights': []
        }
        
        # Get all labels for stratification
        all_labels = [label for _, _, label in dataset]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), all_labels)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Create data loaders for this fold
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler
            )
            val_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=val_sampler
            )
            
            # Reset model for this fold
            self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            
            # Train model
            trainer = ModelTrainer(self.model, device=self.device, use_wandb=False)
            history = trainer.train(
                train_loader, val_loader,
                num_epochs=num_epochs,
                model_save_path=self.output_dir / f'fold_{fold + 1}'
            )
            
            # Store results
            final_metrics = history[-1]
            cv_results['train_loss'].append(final_metrics['train']['loss'])
            cv_results['val_loss'].append(final_metrics['val']['loss'])
            cv_results['train_acc'].append(final_metrics['train']['accuracy'])
            cv_results['val_acc'].append(final_metrics['val']['accuracy'])
            cv_results['train_auc'].append(final_metrics['train']['auc_roc'])
            cv_results['val_auc'].append(final_metrics['val']['auc_roc'])
            
            # Save fold-specific results
            self._save_fold_results(fold + 1, final_metrics)
        
        # Save and plot cross-validation results
        self._save_cv_results(cv_results)
        self._plot_cv_results(cv_results)
        
        return cv_results
    
    def analyze_brain_regions(
        self,
        data_loader: torch.utils.data.DataLoader,
        region_names: List[str],
        atlas_path: str
    ) -> Dict[str, Any]:
        """Analyze importance of different brain regions using attention weights."""
        self.model.eval()
        region_importance = {name: 0.0 for name in region_names}
        total_samples = 0
        
        with torch.no_grad():
            for data, adj, _ in data_loader:
                data = data.to(self.device)
                adj = adj.to(self.device)
                
                # Get model outputs with attention weights
                _, attention_weights = self.model(data, adj)
                
                # Average attention weights across layers and heads
                graph_attention = attention_weights['graph_attention']
                avg_attention = torch.mean(torch.stack([
                    attn.mean(dim=1)  # Average across heads
                    for attn in graph_attention
                ]), dim=0)  # Average across layers
                
                # Accumulate importance scores
                for i, name in enumerate(region_names):
                    region_importance[name] += avg_attention[:, i].mean().item()
                
                total_samples += data.size(0)
        
        # Normalize importance scores
        for name in region_importance:
            region_importance[name] /= total_samples
        
        # Sort regions by importance
        sorted_regions = dict(sorted(
            region_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Save and visualize results
        self._save_region_importance(sorted_regions)
        self._plot_brain_regions(sorted_regions, atlas_path)
        
        return sorted_regions
    
    def generate_performance_report(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_attention = []
        all_connectivity = []
        all_timeseries = []
        control_data = []
        autism_data = []
        
        with torch.no_grad():
            for data, adj, labels in test_loader:
                data = data.to(self.device)
                adj = adj.to(self.device)
                
                outputs, attention_weights = self.model(data, adj)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_attention.append({
                    'graph': [attn.cpu().numpy() for attn in attention_weights['graph_attention']],
                    'temporal': attention_weights['temporal_attention'].cpu().numpy()
                })
                all_connectivity.append(adj.cpu().numpy())
                all_timeseries.append(data.cpu().numpy())
                
                # Separate data by group
                for i, label in enumerate(labels):
                    if label == 0:
                        control_data.append(data[i].cpu().numpy())
                    else:
                        autism_data.append(data[i].cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_preds)
        true_labels = np.array(all_labels)
        connectivity_matrices = np.concatenate(all_connectivity)
        timeseries_data = np.concatenate(all_timeseries)
        control_data = np.array(control_data)
        autism_data = np.array(autism_data)
        
        # Initialize NeuroMetrics
        neuro_metrics = NeuroMetrics()
        
        # Calculate standard performance metrics
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        
        results = {
            'roc_auc': auc(fpr, tpr),
            'pr_auc': auc(recall, precision),
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'pr_curve': {'precision': precision, 'recall': recall}
        }
        
        # Calculate clinical metrics
        results['clinical_metrics'] = neuro_metrics.calculate_clinical_metrics(
            predictions, true_labels
        )
        
        # Calculate network metrics
        network_metrics = []
        for conn_matrix in connectivity_matrices:
            network_metrics.append(
                neuro_metrics.calculate_network_metrics(conn_matrix)
            )
        results['network_metrics'] = {
            k: np.mean([m[k] for m in network_metrics])
            for k in network_metrics[0]
        }
        
        # Calculate regional metrics
        results['regional_metrics'] = neuro_metrics.calculate_regional_metrics(
            [attn['graph'][0] for attn in all_attention],  # Using first attention layer
            connectivity_matrices
        )
        
        # Calculate temporal metrics
        results['temporal_metrics'] = neuro_metrics.calculate_temporal_metrics(
            np.concatenate([attn['temporal'] for attn in all_attention]),
            timeseries_data
        )
        
        # Calculate reliability metrics
        results['reliability_metrics'] = neuro_metrics.calculate_reliability_metrics(
            predictions.reshape(-1, 1),
            connectivity_matrices
        )
        
        # Calculate group difference metrics
        results['group_differences'] = neuro_metrics.calculate_group_difference_metrics(
            control_data,
            autism_data
        )
        
        # Calculate connectivity metrics
        results['connectivity_metrics'] = neuro_metrics.calculate_connectivity_metrics(
            connectivity_matrices
        )
        
        # Generate and save visualizations
        self._plot_performance_curves(results)
        self._plot_attention_patterns(all_attention)
        self._plot_advanced_metrics(results)
        
        # Save results
        self._save_performance_report(results)
        
        return results
    
    def _plot_advanced_metrics(self, results: Dict[str, Any]):
        """Plot advanced neuroimaging metrics."""
        plt.figure(figsize=(20, 15))
        
        # Plot network metrics
        plt.subplot(3, 2, 1)
        network_metrics = results['network_metrics']
        plt.bar(network_metrics.keys(), network_metrics.values())
        plt.xticks(rotation=45)
        plt.title('Network Metrics')
        
        # Plot regional importance
        plt.subplot(3, 2, 2)
        regional_metrics = results['regional_metrics']
        plt.plot(regional_metrics['attention_strength'], label='Attention')
        plt.plot(regional_metrics['connectivity_strength'], label='Connectivity')
        plt.title('Regional Importance')
        plt.legend()
        
        # Plot temporal metrics
        plt.subplot(3, 2, 3)
        temporal_metrics = results['temporal_metrics']
        plt.plot(temporal_metrics['temporal_variability'], label='Variability')
        plt.plot(temporal_metrics['temporal_entropy'], label='Entropy')
        plt.title('Temporal Metrics')
        plt.legend()
        
        # Plot clinical metrics
        plt.subplot(3, 2, 4)
        clinical_metrics = results['clinical_metrics']
        plt.bar(clinical_metrics.keys(), clinical_metrics.values())
        plt.xticks(rotation=45)
        plt.title('Clinical Metrics')
        
        # Plot group differences
        plt.subplot(3, 2, 5)
        group_diff = results['group_differences']
        plt.bar(['Effect Size', 'T-stat', 'P-value', 'Overlap'],
                [group_diff['cohens_d'], group_diff['t_statistic'],
                 group_diff['p_value'], group_diff['distribution_overlap']])
        plt.title('Group Differences')
        
        # Plot reliability metrics
        plt.subplot(3, 2, 6)
        reliability = results['reliability_metrics']
        plt.bar(reliability.keys(), reliability.values())
        plt.xticks(rotation=45)
        plt.title('Reliability Metrics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'advanced_metrics.png')
        plt.close()
    
    def _save_fold_results(self, fold: int, metrics: Dict[str, Any]):
        """Save results for a specific cross-validation fold."""
        fold_dir = self.output_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        
        with open(fold_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def _save_cv_results(self, cv_results: Dict[str, List[float]]):
        """Save cross-validation results."""
        results_file = self.output_dir / 'cv_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                k: np.mean(v) for k, v in cv_results.items()
                if isinstance(v, list) and v and isinstance(v[0], (int, float))
            }, f, indent=4)
    
    def _plot_cv_results(self, cv_results: Dict[str, List[float]]):
        """Plot cross-validation results."""
        plt.figure(figsize=(12, 6))
        
        metrics = ['acc', 'auc']
        for metric in metrics:
            plt.subplot(1, len(metrics), metrics.index(metric) + 1)
            data = {
                'Train': cv_results[f'train_{metric}'],
                'Validation': cv_results[f'val_{metric}']
            }
            df = pd.DataFrame(data)
            sns.boxplot(data=df)
            plt.title(f'Cross-validation {metric.upper()}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_results.png')
        plt.close()
    
    def _save_region_importance(self, region_importance: Dict[str, float]):
        """Save brain region importance scores."""
        with open(self.output_dir / 'region_importance.json', 'w') as f:
            json.dump(region_importance, f, indent=4)
    
    def _plot_brain_regions(self, region_importance: Dict[str, float], atlas_path: str):
        """Plot brain regions with their importance scores."""
        # Load atlas
        atlas_img = nib.load(atlas_path)
        
        # Create importance map
        importance_map = np.zeros(atlas_img.shape)
        atlas_data = atlas_img.get_fdata()
        
        for i, (region, importance) in enumerate(region_importance.items()):
            importance_map[atlas_data == i + 1] = importance
        
        # Create and save visualization
        display = plotting.plot_stat_map(
            nib.Nifti1Image(importance_map, atlas_img.affine),
            title='Brain Region Importance',
            cut_coords=(0, 0, 0)
        )
        display.savefig(self.output_dir / 'region_importance.png')
        plt.close()
    
    def _plot_performance_curves(self, results: Dict[str, Any]):
        """Plot ROC and PR curves."""
        plt.figure(figsize=(12, 5))
        
        # ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(
            results['roc_curve']['fpr'],
            results['roc_curve']['tpr'],
            label=f"AUC = {results['roc_auc']:.3f}"
        )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # PR curve
        plt.subplot(1, 2, 2)
        plt.plot(
            results['pr_curve']['recall'],
            results['pr_curve']['precision'],
            label=f"AUC = {results['pr_auc']:.3f}"
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_curves.png')
        plt.close()
    
    def _plot_attention_patterns(self, attention_data: List[Dict[str, Any]]):
        """Plot attention patterns across samples."""
        # Average attention patterns
        avg_graph_attention = np.mean([
            [layer[0] for layer in sample['graph']]  # First batch item
            for sample in attention_data
        ], axis=0)
        
        avg_temporal_attention = np.mean([
            sample['temporal'][0]  # First batch item
            for sample in attention_data
        ], axis=0)
        
        # Plot
        plt.figure(figsize=(15, 5))
        
        # Graph attention
        for i, layer_attn in enumerate(avg_graph_attention):
            plt.subplot(1, len(avg_graph_attention) + 1, i + 1)
            sns.heatmap(layer_attn, cmap='viridis')
            plt.title(f'Graph Attention Layer {i + 1}')
        
        # Temporal attention
        plt.subplot(1, len(avg_graph_attention) + 1, len(avg_graph_attention) + 1)
        sns.heatmap(avg_temporal_attention.T, cmap='viridis')
        plt.title('Temporal Attention')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_patterns.png')
        plt.close()
