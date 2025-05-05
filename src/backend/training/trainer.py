import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelTrainer:
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_wandb=True
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="autism-detection-ai", entity="your-entity")
            wandb.watch(model)
    
    def train_epoch(self, train_loader, scheduler=None):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        attention_maps = []
        
        for batch_idx, (data, adj, target) in enumerate(train_loader):
            data = data.to(self.device)
            adj = adj.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output, attention_weights = self.model(data, adj)
            loss = self.criterion(output, target)
            
            # Add L1 regularization on attention weights
            attention_l1 = sum(
                torch.mean(torch.abs(attn))
                for attn in attention_weights['graph_attention']
            )
            loss = loss + 0.01 * attention_l1
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy() > 0.5)
            true_labels.extend(target.cpu().numpy())
            
            # Store attention maps for visualization
            attention_maps.append({
                'graph': [attn.detach().cpu().numpy() for attn in attention_weights['graph_attention']],
                'temporal': attention_weights['temporal_attention'].detach().cpu().numpy()
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions)
        metrics['loss'] = total_loss / len(train_loader)
        
        if self.use_wandb:
            wandb.log({f"train_{k}": v for k, v in metrics.items()})
            
            # Log attention visualizations periodically
            if len(attention_maps) > 0:
                self.log_attention_maps(attention_maps[-1], "train")
        
        return metrics, attention_maps
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        attention_maps = []
        
        for data, adj, target in val_loader:
            data = data.to(self.device)
            adj = adj.to(self.device)
            target = target.to(self.device)
            
            output, attention_weights = self.model(data, adj)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy() > 0.5)
            true_labels.extend(target.cpu().numpy())
            
            attention_maps.append({
                'graph': [attn.cpu().numpy() for attn in attention_weights['graph_attention']],
                'temporal': attention_weights['temporal_attention'].cpu().numpy()
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions)
        metrics['loss'] = total_loss / len(val_loader)
        
        if self.use_wandb:
            wandb.log({f"val_{k}": v for k, v in metrics.items()})
            
            # Log attention visualizations
            if len(attention_maps) > 0:
                self.log_attention_maps(attention_maps[-1], "val")
        
        return metrics, attention_maps
    
    def log_attention_maps(self, attention_maps, prefix="train"):
        """Log attention visualizations to wandb."""
        # Create directory for saving plots
        save_dir = Path("visualizations")
        save_dir.mkdir(exist_ok=True)
        
        # Plot graph attention maps
        for layer_idx, attn in enumerate(attention_maps['graph']):
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn[0], cmap='viridis')  # First sample in batch
            plt.title(f'Graph Attention Layer {layer_idx + 1}')
            plt.savefig(save_dir / f'{prefix}_graph_attention_{layer_idx}.png')
            plt.close()
            
            if self.use_wandb:
                wandb.log({
                    f"{prefix}_graph_attention_{layer_idx}": wandb.Image(
                        str(save_dir / f'{prefix}_graph_attention_{layer_idx}.png')
                    )
                })
        
        # Plot temporal attention
        plt.figure(figsize=(12, 4))
        sns.heatmap(attention_maps['temporal'][0].T, cmap='viridis')
        plt.title('Temporal Attention')
        plt.savefig(save_dir / f'{prefix}_temporal_attention.png')
        plt.close()
        
        if self.use_wandb:
            wandb.log({
                f"{prefix}_temporal_attention": wandb.Image(
                    str(save_dir / f'{prefix}_temporal_attention.png')
                )
            })
    
    @staticmethod
    def calculate_metrics(true_labels, predictions):
        """Calculate various performance metrics."""
        pred_probs = np.array(predictions, dtype=float)
        true_labels = np.array(true_labels)
        
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions),
            'auc_roc': roc_auc_score(true_labels, pred_probs)
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=100,
        early_stopping_patience=10,
        model_save_path='models'
    ):
        """Complete training procedure with early stopping and learning rate scheduling."""
        # Create model save directory
        save_path = Path(model_save_path)
        save_path.mkdir(exist_ok=True)
        
        # Learning rate scheduler
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            # Train and evaluate
            train_metrics, train_attention = self.train_epoch(train_loader, scheduler)
            val_metrics, val_attention = self.evaluate(val_loader)
            
            # Save metrics
            history_entry = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            training_history.append(history_entry)
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model and attention maps
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics
                }, save_path / 'best_model.pt')
                
                # Save attention maps for best model
                np.save(save_path / 'best_attention_maps.npy', {
                    'train': train_attention[-1],
                    'val': val_attention[-1]
                })
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Metrics: {', '.join(f'{k}: {v:.4f}' for k, v in val_metrics.items() if k != 'loss')}")
        
        return training_history
