import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch_idx, (data, adj, target) in enumerate(train_loader):
            data = data.to(self.device)
            adj = adj.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data, adj)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy() > 0.5)
            true_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for data, adj, target in val_loader:
            data = data.to(self.device)
            adj = adj.to(self.device)
            target = target.to(self.device)
            
            output = self.model(data, adj)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy() > 0.5)
            true_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    @staticmethod
    def calculate_metrics(true_labels, predictions):
        """Calculate various performance metrics."""
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions)
        }
    
    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=10):
        """Complete training procedure with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            # Train and evaluate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
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
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return training_history
