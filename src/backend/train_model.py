import os
import argparse
import numpy as np
from models.graph_model import AutismClassifier
from training.trainer import ModelTrainer
from data.preprocessing import prepare_data_loaders


def main(args):
    # Load preprocessed connectivity matrices and labels
    # Note: You'll need to implement the data loading from ABIDE dataset
    connectivity_matrices = np.load(args.connectivity_path)
    labels = np.load(args.labels_path)
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(
        connectivity_matrices,
        labels,
        batch_size=args.batch_size
    )
    
    # Initialize model
    num_regions = connectivity_matrices.shape[1]  # Number of brain regions
    model = AutismClassifier(
        num_regions=num_regions,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    
    # Train model
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.patience
    )
    
    # Save training history
    np.save('models/training_history.npy', history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autism detection model')
    parser.add_argument('--connectivity_path', type=str, required=True,
                      help='Path to preprocessed connectivity matrices')
    parser.add_argument('--labels_path', type=str, required=True,
                      help='Path to subject labels')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=32,
                      help='Embedding dimension size')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
