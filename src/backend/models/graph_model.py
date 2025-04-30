import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    """Graph encoder that learns representations from fMRI connectivity matrices."""
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.3):
        super(GraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Graph convolution layers
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gc3 = nn.Linear(hidden_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x, adj):
        # x: node features [batch_size, num_nodes, input_dim]
        # adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        
        # First graph convolution
        x = torch.bmm(adj, x)  # Message passing
        x = self.gc1(x)
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second graph convolution
        x = torch.bmm(adj, x)
        x = self.gc2(x)
        x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final embedding
        x = torch.bmm(adj, x)
        x = self.gc3(x)
        return x


class AutismClassifier(nn.Module):
    """Complete model for autism detection using graph representations."""
    def __init__(self, num_regions, hidden_dim=64, embedding_dim=32, dropout=0.3):
        super(AutismClassifier, self).__init__()
        
        self.graph_encoder = GraphEncoder(
            input_dim=num_regions,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # MLP for classification
        self.fc1 = nn.Linear(embedding_dim * num_regions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
    
    def forward(self, x, adj):
        # Get graph embeddings
        graph_embeddings = self.graph_encoder(x, adj)
        
        # Flatten embeddings
        flat = graph_embeddings.view(graph_embeddings.size(0), -1)
        
        # Classification layers
        x = self.fc1(flat)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final classification
        x = self.fc3(x)
        return torch.sigmoid(x)
