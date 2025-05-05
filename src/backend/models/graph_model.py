import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import GraphAttention, TemporalAttention, MultiHeadAttention


class GraphEncoder(nn.Module):
    """Graph encoder that learns representations from fMRI connectivity matrices."""
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_heads=4, dropout=0.3):
        super(GraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Multi-head attention layer
        self.mha = MultiHeadAttention(
            in_features=input_dim,
            head_dim=hidden_dim // num_heads,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Graph attention layers
        self.gat1 = GraphAttention(input_dim, hidden_dim, dropout)
        self.gat2 = GraphAttention(hidden_dim, hidden_dim, dropout)
        self.gat3 = GraphAttention(hidden_dim, embedding_dim, dropout)
        
        # Residual connections
        self.res1 = nn.Linear(input_dim, hidden_dim)
        self.res2 = nn.Linear(hidden_dim, hidden_dim)
        self.res3 = nn.Linear(hidden_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # Multi-head self attention
        mha_out, _ = self.mha(x, adj)
        x = x + mha_out  # Residual connection
        
        # First graph attention layer with residual
        gat1_out, attn1 = self.gat1(x, adj)
        res1_out = self.res1(x)
        x = self.layer_norm1(gat1_out + res1_out)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second graph attention layer with residual
        gat2_out, attn2 = self.gat2(x, adj)
        res2_out = self.res2(x)
        x = self.layer_norm2(gat2_out + res2_out)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Final graph attention layer with residual
        gat3_out, attn3 = self.gat3(x, adj)
        res3_out = self.res3(x)
        x = self.layer_norm3(gat3_out + res3_out)
        
        return x, (attn1, attn2, attn3)


class TemporalEncoder(nn.Module):
    """Temporal encoder for capturing dynamic patterns in fMRI data."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.temporal_attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional
    
    def forward(self, x):
        # GRU encoding
        outputs, _ = self.gru(x)
        # Apply temporal attention
        context, attention = self.temporal_attention(outputs)
        return context, attention


class AutismClassifier(nn.Module):
    """Complete model for autism detection using graph and temporal representations."""
    def __init__(self, num_regions, hidden_dim=64, embedding_dim=32, num_heads=4, dropout=0.3):
        super(AutismClassifier, self).__init__()
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            input_dim=num_regions,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Classification layers with residual connections
        self.fc1 = nn.Linear(embedding_dim * num_regions + hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Batch normalization and layer normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # Get graph embeddings and attention weights
        graph_embeddings, graph_attention = self.graph_encoder(x, adj)
        
        # Get temporal embeddings
        temporal_embeddings, temporal_attention = self.temporal_encoder(graph_embeddings)
        
        # Combine graph and temporal features
        graph_features = graph_embeddings.view(graph_embeddings.size(0), -1)
        combined = torch.cat([graph_features, temporal_embeddings], dim=1)
        
        # Classification with residual connections
        h1 = self.fc1(combined)
        h1 = self.layer_norm(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)
        
        h2 = self.fc2(h1)
        h2 = self.batch_norm2(h2)
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        
        # Final classification
        out = self.fc3(h2)
        out = torch.sigmoid(out)
        
        # Return predictions and attention weights for interpretability
        return out, {
            'graph_attention': graph_attention,
            'temporal_attention': temporal_attention
        }
