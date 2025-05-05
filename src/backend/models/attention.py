"""Attention mechanisms for the graph neural network model.

This module implements various attention mechanisms to help the model
focus on the most relevant brain regions and their connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    """Graph attention layer for learning region importance."""
    
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Trainable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
    
    def forward(self, h, adj):
        """
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size = h.size(0)
        N = h.size(1)  # number of nodes
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [batch_size, N, out_features]
        
        # Attention mechanism
        # Prepare for concatenation
        a_input = torch.cat([
            Wh.repeat_interleave(N, dim=1),
            Wh.repeat(1, N, 1)
        ], dim=2)  # [batch_size, N*N, 2*out_features]
        
        # Calculate attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a))  # [batch_size, N*N, 1]
        e = e.view(batch_size, N, N)  # [batch_size, N, N]
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)  # [batch_size, N, out_features]
        
        return h_prime, attention


class TemporalAttention(nn.Module):
    """Temporal attention for capturing dynamic patterns."""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Temporal sequence [batch_size, seq_len, hidden_dim]
        """
        # Calculate attention weights
        weights = self.attention(x)  # [batch_size, seq_len, 1]
        weights = F.softmax(weights, dim=1)
        
        # Apply attention
        weighted = x * weights
        return weighted.sum(dim=1), weights  # [batch_size, hidden_dim]


class MultiHeadAttention(nn.Module):
    """Multi-head attention for capturing different aspects of connectivity."""
    
    def __init__(self, in_features, head_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(in_features, head_dim * num_heads)
        self.k_linear = nn.Linear(in_features, head_dim * num_heads)
        self.v_linear = nn.Linear(in_features, head_dim * num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(head_dim * num_heads, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features [batch_size, num_nodes, in_features]
            mask: Optional attention mask [batch_size, num_nodes, num_nodes]
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # Linear projections and reshape
        q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float)
        )
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -9e15)
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Get output
        out = torch.matmul(attn, v)  # [batch_size, num_heads, num_nodes, head_dim]
        out = out.transpose(1, 2)  # [batch_size, num_nodes, num_heads, head_dim]
        out = out.reshape(batch_size, num_nodes, -1)  # [batch_size, num_nodes, num_heads*head_dim]
        
        # Final projection
        out = self.out_proj(out)
        
        return out, attn
