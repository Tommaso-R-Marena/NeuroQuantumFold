"""Neural network encoders for protein sequence representation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for protein sequences.
    
    Encodes amino acid sequences into high-dimensional embeddings
    suitable for downstream neurosymbolic processing.
    """
    
    def __init__(
        self,
        vocab_size: int = 21,  # 20 amino acids + padding
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1000
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projections for different downstream tasks
        self.contact_predictor = nn.Linear(d_model, d_model)
        self.structure_head = nn.Linear(d_model, 3)  # x, y, z coordinates
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.
        
        Args:
            src: Input sequence tensor [batch_size, seq_len]
            src_mask: Optional attention mask
            
        Returns:
            embeddings: Sequence embeddings [batch_size, seq_len, d_model]
            contacts: Predicted contact maps [batch_size, seq_len, d_model]
            coords: Preliminary 3D coordinates [batch_size, seq_len, 3]
        """
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoding
        embeddings = self.transformer_encoder(src, src_mask)
        
        # Downstream predictions
        contacts = self.contact_predictor(embeddings)
        coords = self.structure_head(embeddings)
        
        return embeddings, contacts, coords


class ESM2Encoder(nn.Module):
    """Wrapper for ESM-2 pretrained protein language model.
    
    Uses Meta's ESM-2 for transfer learning with evolutionary context.
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", freeze_base: bool = True):
        super().__init__()
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            
            if freeze_base:
                for param in self.model.parameters():
                    param.requires_grad = False
                    
            self.batch_converter = self.alphabet.get_batch_converter()
            
        except ImportError:
            raise ImportError(
                "ESM-2 requires the 'fair-esm' package. "
                "Install with: pip install fair-esm"
            )
    
    def forward(self, sequences: list) -> torch.Tensor:
        """Encode protein sequences using ESM-2.
        
        Args:
            sequences: List of (label, sequence) tuples
            
        Returns:
            embeddings: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        _, _, batch_tokens = self.batch_converter(sequences)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            results = self.model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33]
        
        return embeddings


class HierarchicalEncoder(nn.Module):
    """Multi-scale hierarchical encoder for capturing both local and global patterns."""
    
    def __init__(self, d_model: int = 512, scales: list = [1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2)
            for k in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale feature extraction.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Multi-scale features [batch_size, seq_len, d_model]
        """
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # Extract features at different scales
        multi_scale_features = []
        for conv in self.conv_layers:
            features = F.relu(conv(x))
            multi_scale_features.append(features)
        
        # Concatenate and fuse
        combined = torch.cat(multi_scale_features, dim=1)
        combined = combined.transpose(1, 2)  # [batch, seq_len, d_model * scales]
        fused = self.fusion(combined)
        
        return fused
