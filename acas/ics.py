import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

class IntrinsicConsistencyScorer:
    """Implements ICS Method B (Self-Similarity) for anomaly detection."""
    
    def __init__(self, k_neighbors: int = 5):
        """
        Initialize the Intrinsic Consistency Scorer.
        
        Args:
            k_neighbors: Number of nearest neighbors to consider
        """
        self.k_neighbors = k_neighbors
        
    def compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            features: Feature tensor of shape [N, D]
            
        Returns:
            Similarity matrix of shape [N, N]
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=-1)
        # Compute cosine similarity
        similarity = torch.mm(features, features.t())
        return similarity
    
    def compute_ics_scores(
        self,
        features: torch.Tensor,
        similarity_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Intrinsic Consistency Scores.
        
        Args:
            features: Feature tensor of shape [N, D]
            similarity_matrix: Optional pre-computed similarity matrix
            
        Returns:
            ICS scores of shape [N]
        """
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix(features)
            
        # Get top-k neighbors for each patch
        topk_values, topk_indices = torch.topk(
            similarity_matrix,
            k=self.k_neighbors + 1,  # +1 because patch is similar to itself
            dim=-1
        )
        
        # Remove self-similarity
        topk_values = topk_values[:, 1:]  # Remove first column (self-similarity)
        topk_indices = topk_indices[:, 1:]
        
        # Compute mean of top-k neighbors
        mean_similar_neighbors = torch.mean(topk_values, dim=-1)
        
        # Compute ICS scores (1 - similarity to mean of neighbors)
        ics_scores = 1.0 - mean_similar_neighbors
        
        return ics_scores
    
    def compute_layer_ics(
        self,
        layer_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ICS scores for a single layer.
        
        Args:
            layer_features: Feature tensor of shape [B, N, D]
            
        Returns:
            ICS scores of shape [B, N]
        """
        batch_size, num_patches, feat_dim = layer_features.shape
        
        # Reshape for batch processing
        features_flat = layer_features.reshape(-1, feat_dim)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(features_flat)
        
        # Compute ICS scores
        ics_scores = self.compute_ics_scores(features_flat, similarity_matrix)
        
        # Reshape back to [B, N]
        ics_scores = ics_scores.reshape(batch_size, num_patches)
        
        return ics_scores
    
    def compute_hierarchical_ics(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ICS scores for all layers.
        
        Args:
            features: Dictionary of layer features from HierarchicalFeatureExtractor
            
        Returns:
            Dictionary of ICS scores for each layer
        """
        ics_scores = {}
        
        for layer_name, layer_features in features.items():
            if layer_name != 'cls_token':  # Skip CLS token
                ics_scores[layer_name] = self.compute_layer_ics(layer_features)
                
        return ics_scores 