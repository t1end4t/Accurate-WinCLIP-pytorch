import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple

class ExtrinsicMatchingScorer:
    """Implements Extrinsic Matching Scorer for anomaly detection."""
    
    def __init__(self, mode: str = "zero_shot"):
        """
        Initialize the Extrinsic Matching Scorer.
        
        Args:
            mode: Either "zero_shot" or "few_shot"
        """
        assert mode in ["zero_shot", "few_shot"], "Mode must be either 'zero_shot' or 'few_shot'"
        self.mode = mode
        
    def compute_similarity(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and reference features.
        
        Args:
            query_features: Query features of shape [N, D]
            reference_features: Reference features of shape [M, D]
            
        Returns:
            Similarity matrix of shape [N, M]
        """
        # Normalize features
        query_features = F.normalize(query_features, p=2, dim=-1)
        reference_features = F.normalize(reference_features, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.mm(query_features, reference_features.t())
        return similarity
    
    def compute_zero_shot_ems(
        self,
        query_features: torch.Tensor,
        normal_text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute EMS scores in zero-shot mode.
        
        Args:
            query_features: Query features of shape [N, D]
            normal_text_embeddings: Normal text embeddings of shape [T, D]
            
        Returns:
            EMS scores of shape [N]
        """
        # Compute similarity with text embeddings
        similarity = self.compute_similarity(query_features, normal_text_embeddings)
        
        # Get maximum similarity for each query feature
        max_similarity = torch.max(similarity, dim=-1)[0]
        
        # Compute EMS scores (1 - max_similarity)
        ems_scores = 1.0 - max_similarity
        
        return ems_scores
    
    def compute_few_shot_ems(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute EMS scores in few-shot mode.
        
        Args:
            query_features: Query features of shape [N, D]
            reference_features: Reference features of shape [K*N, D]
            
        Returns:
            EMS scores of shape [N]
        """
        # Reshape reference features if needed
        if reference_features.dim() > 2:
            batch_size, num_patches, feat_dim = reference_features.shape
            reference_features = reference_features.reshape(-1, feat_dim)
        
        # Compute similarity with reference features
        similarity = self.compute_similarity(query_features, reference_features)
        
        # Get maximum similarity for each query feature
        max_similarity = torch.max(similarity, dim=-1)[0]
        
        # Compute EMS scores (1 - max_similarity)
        ems_scores = 1.0 - max_similarity
        
        return ems_scores
    
    def compute_layer_ems(
        self,
        layer_features: torch.Tensor,
        reference_features: Optional[torch.Tensor] = None,
        normal_text_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute EMS scores for a single layer.
        
        Args:
            layer_features: Layer features of shape [B, N, D]
            reference_features: Optional reference features for few-shot mode
            normal_text_embeddings: Optional text embeddings for zero-shot mode
            
        Returns:
            EMS scores of shape [B, N]
        """
        batch_size, num_patches, feat_dim = layer_features.shape
        
        # Reshape for batch processing
        features_flat = layer_features.reshape(-1, feat_dim)
        
        if self.mode == "zero_shot":
            assert normal_text_embeddings is not None, "Text embeddings required for zero-shot mode"
            ems_scores = self.compute_zero_shot_ems(features_flat, normal_text_embeddings)
        else:  # few_shot
            assert reference_features is not None, "Reference features required for few-shot mode"
            ems_scores = self.compute_few_shot_ems(features_flat, reference_features)
        
        # Reshape back to [B, N]
        ems_scores = ems_scores.reshape(batch_size, num_patches)
        
        # Normalize scores to [0, 1] range
        min_scores = torch.min(ems_scores, dim=1, keepdim=True)[0]
        max_scores = torch.max(ems_scores, dim=1, keepdim=True)[0]
        ems_scores = (ems_scores - min_scores) / (max_scores - min_scores + 1e-8)
        
        return ems_scores
    
    def compute_hierarchical_ems(
        self,
        features: Dict[str, torch.Tensor],
        reference_features: Optional[Dict[str, torch.Tensor]] = None,
        normal_text_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute EMS scores for all layers.
        
        Args:
            features: Dictionary of layer features
            reference_features: Optional dictionary of reference features for few-shot mode
            normal_text_embeddings: Optional text embeddings for zero-shot mode
            
        Returns:
            Dictionary of EMS scores for each layer
        """
        ems_scores = {}
        
        for layer_name, layer_features in features.items():
            if layer_name != 'cls_token':  # Skip CLS token
                layer_ref_features = reference_features.get(layer_name) if reference_features else None
                ems_scores[layer_name] = self.compute_layer_ems(
                    layer_features,
                    layer_ref_features,
                    normal_text_embeddings
                )
                
        return ems_scores 