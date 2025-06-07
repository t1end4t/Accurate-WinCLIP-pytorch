import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

class AdaptiveFusion:
    """Implements Fusion Method B (Softmax Weighting) for anomaly detection."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize the Adaptive Fusion.
        
        Args:
            temperature: Temperature parameter for softmax weighting
        """
        self.temperature = temperature
        
    def fuse_scores(
        self,
        ics_scores: Dict[str, torch.Tensor],
        ems_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse ICS and EMS scores using softmax weighting.
        
        Args:
            ics_scores: Dictionary of ICS scores for each layer
            ems_scores: Dictionary of EMS scores for each layer
            
        Returns:
            Dictionary of fused scores for each layer
        """
        fused_scores = {}
        
        for layer_name in ics_scores.keys():
            # Combine ICS and EMS scores
            combined_scores = ics_scores[layer_name] + ems_scores[layer_name]
            fused_scores[layer_name] = combined_scores
            
        return fused_scores
    
    def compute_weights(
        self,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute softmax weights for scores.
        
        Args:
            scores: Score tensor of shape [L, N] where L is number of layers
            
        Returns:
            Weight tensor of shape [L, N]
        """
        # Scale scores by temperature
        scaled_scores = scores / self.temperature
        
        # Compute softmax weights
        weights = F.softmax(scaled_scores, dim=0)
        
        return weights
    
    def fuse_hierarchical_scores(
        self,
        ics_scores: Dict[str, torch.Tensor],
        ems_scores: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse hierarchical scores using adaptive fusion.
        
        Args:
            ics_scores: Dictionary of ICS scores for each layer
            ems_scores: Dictionary of EMS scores for each layer
            
        Returns:
            Final fused scores of shape [B, N]
        """
        # Get fused scores for each layer
        fused_scores = self.fuse_scores(ics_scores, ems_scores)
        
        # Stack scores from all layers
        stacked_scores = torch.stack([scores for scores in fused_scores.values()], dim=0)
        
        # Compute weights for each patch
        weights = self.compute_weights(stacked_scores)
        
        # Compute weighted sum
        final_scores = torch.sum(weights * stacked_scores, dim=0)
        
        # Normalize scores to [0, 1] range using min-max normalization
        min_scores = torch.min(final_scores, dim=1, keepdim=True)[0]
        max_scores = torch.max(final_scores, dim=1, keepdim=True)[0]
        normalized_scores = (final_scores - min_scores) / (max_scores - min_scores + 1e-8)
        
        return normalized_scores
    
    def to(self, device: torch.device) -> 'AdaptiveFusion':
        """Move to specified device."""
        return self 