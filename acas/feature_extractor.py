import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from src import open_clip

class HierarchicalFeatureExtractor:
    """Extracts hierarchical features from CLIP ViT model."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion400m_e31",
        layers_to_extract: List[int] = [3, 6, 9, 12]
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the CLIP model to use
            pretrained: Pretrained weights to use
            layers_to_extract: List of layer indices to extract features from
        """
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.layers_to_extract = layers_to_extract
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
            
        self.model.eval()
        
    def to(self, device: torch.device) -> 'HierarchicalFeatureExtractor':
        """Move model to specified device."""
        self.model = self.model.to(device)
        return self
        
    def extract_features(
        self,
        images: torch.Tensor,
        return_cls_token: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features from input images.
        
        Args:
            images: Input image tensor of shape [B, C, H, W]
            return_cls_token: Whether to return CLS token features
            
        Returns:
            Dictionary containing features for each layer:
            {
                'layer_3': tensor[B, N, D],
                'layer_6': tensor[B, N, D],
                ...
            }
            If return_cls_token is True, also includes:
            {
                'cls_token': tensor[B, D]
            }
        """
        features = {}
        
        def hook_fn(name: str):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for specified layers
        handles = []
        for layer_idx in self.layers_to_extract:
            handle = self.model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                hook_fn(f"layer_{layer_idx}")
            )
            handles.append(handle)
            
        # Forward pass
        with torch.no_grad():
            self.model(images)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return features
    
    def get_preprocess(self):
        """Get the preprocessing transforms."""
        return self.preprocess 