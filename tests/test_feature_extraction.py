import pytest
import torch
import torch.nn as nn
from src import open_clip
from PIL import Image
import numpy as np

class TestCLIPFeatureExtraction:
    @pytest.fixture
    def setup_clip_model(self):
        """Setup CLIP model and feature extractor"""
        model_name = "ViT-B-16"
        model, _, preprocess = open_clip.create_customer_model_and_transforms(
            model_name, pretrained="laion400m_e31"
        )
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad_(False)
        return model, preprocess

    @pytest.fixture
    def sample_image(self):
        """Create a sample image tensor"""
        # Create a random image tensor [B, C, H, W]
        return torch.randn(2, 3, 224, 224)

    def test_model_frozen(self, setup_clip_model):
        """Test that all model parameters are frozen"""
        model, _ = setup_clip_model
        for param in model.parameters():
            assert not param.requires_grad, "Model parameters should be frozen"

    def test_feature_extraction_shapes(self, setup_clip_model, sample_image):
        """Test feature extraction output shapes"""
        model, _ = setup_clip_model
        layers_to_extract = [3, 6, 9, 12]
        
        # Extract features
        features = {}
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for specified layers
        handles = []
        for layer_idx in layers_to_extract:
            handle = model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                hook_fn(f"layer_{layer_idx}")
            )
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            model(sample_image)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Verify shapes for each layer
        for layer_idx in layers_to_extract:
            layer_features = features[f"layer_{layer_idx}"]
            # Expected shape: [B, N, D] where:
            # B = batch size (2)
            # N = number of patches (14x14 = 196 for ViT-B-16)
            # D = feature dimension (768 for ViT-B-16)
            assert layer_features.shape == (2, 196, 768), \
                f"Layer {layer_idx} features should have shape (2, 196, 768), got {layer_features.shape}"

    def test_feature_extraction_types(self, setup_clip_model, sample_image):
        """Test feature extraction output types"""
        model, _ = setup_clip_model
        layers_to_extract = [3, 6, 9, 12]
        
        # Extract features
        features = {}
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for specified layers
        handles = []
        for layer_idx in layers_to_extract:
            handle = model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                hook_fn(f"layer_{layer_idx}")
            )
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            model(sample_image)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Verify types for each layer
        for layer_idx in layers_to_extract:
            layer_features = features[f"layer_{layer_idx}"]
            assert isinstance(layer_features, torch.Tensor), \
                f"Layer {layer_idx} features should be torch.Tensor, got {type(layer_features)}"
            assert layer_features.dtype == torch.float32, \
                f"Layer {layer_idx} features should be float32, got {layer_features.dtype}"

    def test_feature_extraction_device(self, setup_clip_model, sample_image):
        """Test feature extraction device placement"""
        model, _ = setup_clip_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        sample_image = sample_image.to(device)
        
        layers_to_extract = [3, 6, 9, 12]
        
        # Extract features
        features = {}
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for specified layers
        handles = []
        for layer_idx in layers_to_extract:
            handle = model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                hook_fn(f"layer_{layer_idx}")
            )
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            model(sample_image)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Verify device placement for each layer
        for layer_idx in layers_to_extract:
            layer_features = features[f"layer_{layer_idx}"]
            assert layer_features.device == device, \
                f"Layer {layer_idx} features should be on {device}, got {layer_features.device}"

    def test_feature_extraction_consistency(self, setup_clip_model, sample_image):
        """Test feature extraction consistency across multiple forward passes"""
        model, _ = setup_clip_model
        layers_to_extract = [3, 6, 9, 12]
        
        # First forward pass
        features1 = {}
        def hook_fn1(name):
            def hook(module, input, output):
                features1[name] = output
            return hook
        
        handles1 = []
        for layer_idx in layers_to_extract:
            handle = model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                hook_fn1(f"layer_{layer_idx}")
            )
            handles1.append(handle)
        
        with torch.no_grad():
            model(sample_image)
        
        for handle in handles1:
            handle.remove()
        
        # Second forward pass
        features2 = {}
        def hook_fn2(name):
            def hook(module, input, output):
                features2[name] = output
            return hook
        
        handles2 = []
        for layer_idx in layers_to_extract:
            handle = model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                hook_fn2(f"layer_{layer_idx}")
            )
            handles2.append(handle)
        
        with torch.no_grad():
            model(sample_image)
        
        for handle in handles2:
            handle.remove()
        
        # Verify consistency
        for layer_idx in layers_to_extract:
            layer_features1 = features1[f"layer_{layer_idx}"]
            layer_features2 = features2[f"layer_{layer_idx}"]
            assert torch.allclose(layer_features1, layer_features2), \
                f"Layer {layer_idx} features should be consistent across forward passes" 