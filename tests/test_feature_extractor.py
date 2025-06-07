import pytest
import torch
from acas.feature_extractor import HierarchicalFeatureExtractor

class TestHierarchicalFeatureExtractor:
    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor instance"""
        return HierarchicalFeatureExtractor(
            model_name="ViT-B-16",
            pretrained="laion400m_e31",
            layers_to_extract=[3, 6, 9, 12]
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image tensor"""
        return torch.randn(2, 3, 224, 224)
    
    def test_initialization(self, feature_extractor):
        """Test feature extractor initialization"""
        assert feature_extractor.layers_to_extract == [3, 6, 9, 12]
        assert feature_extractor.model is not None
        assert feature_extractor.preprocess is not None
        
        # Check if model is frozen
        for param in feature_extractor.model.parameters():
            assert not param.requires_grad
    
    def test_feature_extraction_shapes(self, feature_extractor, sample_image):
        """Test feature extraction output shapes"""
        features = feature_extractor.extract_features(sample_image)
        
        for layer_idx in feature_extractor.layers_to_extract:
            layer_features = features[f"layer_{layer_idx}"]
            # Expected shape: [B, N, D] where:
            # B = batch size (2)
            # N = number of patches (14x14 = 196 for ViT-B-16)
            # D = feature dimension (768 for ViT-B-16)
            assert layer_features.shape == (2, 196, 768), \
                f"Layer {layer_idx} features should have shape (2, 196, 768), got {layer_features.shape}"
    
    def test_feature_extraction_types(self, feature_extractor, sample_image):
        """Test feature extraction output types"""
        features = feature_extractor.extract_features(sample_image)
        
        for layer_idx in feature_extractor.layers_to_extract:
            layer_features = features[f"layer_{layer_idx}"]
            assert isinstance(layer_features, torch.Tensor), \
                f"Layer {layer_idx} features should be torch.Tensor, got {type(layer_features)}"
            assert layer_features.dtype == torch.float32, \
                f"Layer {layer_idx} features should be float32, got {layer_features.dtype}"
    
    def test_device_placement(self, feature_extractor, sample_image):
        """Test feature extraction device placement"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = feature_extractor.to(device)
        sample_image = sample_image.to(device)
        
        features = feature_extractor.extract_features(sample_image)
        
        for layer_idx in feature_extractor.layers_to_extract:
            layer_features = features[f"layer_{layer_idx}"]
            assert layer_features.device == device, \
                f"Layer {layer_idx} features should be on {device}, got {layer_features.device}"
    
    def test_feature_consistency(self, feature_extractor, sample_image):
        """Test feature extraction consistency across multiple forward passes"""
        features1 = feature_extractor.extract_features(sample_image)
        features2 = feature_extractor.extract_features(sample_image)
        
        for layer_idx in feature_extractor.layers_to_extract:
            layer_features1 = features1[f"layer_{layer_idx}"]
            layer_features2 = features2[f"layer_{layer_idx}"]
            assert torch.allclose(layer_features1, layer_features2), \
                f"Layer {layer_idx} features should be consistent across forward passes"
    
    def test_cls_token_extraction(self, feature_extractor, sample_image):
        """Test CLS token feature extraction"""
        features = feature_extractor.extract_features(sample_image, return_cls_token=True)
        
        assert "cls_token" in features, "CLS token features should be present"
        cls_token = features["cls_token"]
        assert cls_token.shape == (2, 768), \
            f"CLS token should have shape (2, 768), got {cls_token.shape}"
    
    def test_preprocess(self, feature_extractor):
        """Test preprocessing transforms"""
        preprocess = feature_extractor.get_preprocess()
        assert preprocess is not None, "Preprocessing transforms should be available" 