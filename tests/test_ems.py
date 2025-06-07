import pytest
import torch
from acas.ems import ExtrinsicMatchingScorer

class TestExtrinsicMatchingScorer:
    @pytest.fixture
    def zero_shot_scorer(self):
        """Create zero-shot EMS scorer instance"""
        return ExtrinsicMatchingScorer(mode="zero_shot")
    
    @pytest.fixture
    def few_shot_scorer(self):
        """Create few-shot EMS scorer instance"""
        return ExtrinsicMatchingScorer(mode="few_shot")
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature tensor"""
        # Create random features for 2 batches, 196 patches, 768 dimensions
        return torch.randn(2, 196, 768)
    
    @pytest.fixture
    def sample_text_embeddings(self):
        """Create sample text embeddings"""
        # Create random text embeddings for 10 prompts
        return torch.randn(10, 768)
    
    @pytest.fixture
    def sample_reference_features(self):
        """Create sample reference features"""
        # Create random reference features for 5 shots, 196 patches, 768 dimensions
        return torch.randn(5, 196, 768)
    
    def test_initialization(self, zero_shot_scorer, few_shot_scorer):
        """Test EMS scorer initialization"""
        assert zero_shot_scorer.mode == "zero_shot"
        assert few_shot_scorer.mode == "few_shot"
        
        with pytest.raises(AssertionError):
            ExtrinsicMatchingScorer(mode="invalid_mode")
    
    def test_similarity_computation(self, zero_shot_scorer, sample_features, sample_text_embeddings):
        """Test similarity computation"""
        features_flat = sample_features.reshape(-1, 768)
        similarity = zero_shot_scorer.compute_similarity(features_flat, sample_text_embeddings)
        
        # Check shape: [N, T] where N=392 (2*196), T=10
        assert similarity.shape == (392, 10)
        
        # Check values are between -1 and 1 (cosine similarity)
        assert torch.all(similarity >= -1) and torch.all(similarity <= 1)
    
    def test_zero_shot_ems(self, zero_shot_scorer, sample_features, sample_text_embeddings):
        """Test zero-shot EMS computation"""
        features_flat = sample_features.reshape(-1, 768)
        ems_scores = zero_shot_scorer.compute_zero_shot_ems(features_flat, sample_text_embeddings)
        
        # Check shape: [N] where N=392 (2*196)
        assert ems_scores.shape == (392,)
        
        # Check scores are between 0 and 1
        assert torch.all(ems_scores >= 0) and torch.all(ems_scores <= 1)
    
    def test_few_shot_ems(self, few_shot_scorer, sample_features, sample_reference_features):
        """Test few-shot EMS computation"""
        features_flat = sample_features.reshape(-1, 768)
        ref_features_flat = sample_reference_features.reshape(-1, 768)
        ems_scores = few_shot_scorer.compute_few_shot_ems(features_flat, ref_features_flat)
        
        # Check shape: [N] where N=392 (2*196)
        assert ems_scores.shape == (392,)
        
        # Check scores are between 0 and 1
        assert torch.all(ems_scores >= 0) and torch.all(ems_scores <= 1)
    
    def test_layer_ems_zero_shot(self, zero_shot_scorer, sample_features, sample_text_embeddings):
        """Test layer EMS computation in zero-shot mode"""
        ems_scores = zero_shot_scorer.compute_layer_ems(
            sample_features,
            normal_text_embeddings=sample_text_embeddings
        )
        
        # Check shape: [B, N] where B=2, N=196
        assert ems_scores.shape == (2, 196)
        
        # Check scores are between 0 and 1
        assert torch.all(ems_scores >= 0) and torch.all(ems_scores <= 1)
    
    def test_layer_ems_few_shot(self, few_shot_scorer, sample_features, sample_reference_features):
        """Test layer EMS computation in few-shot mode"""
        ems_scores = few_shot_scorer.compute_layer_ems(
            sample_features,
            reference_features=sample_reference_features
        )
        
        # Check shape: [B, N] where B=2, N=196
        assert ems_scores.shape == (2, 196)
        
        # Check scores are between 0 and 1
        assert torch.all(ems_scores >= 0) and torch.all(ems_scores <= 1)
    
    def test_hierarchical_ems(self, zero_shot_scorer, sample_text_embeddings):
        """Test hierarchical EMS computation"""
        # Create sample hierarchical features
        features = {
            'layer_3': torch.randn(2, 196, 768),
            'layer_6': torch.randn(2, 196, 768),
            'layer_9': torch.randn(2, 196, 768),
            'layer_12': torch.randn(2, 196, 768),
            'cls_token': torch.randn(2, 768)  # Should be ignored
        }
        
        ems_scores = zero_shot_scorer.compute_hierarchical_ems(
            features,
            normal_text_embeddings=sample_text_embeddings
        )
        
        # Check all layers except cls_token are present
        assert set(ems_scores.keys()) == {'layer_3', 'layer_6', 'layer_9', 'layer_12'}
        
        # Check shapes for each layer
        for layer_name in ems_scores:
            assert ems_scores[layer_name].shape == (2, 196)
            assert torch.all(ems_scores[layer_name] >= 0) and torch.all(ems_scores[layer_name] <= 1)
    
    def test_device_placement(self, zero_shot_scorer, sample_features, sample_text_embeddings):
        """Test device placement"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_features = sample_features.to(device)
        sample_text_embeddings = sample_text_embeddings.to(device)
        
        ems_scores = zero_shot_scorer.compute_layer_ems(
            sample_features,
            normal_text_embeddings=sample_text_embeddings
        )
        assert ems_scores.device == device
    
    def test_consistency(self, zero_shot_scorer, sample_features, sample_text_embeddings):
        """Test consistency across multiple computations"""
        ems_scores1 = zero_shot_scorer.compute_layer_ems(
            sample_features,
            normal_text_embeddings=sample_text_embeddings
        )
        ems_scores2 = zero_shot_scorer.compute_layer_ems(
            sample_features,
            normal_text_embeddings=sample_text_embeddings
        )
        
        assert torch.allclose(ems_scores1, ems_scores2) 