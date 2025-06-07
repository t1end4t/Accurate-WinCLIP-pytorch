import pytest
import torch
from acas.ics import IntrinsicConsistencyScorer

class TestIntrinsicConsistencyScorer:
    @pytest.fixture
    def ics_scorer(self):
        """Create ICS scorer instance"""
        return IntrinsicConsistencyScorer(k_neighbors=5)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature tensor"""
        # Create random features for 2 batches, 196 patches, 768 dimensions
        return torch.randn(2, 196, 768)
    
    def test_initialization(self, ics_scorer):
        """Test ICS scorer initialization"""
        assert ics_scorer.k_neighbors == 5
    
    def test_similarity_matrix_shape(self, ics_scorer, sample_features):
        """Test similarity matrix computation"""
        # Flatten features for similarity computation
        features_flat = sample_features.reshape(-1, 768)
        similarity_matrix = ics_scorer.compute_similarity_matrix(features_flat)
        
        # Check shape: [N, N] where N = 2 * 196 = 392
        assert similarity_matrix.shape == (392, 392)
        
        # Check symmetry with appropriate tolerance
        assert torch.allclose(similarity_matrix, similarity_matrix.t(), rtol=1e-5, atol=1e-5)
        
        # Check diagonal is 1.0 (self-similarity)
        assert torch.allclose(torch.diag(similarity_matrix), torch.ones(392), rtol=1e-5, atol=1e-5)
    
    def test_ics_scores_shape(self, ics_scorer, sample_features):
        """Test ICS scores computation"""
        features_flat = sample_features.reshape(-1, 768)
        ics_scores = ics_scorer.compute_ics_scores(features_flat)
        
        # Check shape: [N] where N = 2 * 196 = 392
        assert ics_scores.shape == (392,)
        
        # Check scores are between 0 and 1
        assert torch.all(ics_scores >= 0) and torch.all(ics_scores <= 1)
    
    def test_layer_ics_shape(self, ics_scorer, sample_features):
        """Test layer ICS computation"""
        ics_scores = ics_scorer.compute_layer_ics(sample_features)
        
        # Check shape: [B, N] where B=2, N=196
        assert ics_scores.shape == (2, 196)
        
        # Check scores are between 0 and 1
        assert torch.all(ics_scores >= 0) and torch.all(ics_scores <= 1)
    
    def test_hierarchical_ics(self, ics_scorer):
        """Test hierarchical ICS computation"""
        # Create sample hierarchical features
        features = {
            'layer_3': torch.randn(2, 196, 768),
            'layer_6': torch.randn(2, 196, 768),
            'layer_9': torch.randn(2, 196, 768),
            'layer_12': torch.randn(2, 196, 768),
            'cls_token': torch.randn(2, 768)  # Should be ignored
        }
        
        ics_scores = ics_scorer.compute_hierarchical_ics(features)
        
        # Check all layers except cls_token are present
        assert set(ics_scores.keys()) == {'layer_3', 'layer_6', 'layer_9', 'layer_12'}
        
        # Check shapes for each layer
        for layer_name in ics_scores:
            assert ics_scores[layer_name].shape == (2, 196)
            assert torch.all(ics_scores[layer_name] >= 0) and torch.all(ics_scores[layer_name] <= 1)
    
    def test_device_placement(self, ics_scorer, sample_features):
        """Test device placement"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_features = sample_features.to(device)
        
        ics_scores = ics_scorer.compute_layer_ics(sample_features)
        assert ics_scores.device == device
    
    def test_consistency(self, ics_scorer, sample_features):
        """Test consistency across multiple computations"""
        ics_scores1 = ics_scorer.compute_layer_ics(sample_features)
        ics_scores2 = ics_scorer.compute_layer_ics(sample_features)
        
        assert torch.allclose(ics_scores1, ics_scores2) 