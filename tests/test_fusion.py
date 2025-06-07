import pytest
import torch
from acas.fusion import AdaptiveFusion

class TestAdaptiveFusion:
    @pytest.fixture
    def fusion(self):
        """Create fusion instance"""
        return AdaptiveFusion(temperature=1.0)
    
    @pytest.fixture
    def sample_scores(self):
        """Create sample score dictionaries"""
        ics_scores = {
            'layer_3': torch.randn(2, 196),
            'layer_6': torch.randn(2, 196),
            'layer_9': torch.randn(2, 196),
            'layer_12': torch.randn(2, 196)
        }
        ems_scores = {
            'layer_3': torch.randn(2, 196),
            'layer_6': torch.randn(2, 196),
            'layer_9': torch.randn(2, 196),
            'layer_12': torch.randn(2, 196)
        }
        return ics_scores, ems_scores
    
    def test_initialization(self, fusion):
        """Test fusion initialization"""
        assert fusion.temperature == 1.0
        
        # Test different temperature
        fusion = AdaptiveFusion(temperature=0.5)
        assert fusion.temperature == 0.5
    
    def test_fuse_scores(self, fusion, sample_scores):
        """Test score fusion"""
        ics_scores, ems_scores = sample_scores
        fused_scores = fusion.fuse_scores(ics_scores, ems_scores)
        
        # Check all layers are present
        assert set(fused_scores.keys()) == {'layer_3', 'layer_6', 'layer_9', 'layer_12'}
        
        # Check shapes for each layer
        for layer_name in fused_scores:
            assert fused_scores[layer_name].shape == (2, 196)
            
        # Check fusion is correct (ICS + EMS)
        for layer_name in fused_scores:
            expected = ics_scores[layer_name] + ems_scores[layer_name]
            assert torch.allclose(fused_scores[layer_name], expected)
    
    def test_compute_weights(self, fusion):
        """Test weight computation"""
        # Create sample scores for 4 layers
        scores = torch.randn(4, 196)
        weights = fusion.compute_weights(scores)
        
        # Check shape
        assert weights.shape == (4, 196)
        
        # Check weights sum to 1 for each patch
        assert torch.allclose(weights.sum(dim=0), torch.ones(196))
        
        # Check weights are between 0 and 1
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
    
    def test_fuse_hierarchical_scores(self, fusion, sample_scores):
        """Test hierarchical score fusion"""
        ics_scores, ems_scores = sample_scores
        final_scores = fusion.fuse_hierarchical_scores(ics_scores, ems_scores)
        
        # Check shape: [B, N] where B=2, N=196
        assert final_scores.shape == (2, 196)
        
        # Check scores are between 0 and 1
        assert torch.all(final_scores >= 0) and torch.all(final_scores <= 1)
    
    def test_temperature_effect(self):
        """Test effect of temperature on fusion"""
        # Create fusion with different temperatures
        fusion_low_temp = AdaptiveFusion(temperature=0.1)
        fusion_high_temp = AdaptiveFusion(temperature=10.0)
        
        # Create sample scores
        scores = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        
        # Compute weights
        weights_low = fusion_low_temp.compute_weights(scores)
        weights_high = fusion_high_temp.compute_weights(scores)
        
        # Low temperature should make weights more extreme
        assert torch.std(weights_low) > torch.std(weights_high)
    
    def test_device_placement(self, fusion, sample_scores):
        """Test device placement"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ics_scores, ems_scores = sample_scores
        
        # Move scores to device
        ics_scores = {k: v.to(device) for k, v in ics_scores.items()}
        ems_scores = {k: v.to(device) for k, v in ems_scores.items()}
        
        # Move fusion to device
        fusion = fusion.to(device)
        
        # Compute final scores
        final_scores = fusion.fuse_hierarchical_scores(ics_scores, ems_scores)
        assert final_scores.device == device
    
    def test_consistency(self, fusion, sample_scores):
        """Test consistency across multiple computations"""
        ics_scores, ems_scores = sample_scores
        
        final_scores1 = fusion.fuse_hierarchical_scores(ics_scores, ems_scores)
        final_scores2 = fusion.fuse_hierarchical_scores(ics_scores, ems_scores)
        
        assert torch.allclose(final_scores1, final_scores2) 