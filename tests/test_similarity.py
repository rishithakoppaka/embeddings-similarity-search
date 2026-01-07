"""
Unit tests for cosine similarity implementation.
"""

import os
import sys

# Add parent directory to path to import root modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    find_top_k_similar,
    normalize_vector,
    normalize_vectors
)


class TestCosineSimilarity:
    """Test cases for cosine_similarity function."""
    
    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0"""
        vec = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0, abs=1e-6)
    
    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0.0"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(0.0, abs=1e-6)
    
    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1.0"""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(-1.0, abs=1e-6)
    
    def test_different_dimensions_raises_error(self):
        """Test that vectors with different dimensions raise ValueError"""
        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)
    
    def test_zero_vector(self):
        """Test that zero vectors return 0.0 similarity"""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(vec1, vec2)
        assert result == 0.0
    
    def test_positive_similarity(self):
        """Test that vectors pointing in similar direction have positive similarity"""
        vec1 = np.array([1.0, 1.0])
        vec2 = np.array([2.0, 2.0])  # Same direction, different magnitude
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0, abs=1e-6)
    
    def test_known_calculation(self):
        """Test with known values to verify calculation"""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.707, 0.707])  # 45 degrees
        result = cosine_similarity(vec1, vec2)
        # cos(45°) ≈ 0.707
        assert result == pytest.approx(0.707, abs=0.01)


class TestCosineSimilarityBatch:
    """Test cases for cosine_similarity_batch function."""
    
    def test_single_candidate(self):
        """Test with single candidate vector"""
        query = np.array([1.0, 0.0, 0.0])
        candidates = [np.array([1.0, 0.0, 0.0])]
        result = cosine_similarity_batch(query, candidates)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0, abs=1e-6)
    
    def test_multiple_candidates(self):
        """Test with multiple candidate vectors"""
        query = np.array([1.0, 0.0])
        candidates = [
            np.array([1.0, 0.0]),  # Same direction
            np.array([0.0, 1.0]),  # Orthogonal
            np.array([-1.0, 0.0])  # Opposite
        ]
        result = cosine_similarity_batch(query, candidates)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(-1.0, abs=1e-6)
    
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        query = np.array([1.0, 2.0, 3.0])
        candidates = []
        result = cosine_similarity_batch(query, candidates)
        assert len(result) == 0
    
    def test_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises ValueError"""
        query = np.array([1.0, 2.0])
        candidates = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(ValueError):
            cosine_similarity_batch(query, candidates)


class TestFindTopKSimilar:
    """Test cases for find_top_k_similar function."""
    
    def test_top_3_results(self):
        """Test finding top 3 similar texts"""
        query = np.array([1.0, 0.0])
        candidates = [
            np.array([0.0, 1.0]),      # Orthogonal (similarity = 0)
            np.array([1.0, 0.0]),      # Identical (similarity = 1)
            np.array([0.707, 0.707]),  # 45 degrees (similarity ≈ 0.707)
            np.array([-1.0, 0.0]),     # Opposite (similarity = -1)
        ]
        texts = ["text0", "text1", "text2", "text3"]
        
        results = find_top_k_similar(query, candidates, texts, k=3)
        
        assert len(results) == 3
        # Check that results are sorted by similarity (highest first)
        assert results[0][0] == "text1"  # Highest similarity
        assert results[0][1] == pytest.approx(1.0, abs=1e-6)
        assert results[1][0] == "text2"  # Second highest
        assert results[2][0] == "text0"  # Third highest
    
    def test_k_larger_than_candidates(self):
        """Test when k is larger than number of candidates"""
        query = np.array([1.0, 0.0])
        candidates = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        texts = ["text1", "text2"]
        
        results = find_top_k_similar(query, candidates, texts, k=5)
        
        assert len(results) == 2  # Should return only available candidates
    
    def test_k_zero_returns_empty(self):
        """Test that k=0 returns empty list"""
        query = np.array([1.0, 0.0])
        candidates = [np.array([1.0, 0.0])]
        texts = ["text1"]
        
        results = find_top_k_similar(query, candidates, texts, k=0)
        
        assert len(results) == 0
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched vector and text lengths raise ValueError"""
        query = np.array([1.0, 0.0])
        candidates = [np.array([1.0, 0.0])]
        texts = ["text1", "text2"]  # Different length
        
        with pytest.raises(ValueError):
            find_top_k_similar(query, candidates, texts, k=1)
    
    def test_results_include_original_index(self):
        """Test that results include original index"""
        query = np.array([1.0, 0.0])
        candidates = [
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
        ]
        texts = ["text0", "text1"]
        
        results = find_top_k_similar(query, candidates, texts, k=2)
        
        assert results[0][2] == 1  # Original index of top result
        assert results[1][2] == 0  # Original index of second result


class TestNormalizeVector:
    """Test cases for normalize_vector function."""
    
    def test_normalize_non_zero_vector(self):
        """Test normalizing a non-zero vector"""
        vec = np.array([3.0, 4.0])
        normalized = normalize_vector(vec)
        norm = np.linalg.norm(normalized)
        assert norm == pytest.approx(1.0, abs=1e-6)
    
    def test_normalize_zero_vector(self):
        """Test that zero vector remains unchanged"""
        vec = np.array([0.0, 0.0, 0.0])
        normalized = normalize_vector(vec)
        np.testing.assert_array_equal(normalized, vec)
    
    def test_normalize_preserves_direction(self):
        """Test that normalization preserves direction"""
        vec = np.array([2.0, 2.0])
        normalized = normalize_vector(vec)
        # Should point in same direction (all components have same sign and ratio)
        assert normalized[0] == pytest.approx(normalized[1], abs=1e-6)
        assert normalized[0] > 0


class TestNormalizeVectors:
    """Test cases for normalize_vectors function."""
    
    def test_normalize_multiple_vectors(self):
        """Test normalizing multiple vectors"""
        vectors = [
            np.array([3.0, 4.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([5.0, 0.0])
        ]
        normalized = normalize_vectors(vectors)
        
        assert len(normalized) == 3
        for vec in normalized:
            norm = np.linalg.norm(vec)
            assert norm == pytest.approx(1.0, abs=1e-6) or norm == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

