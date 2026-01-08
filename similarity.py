"""
Cosine similarity implementation using NumPy.
Provides functions to calculate similarity between embeddings.
"""

import numpy as np
from typing import List, Tuple


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity = (A · B) / (||A|| * ||B||)
    where · is dot product and || || is L2 norm
    
    Args:
        vec1: First vector (numpy array)
        vec2: Second vector (numpy array)
        
    Returns:
        Cosine similarity score between -1 and 1 (typically 0 to 1 for embeddings)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vectors must have same shape. Got {vec1.shape} and {vec2.shape}")
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate L2 norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)


def cosine_similarity_batch(query_vector: np.ndarray, 
                           candidate_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Calculate cosine similarity between a query vector and multiple candidate vectors.
    Uses vectorized operations for efficiency.
    
    Args:
        query_vector: Query embedding vector (numpy array)
        candidate_vectors: List of candidate embedding vectors
        
    Returns:
        Numpy array of similarity scores
    """
    if not candidate_vectors:
        return np.array([])
    
    # Convert list to numpy array (matrix)
    candidates_matrix = np.array(candidate_vectors)
    
    # Ensure query_vector is 1D and has correct shape
    query_vector = np.array(query_vector).flatten()
    
    # Check dimensions
    if query_vector.shape[0] != candidates_matrix.shape[1]:
        raise ValueError(
            f"Query vector dimension {query_vector.shape[0]} must match "
            f"candidate vector dimension {candidates_matrix.shape[1]}"
        )
    
    # Calculate dot products (vectorized)
    # Shape: (num_candidates,)
    dot_products = np.dot(candidates_matrix, query_vector)
    
    # Calculate L2 norms for query vector
    query_norm = np.linalg.norm(query_vector)
    
    # Calculate L2 norms for all candidate vectors (vectorized)
    # Shape: (num_candidates,)
    candidate_norms = np.linalg.norm(candidates_matrix, axis=1)
    
    # Handle division by zero
    denominator = query_norm * candidate_norms
    denominator[denominator == 0] = 1.0  # Avoid division by zero
    
    # Calculate cosine similarities (vectorized)
    similarities = dot_products / denominator
    
    return similarities


def find_top_k_similar(query_vector: np.ndarray,
                      candidate_vectors: List[np.ndarray],
                      candidate_texts: List[str],
                      k: int = 3) -> List[Tuple[str, float, int]]:
    """
    Find top-k most similar texts to a query vector.
    
    Args:
        query_vector: Query embedding vector
        candidate_vectors: List of candidate embedding vectors
        candidate_texts: List of candidate text strings (same order as vectors)
        k: Number of top results to return
        
    Returns:
        List of tuples: (text, similarity_score, original_index)
        Sorted by similarity (highest first)
    """
    if len(candidate_vectors) != len(candidate_texts):
        raise ValueError(
            f"Number of vectors ({len(candidate_vectors)}) must match "
            f"number of texts ({len(candidate_texts)})"
        )
    
    if k <= 0:
        return []
    
    # Calculate similarities for all candidates
    similarities = cosine_similarity_batch(query_vector, candidate_vectors)
    
    # Get all indices sorted by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Build results list, removing duplicates
    results = []
    seen_texts = set()
    
    for idx in sorted_indices:
        text = candidate_texts[idx]
        score = float(similarities[idx])
        
        # Skip if we've already seen this exact text
        if text not in seen_texts:
            results.append((text, score, int(idx)))
            seen_texts.add(text)
            
            # Stop when we have k unique results
            if len(results) >= k:
                break
    
    return results


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length (L2 norm = 1).
    This can speed up cosine similarity calculations.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def normalize_vectors(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize multiple vectors to unit length.
    
    Args:
        vectors: List of vectors to normalize
        
    Returns:
        List of normalized vectors
    """
    return [normalize_vector(vec) for vec in vectors]

