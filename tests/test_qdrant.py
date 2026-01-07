"""
Unit tests for Qdrant vector database operations.
"""

import os
import sys

# Add parent directory to path to import root modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from qdrant_utils import QdrantManager
from qdrant_client.models import Distance

# Check if Qdrant is available
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Test collection name
TEST_COLLECTION = "test_collection"


@pytest.fixture(scope="module")
def qdrant_manager():
    """Create Qdrant manager for testing."""
    if not QDRANT_AVAILABLE:
        pytest.skip("Qdrant client not available")
    
    try:
        manager = QdrantManager(url="localhost", port=6333)
        # Test connection
        manager.list_collections()
        return manager
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


@pytest.fixture(autouse=True)
def cleanup_collection(qdrant_manager):
    """Clean up test collection before and after each test."""
    # Delete collection if it exists
    try:
        qdrant_manager.delete_collection(TEST_COLLECTION)
    except:
        pass
    
    yield
    
    # Clean up after test
    try:
        qdrant_manager.delete_collection(TEST_COLLECTION)
    except:
        pass


class TestQdrantCollection:
    """Test cases for collection creation."""
    
    def test_create_collection(self, qdrant_manager):
        """Test creating a new collection."""
        success = qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=128,
            distance=Distance.COSINE
        )
        assert success is True
        
        # Verify collection exists
        collections = qdrant_manager.list_collections()
        assert TEST_COLLECTION in collections
    
    def test_create_collection_with_recreate(self, qdrant_manager):
        """Test creating collection with recreate flag."""
        # Create collection first
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=128
        )
        
        # Recreate it
        success = qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=128,
            recreate=True
        )
        assert success is True
    
    def test_create_collection_duplicate(self, qdrant_manager):
        """Test that creating duplicate collection doesn't fail."""
        # Create collection
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=128
        )
        
        # Try to create again (should not fail, just skip)
        success = qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=128
        )
        assert success is True
    
    def test_get_collection_info(self, qdrant_manager):
        """Test getting collection information."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=256,
            distance=Distance.EUCLID
        )
        
        info = qdrant_manager.get_collection_info(TEST_COLLECTION)
        assert info is not None
        assert info["name"] == TEST_COLLECTION
        assert info["config"]["vector_size"] == 256
        assert info["config"]["distance"] == "EUCLID"


class TestQdrantInsert:
    """Test cases for vector insertion."""
    
    def test_insert_single_vector(self, qdrant_manager):
        """Test inserting a single vector."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=10
        )
        
        vector = np.random.rand(10).astype(np.float32)
        text = "Test sentence"
        
        success = qdrant_manager.insert_vectors(
            collection_name=TEST_COLLECTION,
            vectors=[vector],
            texts=[text]
        )
        assert success is True
        
        # Verify insertion
        info = qdrant_manager.get_collection_info(TEST_COLLECTION)
        assert info["vectors_count"] == 1
    
    def test_insert_multiple_vectors(self, qdrant_manager):
        """Test inserting multiple vectors."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=5
        )
        
        vectors = [np.random.rand(5).astype(np.float32) for _ in range(10)]
        texts = [f"Text {i}" for i in range(10)]
        
        success = qdrant_manager.insert_vectors(
            collection_name=TEST_COLLECTION,
            vectors=vectors,
            texts=texts
        )
        assert success is True
        
        # Verify insertion
        info = qdrant_manager.get_collection_info(TEST_COLLECTION)
        assert info["vectors_count"] == 10
    
    def test_insert_with_custom_ids(self, qdrant_manager):
        """Test inserting vectors with custom IDs."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=8
        )
        
        vectors = [np.random.rand(8).astype(np.float32) for _ in range(5)]
        texts = [f"Text {i}" for i in range(5)]
        ids = [100, 200, 300, 400, 500]
        
        success = qdrant_manager.insert_vectors(
            collection_name=TEST_COLLECTION,
            vectors=vectors,
            texts=texts,
            ids=ids
        )
        assert success is True
        
        # Verify insertion
        info = qdrant_manager.get_collection_info(TEST_COLLECTION)
        assert info["vectors_count"] == 5
    
    def test_insert_mismatched_lengths(self, qdrant_manager):
        """Test that mismatched vector and text lengths raise error."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=5
        )
        
        vectors = [np.random.rand(5).astype(np.float32) for _ in range(3)]
        texts = ["Text 1", "Text 2"]  # Different length
        
        with pytest.raises(ValueError):
            qdrant_manager.insert_vectors(
                collection_name=TEST_COLLECTION,
                vectors=vectors,
                texts=texts
            )


class TestQdrantSearch:
    """Test cases for similarity search."""
    
    def test_search_similar_vectors(self, qdrant_manager):
        """Test searching for similar vectors."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=10
        )
        
        # Insert test vectors
        vectors = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        texts = ["Vector pointing in X direction", 
                 "Vector pointing in Y direction",
                 "Vector at 45 degrees"]
        
        qdrant_manager.insert_vectors(
            collection_name=TEST_COLLECTION,
            vectors=vectors,
            texts=texts
        )
        
        # Search with query vector similar to first vector
        query_vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        results = qdrant_manager.search_similar(
            collection_name=TEST_COLLECTION,
            query_vector=query_vector,
            top_k=2
        )
        
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]  # Should be sorted
        assert "text" in results[0]["payload"]
    
    def test_search_top_k(self, qdrant_manager):
        """Test that search returns correct number of results."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=5
        )
        
        # Insert 10 vectors
        vectors = [np.random.rand(5).astype(np.float32) for _ in range(10)]
        texts = [f"Text {i}" for i in range(10)]
        
        qdrant_manager.insert_vectors(
            collection_name=TEST_COLLECTION,
            vectors=vectors,
            texts=texts
        )
        
        # Search with top_k=3
        query_vector = np.random.rand(5).astype(np.float32)
        results = qdrant_manager.search_similar(
            collection_name=TEST_COLLECTION,
            query_vector=query_vector,
            top_k=3
        )
        
        assert len(results) == 3
    
    def test_search_empty_collection(self, qdrant_manager):
        """Test searching in empty collection."""
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=5
        )
        
        query_vector = np.random.rand(5).astype(np.float32)
        results = qdrant_manager.search_similar(
            collection_name=TEST_COLLECTION,
            query_vector=query_vector,
            top_k=3
        )
        
        assert len(results) == 0


class TestQdrantDelete:
    """Test cases for collection deletion."""
    
    def test_delete_collection(self, qdrant_manager):
        """Test deleting a collection."""
        # Create collection
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=10
        )
        
        # Verify it exists
        collections = qdrant_manager.list_collections()
        assert TEST_COLLECTION in collections
        
        # Delete it
        success = qdrant_manager.delete_collection(TEST_COLLECTION)
        assert success is True
        
        # Verify it's deleted
        collections = qdrant_manager.list_collections()
        assert TEST_COLLECTION not in collections


class TestQdrantList:
    """Test cases for listing collections."""
    
    def test_list_collections(self, qdrant_manager):
        """Test listing all collections."""
        # Create a test collection
        qdrant_manager.create_collection(
            collection_name=TEST_COLLECTION,
            vector_size=10
        )
        
        collections = qdrant_manager.list_collections()
        assert isinstance(collections, list)
        assert TEST_COLLECTION in collections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

