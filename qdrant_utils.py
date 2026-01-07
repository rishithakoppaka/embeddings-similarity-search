"""
Qdrant vector database utilities.
Functions to create collection, insert vectors, and perform similarity search.
"""

import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Load environment variables
load_dotenv()


class QdrantManager:
    """Manager class for Qdrant vector database operations."""
    
    def __init__(self, 
                 url: Optional[str] = None,
                 port: Optional[int] = None,
                 api_key: Optional[str] = None,
                 prefer_grpc: bool = False):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL (default: localhost)
            port: Qdrant server port (default: 6333 for HTTP, 6334 for gRPC)
            api_key: API key for Qdrant Cloud (optional)
            prefer_grpc: Use gRPC instead of HTTP (default: False)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not installed. Install with: pip install qdrant-client"
            )
        
        # Default to local Qdrant
        self.url = url or os.getenv("QDRANT_URL", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize client
        if api_key:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=self.url,
                api_key=api_key,
                prefer_grpc=prefer_grpc
            )
        else:
            # Local Qdrant
            self.client = QdrantClient(
                host=self.url,
                port=self.port,
                prefer_grpc=prefer_grpc
            )
        
        print(f"Connected to Qdrant at {self.url}:{self.port}")
    
    def create_collection(self,
                        collection_name: str,
                        vector_size: int,
                        distance: Distance = Distance.COSINE,
                        recreate: bool = False) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete existing collection and create new one
            
        Returns:
            True if collection was created successfully
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if recreate:
                    print(f"Deleting existing collection '{collection_name}'...")
                    self.client.delete_collection(collection_name)
                else:
                    print(f"Collection '{collection_name}' already exists. Skipping creation.")
                    return True
            
            # Create collection
            print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            print(f"Collection '{collection_name}' created successfully!")
            return True
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def insert_vectors(self,
                      collection_name: str,
                      vectors: List[np.ndarray],
                      texts: List[str],
                      ids: Optional[List[int]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      batch_size: int = 100) -> bool:
        """
        Insert vectors into Qdrant collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors (numpy arrays)
            texts: List of text strings corresponding to vectors
            ids: Optional list of point IDs (auto-generated if None)
            metadata: Optional list of metadata dictionaries
            batch_size: Number of points to insert per batch
            
        Returns:
            True if insertion was successful
        """
        # Validate inputs first (raise errors immediately, don't catch)
        if len(vectors) != len(texts):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match "
                f"number of texts ({len(texts)})"
            )
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = list(range(len(vectors)))
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"text": text} for text in texts]
            else:
                # Add text to metadata if not present
                for i, meta in enumerate(metadata):
                    if "text" not in meta:
                        meta["text"] = texts[i]
            
            # Convert numpy arrays to lists
            vector_list = [vec.tolist() if isinstance(vec, np.ndarray) else vec 
                          for vec in vectors]
            
            # Insert in batches
            total_points = len(vectors)
            for i in range(0, total_points, batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vector_list[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                points = [
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=meta
                    )
                    for point_id, vector, meta in zip(batch_ids, batch_vectors, batch_metadata)
                ]
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                print(f"Inserted batch: {min(i + batch_size, total_points)}/{total_points} points")
            
            print(f"Successfully inserted {total_points} vectors into '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error inserting vectors: {e}")
            return False
    
    def search_similar(self,
                      collection_name: str,
                      query_vector: np.ndarray,
                      top_k: int = 3,
                      score_threshold: Optional[float] = None,
                      filter_condition: Optional[Filter] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            filter_condition: Optional filter condition
            
        Returns:
            List of search results with id, score, and payload
        """
        try:
            # Convert numpy array to list
            query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            
            # Perform search using search method
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results (search() returns a list of ScoredPoint objects)
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        try:
            # Try the standard approach first
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.points_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name
                }
            }
        except Exception as e:
            # Handle Pydantic validation errors (version mismatch between client/server)
            # Use alternative approach: HTTP API directly + sample point
            try:
                # Get vector count using count API
                count_result = self.client.count(collection_name)
                points_count = count_result.count if hasattr(count_result, 'count') else 0
                
                # Try to infer vector size by getting a sample point
                vector_size = None
                distance = "COSINE"  # Default
                
                if points_count > 0:
                    # Get a single point to infer vector size
                    try:
                        scroll_result = self.client.scroll(
                            collection_name=collection_name,
                            limit=1,
                            with_vectors=True
                        )
                        if scroll_result[0] and len(scroll_result[0]) > 0:
                            point = scroll_result[0][0]
                            if hasattr(point, 'vector') and point.vector:
                                if isinstance(point.vector, list):
                                    vector_size = len(point.vector)
                                elif hasattr(point.vector, '__len__'):
                                    vector_size = len(point.vector)
                    except Exception:
                        pass
                
                # Try to get distance using HTTP API directly
                try:
                    import requests
                    # Use http for local, https for cloud (check if port is standard)
                    protocol = "https" if self.port == 443 or hasattr(self.client, '_client') and hasattr(self.client._client, 'api_key') else "http"
                    qdrant_url = f"{protocol}://{self.url}:{self.port}/collections/{collection_name}"
                    response = requests.get(qdrant_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        result = data.get('result', {})
                        config = result.get('config', {})
                        params = config.get('params', {})
                        vectors = params.get('vectors', {})
                        
                        if vectors.get('size'):
                            vector_size = vectors['size']
                        if vectors.get('distance'):
                            dist = vectors['distance']
                            # Normalize distance name
                            if isinstance(dist, str):
                                distance = dist.upper()
                            elif hasattr(dist, 'upper'):
                                distance = dist.upper()
                            else:
                                distance = str(dist).upper()
                            # Map common variations
                            if 'COSINE' in distance or distance == 'Cosine':
                                distance = 'COSINE'
                            elif 'EUCLID' in distance or distance == 'Euclid':
                                distance = 'EUCLID'
                            elif 'DOT' in distance:
                                distance = 'DOT'
                except Exception:
                    pass  # Fall back to defaults
                
                return {
                    "name": collection_name,
                    "vectors_count": points_count,
                    "config": {
                        "vector_size": vector_size,
                        "distance": distance
                    }
                }
            except Exception as inner_e:
                # If alternative approach also fails, return None
                print(f"Error getting collection info: {e}")
                return None
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted successfully")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []


def create_collection(collection_name: str,
                     vector_size: int,
                     distance: Distance = Distance.COSINE,
                     url: Optional[str] = None,
                     port: Optional[int] = None,
                     recreate: bool = False) -> bool:
    """
    Convenience function to create a Qdrant collection.
    
    Args:
        collection_name: Name of the collection
        vector_size: Dimension of the vectors
        distance: Distance metric
        url: Qdrant server URL
        port: Qdrant server port
        recreate: If True, recreate existing collection
        
    Returns:
        True if collection was created successfully
    """
    manager = QdrantManager(url=url, port=port)
    return manager.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        distance=distance,
        recreate=recreate
    )


def insert_vectors_to_qdrant(collection_name: str,
                             vectors: List[np.ndarray],
                             texts: List[str],
                             ids: Optional[List[int]] = None,
                             url: Optional[str] = None,
                             port: Optional[int] = None,
                             batch_size: int = 100) -> bool:
    """
    Convenience function to insert vectors into Qdrant.
    
    Args:
        collection_name: Name of the collection
        vectors: List of embedding vectors
        texts: List of text strings
        ids: Optional list of point IDs
        url: Qdrant server URL
        port: Qdrant server port
        batch_size: Batch size for insertion
        
    Returns:
        True if insertion was successful
    """
    manager = QdrantManager(url=url, port=port)
    return manager.insert_vectors(
        collection_name=collection_name,
        vectors=vectors,
        texts=texts,
        ids=ids,
        batch_size=batch_size
    )


def search_similar_in_qdrant(collection_name: str,
                             query_vector: np.ndarray,
                             top_k: int = 3,
                             url: Optional[str] = None,
                             port: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to search for similar vectors in Qdrant.
    
    Args:
        collection_name: Name of the collection
        query_vector: Query embedding vector
        top_k: Number of top results
        url: Qdrant server URL
        port: Qdrant server port
        
    Returns:
        List of search results
    """
    manager = QdrantManager(url=url, port=port)
    return manager.search_similar(
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k
    )

