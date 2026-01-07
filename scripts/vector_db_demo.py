"""
Demo script to store and search embedded data using Qdrant vector database.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
import numpy as np

# Add parent directory to path to import root modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import EmbeddingGenerator, generate_sample_sentences
from qdrant_utils import QdrantManager

# Load environment variables
load_dotenv()


def load_embeddings_from_file(filepath: str = "embeddings.json"):
    """
    Load embeddings from JSON file.
    
    Args:
        filepath: Path to embeddings JSON file
        
    Returns:
        Tuple of (texts, embeddings)
    """
    import json
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        print("Please run 'python embeddings.py' first to generate embeddings.")
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    texts = data["texts"]
    embeddings = [np.array(emb) for emb in data["embeddings"]]
    
    print(f"Loaded {len(embeddings)} embeddings from {filepath}")
    return texts, embeddings


def store_embeddings_in_qdrant(collection_name: str,
                               texts: list,
                               embeddings: list,
                               recreate: bool = False,
                               qdrant_url: str = None,
                               qdrant_port: int = None):
    """
    Store embeddings in Qdrant vector database.
    
    Args:
        collection_name: Name of Qdrant collection
        texts: List of text strings
        embeddings: List of embedding vectors
        recreate: If True, recreate collection if it exists
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
    """
    if not embeddings:
        print("Error: No embeddings to store")
        return False
    
    vector_size = len(embeddings[0])
    
    # Initialize Qdrant manager
    try:
        manager = QdrantManager(url=qdrant_url, port=qdrant_port)
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)
    
    # Create collection
    success = manager.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=recreate
    )
    
    if not success:
        print("Failed to create collection")
        return False
    
    # Insert vectors
    print(f"\nInserting {len(embeddings)} vectors into Qdrant...")
    success = manager.insert_vectors(
        collection_name=collection_name,
        vectors=embeddings,
        texts=texts
    )
    
    if success:
        # Show collection info
        info = manager.get_collection_info(collection_name)
        if info:
            print(f"\nCollection Info:")
            print(f"  Name: {info['name']}")
            print(f"  Vectors: {info['vectors_count']}")
            print(f"  Vector Size: {info['config']['vector_size']}")
            print(f"  Distance: {info['config']['distance']}")
    
    return success


def search_in_qdrant(collection_name: str,
                    query_text: str,
                    top_k: int = 3,
                    provider: str = "openai",
                    qdrant_url: str = None,
                    qdrant_port: int = None):
    """
    Search for similar texts in Qdrant.
    
    Args:
        collection_name: Name of Qdrant collection
        query_text: Query text to search for
        top_k: Number of top results
        provider: Embedding provider for query
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
    """
    # Generate query embedding
    print(f"Generating embedding for query: '{query_text}'...")
    try:
        generator = EmbeddingGenerator(provider=provider)
        query_embedding = generator.generate_embedding(query_text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return
    
    # Initialize Qdrant manager
    try:
        manager = QdrantManager(url=qdrant_url, port=qdrant_port)
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return
    
    # Search
    print(f"Searching for top-{top_k} similar vectors in Qdrant...")
    results = manager.search_similar(
        collection_name=collection_name,
        query_vector=query_embedding,
        top_k=top_k
    )
    
    if not results:
        print("No results found")
        return
    
    # Display results
    print("\n" + "="*80)
    print(f"Query: {query_text}")
    print("="*80)
    print(f"\nTop {len(results)} Most Similar Results from Qdrant:\n")
    
    for i, result in enumerate(results, 1):
        text = result["payload"].get("text", "N/A")
        score = result["score"]
        point_id = result["id"]
        
        print(f"{i}. [Similarity: {score:.4f}] (ID: {point_id})")
        print(f"   {text}")
        print()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Store and search embedded data using Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store embeddings in Qdrant
  python vector_db_demo.py store
  
  # Search in Qdrant
  python vector_db_demo.py search "machine learning algorithms"
  
  # Store with custom collection name
  python vector_db_demo.py store --collection my_collection
  
  # Search with top-5 results
  python vector_db_demo.py search "data science" -k 5
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store embeddings in Qdrant")
    store_parser.add_argument(
        "--collection",
        type=str,
        default="embeddings",
        help="Collection name (default: embeddings)"
    )
    store_parser.add_argument(
        "--embeddings-file",
        type=str,
        default="embeddings.json",
        help="Path to embeddings JSON file (default: embeddings.json)"
    )
    store_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if it exists"
    )
    store_parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant server URL (default: localhost)"
    )
    store_parser.add_argument(
        "--qdrant-port",
        type=int,
        default=None,
        help="Qdrant server port (default: 6333)"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search in Qdrant")
    search_parser.add_argument(
        "query",
        type=str,
        help="Query text to search for"
    )
    search_parser.add_argument(
        "--collection",
        type=str,
        default="embeddings",
        help="Collection name (default: embeddings)"
    )
    search_parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Number of top results (default: 3)"
    )
    search_parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini", "mock"],
        default=None,
        help="Embedding provider (default: from EMBEDDING_PROVIDER env var)"
    )
    search_parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant server URL (default: localhost)"
    )
    search_parser.add_argument(
        "--qdrant-port",
        type=int,
        default=None,
        help="Qdrant server port (default: 6333)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Determine provider (only needed for search command)
    provider = getattr(args, 'provider', None) or os.getenv("EMBEDDING_PROVIDER", "openai")
    
    try:
        if args.command == "store":
            # Load embeddings
            texts, embeddings = load_embeddings_from_file(args.embeddings_file)
            
            # Store in Qdrant
            store_embeddings_in_qdrant(
                collection_name=args.collection,
                texts=texts,
                embeddings=embeddings,
                recreate=args.recreate,
                qdrant_url=args.qdrant_url,
                qdrant_port=args.qdrant_port
            )
        
        elif args.command == "search":
            # Search in Qdrant
            search_in_qdrant(
                collection_name=args.collection,
                query_text=args.query,
                top_k=args.top_k,
                provider=provider,
                qdrant_url=args.qdrant_url,
                qdrant_port=args.qdrant_port
            )
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

