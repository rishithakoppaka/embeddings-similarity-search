"""
CLI script to demonstrate similarity search.
Compares a query sentence to existing embeddings and returns top-3 most similar sentences.
"""

import os
import sys
import argparse
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import numpy as np

# Add parent directory to path to import root modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import EmbeddingGenerator, generate_sample_sentences
from similarity import find_top_k_similar

# Load environment variables
load_dotenv()


def load_or_generate_embeddings(embeddings_file: str = "embeddings.json", 
                                provider: str = "openai",
                                force_regenerate: bool = False) -> Tuple[List[str], List[np.ndarray]]:
    """
    Load embeddings from file or generate new ones if file doesn't exist.
    
    Args:
        embeddings_file: Path to embeddings JSON file
        provider: Embedding provider ("openai" or "gemini")
        force_regenerate: If True, regenerate embeddings even if file exists
        
    Returns:
        Tuple of (texts, embeddings)
    """
    generator = EmbeddingGenerator(provider=provider)
    
    # Check if embeddings file exists
    if os.path.exists(embeddings_file) and not force_regenerate:
        print(f"Loading embeddings from {embeddings_file}...")
        try:
            texts, embeddings = generator.load_embeddings(embeddings_file)
            return texts, embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("Generating new embeddings...")
    
    # Generate new embeddings
    print("Generating 510 sample sentences...")
    sentences = generate_sample_sentences(510)
    
    print(f"Generating embeddings using {provider}...")
    embeddings = generator.generate_embeddings_batch(sentences)
    
    print(f"Saving embeddings to {embeddings_file}...")
    generator.save_embeddings(sentences, embeddings, embeddings_file)
    
    return sentences, embeddings


def search_similar(query_text: str,
                  texts: List[str],
                  embeddings: List[np.ndarray],
                  query_embedding: Optional[np.ndarray] = None,
                  provider: str = "openai",
                  k: int = 3) -> List[Tuple[str, float, int]]:
    """
    Search for top-k most similar sentences to a query.
    
    Args:
        query_text: Query sentence
        texts: List of candidate texts
        embeddings: List of candidate embeddings
        query_embedding: Pre-computed query embedding (optional)
        provider: Embedding provider for generating query embedding
        k: Number of top results to return
        
    Returns:
        List of tuples: (text, similarity_score, original_index)
    """
    # Generate query embedding if not provided
    if query_embedding is None:
        print(f"Generating embedding for query: '{query_text}'...")
        generator = EmbeddingGenerator(provider=provider)
        query_embedding = generator.generate_embedding(query_text)
    
    # Find top-k similar
    print(f"Searching for top-{k} most similar sentences...")
    results = find_top_k_similar(query_embedding, embeddings, texts, k=k)
    
    return results


def display_results(query_text: str, results: List[Tuple[str, float, int]]):
    """
    Display search results in a formatted way.
    
    Args:
        query_text: Original query text
        results: List of (text, similarity_score, original_index) tuples
    """
    print("\n" + "="*80)
    print(f"Query: {query_text}")
    print("="*80)
    print(f"\nTop {len(results)} Most Similar Sentences:\n")
    
    for i, (text, score, idx) in enumerate(results, 1):
        print(f"{i}. [Similarity: {score:.4f}] (Index: {idx})")
        print(f"   {text}")
        print()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Find top-3 most similar sentences to a query using cosine similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python similarity_demo.py "machine learning algorithms"
  python similarity_demo.py "artificial intelligence" --provider gemini
  python similarity_demo.py "data science" --k 5
  python similarity_demo.py "python programming" --regenerate
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Query sentence to search for"
    )
    
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default="embeddings.json",
        help="Path to embeddings JSON file (default: embeddings.json)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        default=None,
        help="Embedding provider (default: from EMBEDDING_PROVIDER env var or 'openai')"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Number of top results to return (default: 3)"
    )
    
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of embeddings even if file exists"
    )
    
    args = parser.parse_args()
    
    # Determine provider
    provider = args.provider or os.getenv("EMBEDDING_PROVIDER", "openai")
    
    try:
        # Load or generate embeddings
        texts, embeddings = load_or_generate_embeddings(
            embeddings_file=args.embeddings_file,
            provider=provider,
            force_regenerate=args.regenerate
        )
        
        print(f"\nLoaded {len(texts)} sentences with embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}\n")
        
        # Search for similar sentences
        results = search_similar(
            query_text=args.query,
            texts=texts,
            embeddings=embeddings,
            provider=provider,
            k=args.top_k
        )
        
        # Display results
        display_results(args.query, results)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure you have set the API key in your .env file:")
        if provider == "openai":
            print("  OPENAI_API_KEY=your_key_here")
        else:
            print("  GEMINI_API_KEY=your_key_here")
        sys.exit(1)
    
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

