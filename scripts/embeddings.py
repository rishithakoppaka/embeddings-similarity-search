"""
Embedding generation module using OpenAI or Gemini API.
Supports generating embeddings from text sentences.
"""

import os
import json
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Try importing OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try importing Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or Gemini API."""
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize the embedding generator.
        
        Args:
            provider: Either "openai", "gemini", or "mock" (for testing without API)
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=api_key)
            self.model = "text-embedding-3-small"  # or "text-embedding-ada-002"
            
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.model = "models/embedding-001"
            
        elif self.provider == "mock":
            # Mock provider for testing without API access
            # Uses deterministic random embeddings based on text hash
            import hashlib
            self.hashlib = hashlib
            self.embedding_dim = 384  # Common embedding dimension
            print("Using MOCK embeddings (for testing without API access)")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'gemini', or 'mock'")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embedding vector
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding)
        
        elif self.provider == "gemini":
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
        
        elif self.provider == "mock":
            # Generate deterministic mock embedding based on text hash
            # This ensures similar texts get similar embeddings
            hash_obj = self.hashlib.md5(text.encode('utf-8'))
            hash_int = int(hash_obj.hexdigest(), 16) % (2**32 - 1)  # Limit to valid seed range
            rng = np.random.RandomState(hash_int)
            embedding = rng.normal(0, 1, self.embedding_dim).astype(np.float32)
            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process per batch
            
        Returns:
            List of numpy arrays (embeddings)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                if self.provider == "openai":
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [np.array(item.embedding) for item in response.data]
                    embeddings.extend(batch_embeddings)
                
                elif self.provider == "gemini":
                    # Gemini processes one at a time in batch
                    for text in batch:
                        result = genai.embed_content(
                            model=self.model,
                            content=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(np.array(result['embedding']))
                
                elif self.provider == "mock":
                    # Mock embeddings - fast generation
                    for text in batch:
                        embeddings.append(self.generate_embedding(text))
                
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            except Exception as e:
                # Re-raise if not a quota error, or if we're already using mock
                if self.provider == "mock":
                    raise
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg or "resourceexhausted" in error_msg:
                    raise  # Let the caller handle quota errors
                else:
                    raise
        
        return embeddings
    
    def save_embeddings(self, texts: List[str], embeddings: List[np.ndarray], 
                       filepath: str = "embeddings.json"):
        """
        Save texts and their embeddings to a JSON file.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding arrays
            filepath: Path to save the JSON file
        """
        data = {
            "texts": texts,
            "embeddings": [emb.tolist() for emb in embeddings]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(embeddings)} embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str = "embeddings.json") -> Tuple[List[str], List[np.ndarray]]:
        """
        Load texts and embeddings from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Tuple of (texts, embeddings)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        texts = data["texts"]
        embeddings = [np.array(emb) for emb in data["embeddings"]]
        
        print(f"Loaded {len(embeddings)} embeddings from {filepath}")
        return texts, embeddings


def generate_sample_sentences(count: int = 510) -> List[str]:
    """
    Generate sample sentences for testing.
    Creates diverse sentences covering various topics.
    
    Args:
        count: Number of sentences to generate
        
    Returns:
        List of sample sentences
    """
    sentences = []
    
    # Technology topics
    tech_sentences = [
        "Python is a versatile programming language used for web development.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Artificial intelligence is transforming various industries.",
        "Cloud computing enables scalable and flexible infrastructure.",
        "Data science combines statistics, programming, and domain expertise.",
        "Neural networks are inspired by the structure of the human brain.",
        "API integration allows different software systems to communicate.",
        "Database optimization improves query performance significantly.",
        "Version control systems help track changes in code.",
        "Cybersecurity measures protect against unauthorized access.",
    ]
    
    # Science topics
    science_sentences = [
        "The theory of relativity revolutionized our understanding of space and time.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "DNA contains the genetic instructions for all living organisms.",
        "The periodic table organizes chemical elements by their properties.",
        "Evolution explains the diversity of life on Earth.",
        "Quantum mechanics describes behavior at atomic and subatomic scales.",
        "Climate change affects weather patterns globally.",
        "The human brain contains billions of interconnected neurons.",
        "Gravity is the force that attracts objects toward each other.",
        "Cells are the basic structural units of all living things.",
    ]
    
    # Business topics
    business_sentences = [
        "Effective marketing strategies increase brand awareness and sales.",
        "Customer satisfaction is crucial for business success.",
        "Financial planning helps manage resources efficiently.",
        "Leadership skills are essential for managing teams.",
        "Innovation drives competitive advantage in markets.",
        "Supply chain management optimizes product delivery.",
        "Market research provides insights into consumer behavior.",
        "Strategic planning aligns business goals with actions.",
        "Employee engagement improves productivity and retention.",
        "Digital transformation modernizes business operations.",
    ]
    
    # General knowledge
    general_sentences = [
        "Reading books expands knowledge and improves vocabulary.",
        "Exercise promotes physical and mental well-being.",
        "Traveling exposes people to different cultures and perspectives.",
        "Cooking is both an art and a science.",
        "Music has the power to evoke emotions and memories.",
        "Education is the foundation of personal and professional growth.",
        "Communication skills are vital in all aspects of life.",
        "Time management helps achieve goals more efficiently.",
        "Creativity involves thinking outside conventional boundaries.",
        "Friendship provides support and companionship throughout life.",
    ]
    
    # Combine all categories
    base_sentences = tech_sentences + science_sentences + business_sentences + general_sentences
    
    # Generate variations to reach the desired count
    variations = [
        "This concept is fundamental to understanding the topic.",
        "Many experts have studied this phenomenon extensively.",
        "Research shows that this approach yields positive results.",
        "The implementation requires careful planning and execution.",
        "Understanding the basics is essential before advanced topics.",
        "Practice and repetition improve skill development.",
        "Collaboration enhances problem-solving capabilities.",
        "Feedback helps identify areas for improvement.",
        "Documentation ensures knowledge transfer and continuity.",
        "Testing validates the correctness of implementations.",
    ]
    
    # Generate sentences by combining and varying
    all_sentences = base_sentences + variations
    
    # Create variations with different structures
    while len(sentences) < count:
        for base in all_sentences:
            if len(sentences) >= count:
                break
            
            # Add original
            if base not in sentences:
                sentences.append(base)
            
            # Add variations
            if len(sentences) < count:
                sentences.append(f"Note that {base.lower()}")
            if len(sentences) < count:
                sentences.append(f"It is important to understand that {base.lower()}")
            if len(sentences) < count:
                sentences.append(f"One should consider that {base.lower()}")
    
    return sentences[:count]


if __name__ == "__main__":
    # Example usage
    print("Generating sample sentences...")
    sentences = generate_sample_sentences(510)
    print(f"Generated {len(sentences)} sample sentences")
    
    # Initialize generator (defaults to OpenAI, can be changed to "gemini" or "mock")
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    print(f"\nUsing {provider} for embeddings...")
    
    try:
        generator = EmbeddingGenerator(provider=provider)
        
        print("Generating embeddings (this may take a while)...")
        embeddings = generator.generate_embeddings_batch(sentences)
        
        print(f"\nGenerated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Save embeddings
        generator.save_embeddings(sentences, embeddings)
        print("\nEmbeddings saved successfully!")
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            print(f"\nAPI quota exceeded. Switching to MOCK embeddings for testing...")
            print("(Mock embeddings are deterministic and suitable for testing similarity)")
            provider = "mock"
            generator = EmbeddingGenerator(provider=provider)
            print("Generating mock embeddings...")
            embeddings = generator.generate_embeddings_batch(sentences)
            print(f"\nGenerated {len(embeddings)} mock embeddings")
            print(f"Embedding dimension: {len(embeddings[0])}")
            generator.save_embeddings(sentences, embeddings)
            print("\nMock embeddings saved successfully!")
            print("\nNote: To use real API embeddings, wait for quota reset or upgrade your plan.")
        else:
            print(f"Error: {e}")
            print("\nMake sure you have:")
            print("1. Set OPENAI_API_KEY or GEMINI_API_KEY in .env file")
            print("2. Installed required packages: pip install -r requirements.txt")
            print("\nAlternatively, set EMBEDDING_PROVIDER=mock in .env to use mock embeddings")

