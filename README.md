# Embeddings & Cosine Similarity Project

A semantic search system that finds similar text using embeddings and cosine similarity. Supports both in-memory search and Qdrant vector database for scalable similarity search.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

3. **Generate embeddings:**
   ```bash
   python scripts/embeddings.py
   ```

4. **Search for similar text:**
   ```bash
   python scripts/similarity_demo.py "machine learning algorithms"
   ```

## Project Structure

```
module2/
├── scripts/          # CLI tools and scripts
├── tests/            # Unit tests
├── docs/             # Documentation and guides
├── embeddings.py     # Core embedding generation module
├── similarity.py     # Cosine similarity implementation
└── qdrant_utils.py   # Qdrant vector database utilities
```

## Documentation

- **[Main Documentation](docs/README.md)** - Complete setup and usage guide
- **[Interview Prep Guide](docs/INTERVIEW_PREP_GUIDE.md)** - Technical deep dive and interview questions
- **[Qdrant Setup](docs/QDRANT_SETUP.md)** - Vector database setup instructions

## Features

- ✅ Multiple embedding providers (OpenAI, Gemini, Mock)
- ✅ In-memory similarity search using NumPy
- ✅ Qdrant vector database integration
- ✅ Comprehensive unit tests
- ✅ CLI tools for easy usage

## Quick Examples

### In-Memory Search
```bash
python scripts/similarity_demo.py "artificial intelligence" -k 5
```

### Qdrant Vector Database
```bash
# Store embeddings
python scripts/vector_db_demo.py store

# Search
python scripts/vector_db_demo.py search "data science"
```

### Run Tests
```bash
pytest tests/ -v
```

## License

MIT
