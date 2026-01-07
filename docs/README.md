# Embeddings & Cosine Similarity Project

This project demonstrates embedding generation using OpenAI/Gemini APIs and cosine similarity calculations using NumPy.

## Features

- Generate embeddings from 510 sample sentences using OpenAI or Gemini API
- Manual cosine similarity implementation using NumPy
- CLI script to find top-3 most similar sentences to a query
- Comprehensive unit tests

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file from the example:
```bash
cp .env.example .env
```

3. Add your API key(s) to `.env`:
   - For OpenAI: Set `OPENAI_API_KEY=your_key_here`
   - For Gemini: Set `GEMINI_API_KEY=your_key_here`
   - Set `EMBEDDING_PROVIDER=openai` or `EMBEDDING_PROVIDER=gemini`

## Usage

### Generate Embeddings

Generate embeddings for 510 sample sentences:
```bash
python scripts/embeddings.py
```

This will create `embeddings.json` containing all sentences and their embeddings.

### Run Similarity Search (CLI)

Find top-3 most similar sentences to a query:
```bash
python scripts/similarity_demo.py "your query sentence here"
```

Options:
- `--provider openai|gemini`: Choose embedding provider
- `-k N`: Return top N results (default: 3)
- `--regenerate`: Force regeneration of embeddings
- `--embeddings-file PATH`: Specify embeddings file path

Examples:
```bash
python scripts/similarity_demo.py "machine learning algorithms"
python scripts/similarity_demo.py "artificial intelligence" --provider gemini
python scripts/similarity_demo.py "data science" -k 5
```

### Run Tests

Run unit tests for similarity calculations:
```bash
pytest tests/test_similarity.py -v
```

Or:
```bash
python -m pytest tests/test_similarity.py -v
```

## Project Structure

```
module2/
├── scripts/              # CLI tools
│   ├── embeddings.py     # Embedding generation script
│   ├── similarity_demo.py # In-memory similarity search
│   └── vector_db_demo.py # Qdrant vector database operations
├── tests/                # Unit tests
│   ├── test_similarity.py # Tests for similarity calculations
│   └── test_qdrant.py    # Tests for Qdrant operations
├── docs/                 # Documentation
│   ├── README.md         # Main documentation (this file)
│   ├── INTERVIEW_PREP_GUIDE.md
│   ├── QDRANT_SETUP.md
│   └── test_understanding.md
├── embeddings.py          # Core embedding generation module
├── similarity.py         # Cosine similarity implementation
├── qdrant_utils.py       # Qdrant vector database utilities
├── docker-compose.yml    # Docker setup for Qdrant
├── requirements.txt      # Python dependencies
└── .env.example         # Environment variables template
```

## How It Works

1. **Embedding Generation**: Uses OpenAI's `text-embedding-3-small` or Gemini's `embedding-001` model to convert text into high-dimensional vectors.

2. **Cosine Similarity**: Calculates the cosine of the angle between two vectors:
   ```
   similarity = (A · B) / (||A|| * ||B||)
   ```
   Values range from -1 (opposite) to 1 (identical), with 0 meaning orthogonal.

3. **Top-K Search**: Computes similarity between query embedding and all candidate embeddings, then returns the top-k most similar sentences.

## Qdrant Vector Database

### Setup Qdrant

1. Start Qdrant using Docker:
```bash
docker-compose up -d
```

Or using Docker directly:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Store embeddings in Qdrant:
```bash
python scripts/vector_db_demo.py store
```

3. Search in Qdrant:
```bash
python scripts/vector_db_demo.py search "machine learning algorithms"
```

See `docs/QDRANT_SETUP.md` for detailed instructions.

### Run Qdrant Tests

```bash
# Make sure Qdrant is running first
docker-compose up -d

# Run tests
pytest tests/test_qdrant.py -v
```

## Notes

- Embeddings are cached in `embeddings.json` to avoid regenerating them on each run
- The first run will take time to generate 510 embeddings via API calls
- Subsequent runs will load from the cached file for faster performance
- Qdrant provides fast similarity search for large-scale vector databases

