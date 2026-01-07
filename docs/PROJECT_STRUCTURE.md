# Project Structure

This document describes the reorganized project structure for GitHub.

## Directory Layout

```
module2/
├── scripts/                    # CLI tools and executable scripts
│   ├── __init__.py
│   ├── embeddings.py           # Generate embeddings from text
│   ├── similarity_demo.py      # In-memory similarity search CLI
│   └── vector_db_demo.py        # Qdrant vector database CLI
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_similarity.py      # Tests for similarity calculations
│   └── test_qdrant.py          # Tests for Qdrant operations
│
├── docs/                        # Documentation
│   ├── README.md                # Main documentation
│   ├── INTERVIEW_PREP_GUIDE.md # Technical deep dive
│   ├── QDRANT_SETUP.md         # Qdrant setup instructions
│   ├── example_outputs.md       # Example CLI outputs
│   ├── PROJECT_STRUCTURE.md    # This file
│   └── test_understanding.md    # Test documentation
│
├── embeddings.py                # Core embedding generation module
├── similarity.py                # Cosine similarity implementation
├── qdrant_utils.py              # Qdrant vector database utilities
├── README.md                    # Root README (points to docs/)
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker setup for Qdrant
└── .env.example                 # Environment variables template
```

## File Organization

### `/scripts` - CLI Tools
All command-line interface scripts that users can run directly:
- **embeddings.py**: Generates embeddings for sample sentences
- **similarity_demo.py**: Performs in-memory similarity search
- **vector_db_demo.py**: Manages Qdrant vector database operations

**Usage:**
```bash
python scripts/embeddings.py
python scripts/similarity_demo.py "query text"
python scripts/vector_db_demo.py store
```

### `/tests` - Unit Tests
All test files for the project:
- **test_similarity.py**: Tests for cosine similarity functions
- **test_qdrant.py**: Tests for Qdrant operations

**Usage:**
```bash
pytest tests/ -v
pytest tests/test_similarity.py -v
pytest tests/test_qdrant.py -v
```

### `/docs` - Documentation
All documentation files:
- **README.md**: Complete setup and usage guide
- **INTERVIEW_PREP_GUIDE.md**: Technical explanations and interview prep
- **QDRANT_SETUP.md**: Detailed Qdrant setup instructions
- **example_outputs.md**: Example outputs from CLI tools
- **PROJECT_STRUCTURE.md**: This file

### Root Directory - Core Modules
Core Python modules that are imported by scripts and tests:
- **embeddings.py**: Embedding generation logic
- **similarity.py**: Cosine similarity calculations
- **qdrant_utils.py**: Qdrant database utilities

## Import Paths

Scripts and tests automatically add the root directory to `sys.path` to import core modules:

```python
# In scripts/similarity_demo.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import EmbeddingGenerator
from similarity import find_top_k_similar
```

## Benefits of This Structure

1. **Clear Separation**: Scripts, tests, and docs are clearly separated
2. **GitHub Ready**: Follows common GitHub project conventions
3. **Easy Navigation**: Users can quickly find what they need
4. **Scalable**: Easy to add new scripts, tests, or docs
5. **Professional**: Clean structure suitable for portfolio/interview

## Migration Notes

If you have existing code that references the old paths:

**Old:**
```bash
python similarity_demo.py "query"
pytest test_similarity.py
```

**New:**
```bash
python scripts/similarity_demo.py "query"
pytest tests/test_similarity.py
```

All imports are automatically handled - no code changes needed in your own scripts that import the core modules.

