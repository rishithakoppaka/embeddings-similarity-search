# Repository Setup Guide

This guide will walk you through setting up this repository for GitHub.

## Prerequisites

- Git installed on your system
- GitHub account
- Python 3.8+ installed

## Step 1: Initial Git Setup

### Configure Git (if not already done)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 2: Create Environment File

Create a `.env` file from the example:

```bash
# Copy the example (you'll need to create .env.example first)
cp .env.example .env
```

Or manually create `.env` with:

```env
# Embedding Provider Configuration
EMBEDDING_PROVIDER=openai

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Gemini API Key (optional)
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important:** The `.env` file is already in `.gitignore` and will NOT be committed to GitHub.

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Initialize Git Repository

If you haven't already:

```bash
git init
```

## Step 5: Add Files to Git

```bash
# Check what will be added
git status

# Add all files (respects .gitignore)
git add .

# Verify what's staged
git status
```

## Step 6: Create Initial Commit

```bash
git commit -m "Initial commit: Embeddings & Cosine Similarity Project

- Add core modules (embeddings.py, similarity.py, qdrant_utils.py)
- Add CLI scripts in scripts/ directory
- Add unit tests in tests/ directory
- Add comprehensive documentation in docs/
- Add Docker setup for Qdrant
- Add project structure and examples"
```

## Step 7: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right
3. Select "New repository"
4. Repository name: `embeddings-similarity-search` (or your preferred name)
5. Description: "Semantic search system using embeddings and cosine similarity"
6. Choose Public or Private
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 8: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add remote repository (replace with your username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/embeddings-similarity-search.git

# Verify remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 9: Verify Setup

1. Go to your GitHub repository page
2. Verify all files are present
3. Check that `.env` is NOT visible (it's in .gitignore)
4. Verify `embeddings.json` is present (if you want to include it) or excluded

## Step 10: Test the Setup

Clone the repository in a new location to test:

```bash
cd /tmp  # or another directory
git clone https://github.com/YOUR_USERNAME/embeddings-similarity-search.git
cd embeddings-similarity-search

# Create .env file
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Repository Structure on GitHub

Your repository should have this structure:

```
embeddings-similarity-search/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── docker-compose.yml
├── scripts/
│   ├── embeddings.py
│   ├── similarity_demo.py
│   └── vector_db_demo.py
├── tests/
│   ├── test_similarity.py
│   └── test_qdrant.py
├── docs/
│   ├── README.md
│   ├── INTERVIEW_PREP_GUIDE.md
│   ├── QDRANT_SETUP.md
│   ├── example_outputs.md
│   └── PROJECT_STRUCTURE.md
├── embeddings.py
├── similarity.py
└── qdrant_utils.py
```

## Files NOT Included (in .gitignore)

- `.env` - Contains API keys (sensitive)
- `__pycache__/` - Python cache files
- `qdrant_storage/` - Local Qdrant database files
- `*.pyc` - Compiled Python files
- Virtual environment directories

## Optional: Add Repository Topics

On GitHub, add topics to help others find your repository:
- `embeddings`
- `vector-search`
- `cosine-similarity`
- `qdrant`
- `semantic-search`
- `python`
- `machine-learning`

## Optional: Add Badges to README

You can add badges to your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
```

## Troubleshooting

### Issue: "Permission denied" when pushing

**Solution:** Use SSH instead of HTTPS, or configure GitHub credentials:
```bash
# Use SSH
git remote set-url origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# Or configure credential helper
git config --global credential.helper store
```

### Issue: Large file warnings

**Solution:** If `embeddings.json` is too large:
1. Add it to `.gitignore`
2. Or use Git LFS: `git lfs track "*.json"`

### Issue: Tests fail after cloning

**Solution:** Make sure you:
1. Created `.env` file with API keys
2. Installed dependencies: `pip install -r requirements.txt`
3. Started Qdrant (for Qdrant tests): `docker-compose up -d`

## Next Steps

1. ✅ Repository is set up on GitHub
2. ✅ Code is pushed
3. 📝 Add a detailed README (already done!)
4. 🧪 Set up GitHub Actions for CI/CD (optional)
5. 📊 Add contribution guidelines (optional)
6. 🐛 Set up issue templates (optional)

## GitHub Actions (Optional)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/test_similarity.py -v
```

## Success Checklist

- [ ] Git repository initialized
- [ ] `.env` file created (not committed)
- [ ] Dependencies installed
- [ ] GitHub repository created
- [ ] Local repo connected to GitHub
- [ ] Code pushed to GitHub
- [ ] Repository structure verified
- [ ] Tests pass locally
- [ ] README displays correctly on GitHub

Congratulations! Your repository is now set up on GitHub! 🎉

