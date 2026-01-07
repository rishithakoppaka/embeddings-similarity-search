# Quick Setup Steps

Follow these steps to set up your repository for GitHub:

## Step 1: Configure Git (if needed)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 2: Create .env file

```bash
# Copy the example
cp .env.example .env

# Then edit .env and add your API keys
# Use any text editor to add your OPENAI_API_KEY or GEMINI_API_KEY
```

## Step 3: Stage and commit files

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Embeddings & Cosine Similarity Project"
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `embeddings-similarity-search` (or your choice)
3. Description: "Semantic search using embeddings and cosine similarity"
4. Choose Public or Private
5. **DO NOT** check "Initialize with README" (we already have one)
6. Click "Create repository"

## Step 5: Connect and Push

```bash
# Add remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 6: Verify

1. Go to your GitHub repository
2. Check that all files are present
3. Verify `.env` is NOT visible (it's in .gitignore)

## Done! 🎉

For detailed instructions, see `docs/SETUP_GUIDE.md`

