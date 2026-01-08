# Fix API Key Error

## Problem

You're seeing this error:
```
Incorrect API key provided: your_ope************here
```

This means your `.env` file has a placeholder API key instead of a real one.

## Solution Options

### Option 1: Add Your Real OpenAI API Key (Recommended)

1. **Get your API key:**
   - Go to: https://platform.openai.com/api-keys
   - Sign in or create an account
   - Click "Create new secret key"
   - Copy the key (you'll only see it once!)

2. **Update your .env file:**
   - Open `.env` in any text editor (Notepad, VS Code, etc.)
   - Find the line: `OPENAI_API_KEY=your_openai_api_key_here`
   - Replace `your_openai_api_key_here` with your actual key
   - Save the file

   Example:
   ```
   OPENAI_API_KEY=sk-proj-abc123xyz789...
   ```

3. **Try again:**
   ```bash
   python scripts/similarity_demo.py "machine learning algorithms"
   ```

### Option 2: Use Mock Provider (No API Key Needed)

If you don't have an API key or want to test without one:

1. **Update .env file:**
   ```
   EMBEDDING_PROVIDER=mock
   ```

2. **Or use the flag:**
   ```bash
   python scripts/similarity_demo.py "machine learning algorithms" --provider mock
   ```

   **Note:** Mock embeddings are deterministic and good for testing, but won't give real semantic similarity.

### Option 3: Use Gemini Instead

1. **Get Gemini API key:**
   - Go to: https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy it

2. **Update .env file:**
   ```
   EMBEDDING_PROVIDER=gemini
   GEMINI_API_KEY=your_actual_gemini_key_here
   ```

3. **Try again:**
   ```bash
   python scripts/similarity_demo.py "machine learning algorithms" --provider gemini
   ```

## Quick Check

To verify your .env file is set up correctly:

```bash
# Check if .env exists
Test-Path .env

# View .env (be careful - contains sensitive info!)
# Don't share this output publicly
Get-Content .env
```

## Important Notes

- ✅ `.env` is in `.gitignore` - it won't be committed to GitHub
- ✅ Never share your API keys publicly
- ✅ The `.env.example` file is just a template (safe to commit)
- ⚠️ If you see the placeholder text, you need to replace it with a real key

## Troubleshooting

### "File not found" error
- Make sure `.env` is in the root directory (same folder as `scripts/`)
- Copy from example: `cp .env.example .env`

### "Invalid API key" error
- Check that you copied the entire key (no spaces, no quotes)
- Make sure there are no extra characters
- Verify the key is active on OpenAI's website

### "Quota exceeded" error
- Your API key has hit its usage limit
- Wait for quota reset or upgrade your plan
- Or use `--provider mock` for testing


