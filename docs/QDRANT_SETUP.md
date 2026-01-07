# Qdrant Vector Database Setup

This guide explains how to set up and use Qdrant vector database for storing and searching embeddings.

## Prerequisites

- Docker installed on your system
- Python dependencies installed: `pip install -r requirements.txt`

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. Start Qdrant:
```bash
docker-compose up -d
```

2. Verify Qdrant is running:
```bash
curl http://localhost:6333/
```

You should see a JSON response with Qdrant version information. The `/health` endpoint doesn't exist in Qdrant - use the root endpoint `/` instead.

3. Stop Qdrant:
```bash
docker-compose down
```

### Option 2: Using Docker Run

1. Start Qdrant container:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

2. Qdrant will be available at:
   - HTTP API: http://localhost:6333
   - gRPC API: localhost:6334

## Usage

### 1. Generate Embeddings (if not already done)

```bash
python embeddings.py
```

This creates `embeddings.json` with 510 sample sentences and their embeddings.

### 2. Store Embeddings in Qdrant

```bash
python vector_db_demo.py store
```

This will:
- Create a collection named "embeddings"
- Insert all 510 embeddings from `embeddings.json`
- Store vectors with their corresponding text

Options:
```bash
# Use custom collection name
python vector_db_demo.py store --collection my_collection

# Recreate collection if it exists
python vector_db_demo.py store --recreate

# Use custom embeddings file
python vector_db_demo.py store --embeddings-file my_embeddings.json
```

### 3. Search in Qdrant

```bash
python vector_db_demo.py search "machine learning algorithms"
```

This will:
- Generate embedding for your query
- Search Qdrant for top-3 most similar vectors
- Display results with similarity scores

Options:
```bash
# Get top-5 results
python vector_db_demo.py search "data science" -k 5

# Use different embedding provider
python vector_db_demo.py search "AI" --provider gemini
```

## Qdrant Web UI

Access the Qdrant dashboard at:
```
http://localhost:6333/dashboard
```

You can:
- View collections
- Inspect stored vectors
- Test queries
- Monitor performance

## API Endpoints

- Root/Version: `http://localhost:6333/` (returns Qdrant version info)
- Collections: `http://localhost:6333/collections`
- API docs: `http://localhost:6333/docs`
- Dashboard: `http://localhost:6333/dashboard`

## Configuration

### Environment Variables

Add to your `.env` file:

```env
# Qdrant Configuration
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # Only needed for Qdrant Cloud
```

### Custom Qdrant Server

If running Qdrant on a different host/port:

```bash
python vector_db_demo.py store --qdrant-url my-server.com --qdrant-port 6333
```

## Testing

Run unit tests for Qdrant operations:

```bash
pytest test_qdrant.py -v
```

**Note:** Tests require Qdrant to be running. Start it first:
```bash
docker-compose up -d
```

## Troubleshooting

### Connection Error

If you see "Error connecting to Qdrant":
1. Verify Qdrant is running: `curl http://localhost:6333/health`
2. Check Docker container: `docker ps`
3. Check logs: `docker logs qdrant`

### Port Already in Use

If port 6333 is already in use:
1. Change port in `docker-compose.yml`
2. Or use: `docker run -p 6335:6333 qdrant/qdrant`
3. Update `QDRANT_PORT` in `.env` or use `--qdrant-port 6335`

### Collection Already Exists

If collection already exists:
- Use `--recreate` flag to delete and recreate
- Or use a different collection name with `--collection`

## Data Persistence

Data is stored in `./qdrant_storage` directory (created by docker-compose).
This persists even when the container stops.

To start fresh:
```bash
docker-compose down
rm -rf qdrant_storage
docker-compose up -d
```

## Performance Tips

1. **Batch Size**: Default is 100 vectors per batch. Adjust in `qdrant_utils.py` if needed.
2. **Vector Size**: Smaller vectors (e.g., 384) are faster than larger ones (e.g., 1536).
3. **Distance Metric**: COSINE is typically best for embeddings, but EUCLID or DOT may work better for some use cases.

## Next Steps

- Explore Qdrant features: filtering, payload, hybrid search
- Scale to production with Qdrant Cloud
- Add metadata filtering to search queries
- Implement batch search for multiple queries

