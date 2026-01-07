# Example Outputs

This document shows example outputs from the CLI tools.

## Example 1: Generate Embeddings

**Command:**
```bash
python scripts/embeddings.py
```

**Output:**
```
Generating sample sentences...
Generated 510 sample sentences

Using openai for embeddings...
Generating embeddings (this may take a while)...
Processed 100/510 texts
Processed 200/510 texts
Processed 300/510 texts
Processed 400/510 texts
Processed 500/510 texts
Processed 510/510 texts

Generated 510 embeddings
Embedding dimension: 1536

Saved 510 embeddings to embeddings.json

Embeddings saved successfully!
```

---

## Example 2: In-Memory Similarity Search

**Command:**
```bash
python scripts/similarity_demo.py "machine learning algorithms"
```

**Output:**
```
Loading embeddings from embeddings.json...
Loaded 510 embeddings from embeddings.json

Loaded 510 sentences with embeddings
Embedding dimension: 1536

Generating embedding for query: 'machine learning algorithms'...
Searching for top-3 most similar sentences...

================================================================================
Query: machine learning algorithms
================================================================================

Top 3 Most Similar Sentences:

1. [Similarity: 0.9234] (Index: 42)
   Machine learning algorithms can identify patterns in large datasets.

2. [Similarity: 0.8567] (Index: 15)
   Artificial intelligence is transforming various industries.

3. [Similarity: 0.8123] (Index: 89)
   Data science combines statistics, programming, and domain expertise.
```

---

## Example 3: Qdrant Store Operation

**Command:**
```bash
python scripts/vector_db_demo.py store
```

**Output:**
```
Loaded 510 embeddings from embeddings.json
Connected to Qdrant at localhost:6333
Creating collection 'embeddings' with vector size 1536...
Collection 'embeddings' created successfully!

Inserting 510 vectors into Qdrant...
Inserted batch: 100/510 points
Inserted batch: 200/510 points
Inserted batch: 300/510 points
Inserted batch: 400/510 points
Inserted batch: 500/510 points
Inserted batch: 510/510 points
Successfully inserted 510 vectors into 'embeddings'

Collection Info:
  Name: embeddings
  Vectors: 510
  Vector Size: 1536
  Distance: COSINE
```

---

## Example 4: Qdrant Search Operation

**Command:**
```bash
python scripts/vector_db_demo.py search "artificial intelligence" -k 5
```

**Output:**
```
Generating embedding for query: 'artificial intelligence'...
Connected to Qdrant at localhost:6333
Searching for top-5 similar vectors in Qdrant...

================================================================================
Query: artificial intelligence
================================================================================

Top 5 Most Similar Results from Qdrant:

1. [Similarity: 0.9156] (ID: 15)
   Artificial intelligence is transforming various industries.

2. [Similarity: 0.8923] (ID: 42)
   Machine learning algorithms can identify patterns in large datasets.

3. [Similarity: 0.8745] (ID: 89)
   Data science combines statistics, programming, and domain expertise.

4. [Similarity: 0.8567] (ID: 123)
   Neural networks are inspired by the structure of the human brain.

5. [Similarity: 0.8345] (ID: 67)
   Cloud computing enables scalable and flexible infrastructure.
```

---

## Example 5: Test Output

**Command:**
```bash
pytest tests/test_similarity.py -v
```

**Output:**
```
======================== test session starts =========================
platform win32 -- Python 3.12.10, pytest-7.4.3
collected 12 items

tests/test_similarity.py::TestCosineSimilarity::test_identical_vectors PASSED
tests/test_similarity.py::TestCosineSimilarity::test_orthogonal_vectors PASSED
tests/test_similarity.py::TestCosineSimilarity::test_opposite_vectors PASSED
tests/test_similarity.py::TestCosineSimilarity::test_zero_vector PASSED
tests/test_similarity.py::TestCosineSimilarity::test_different_dimensions PASSED
tests/test_similarity.py::TestCosineSimilarityBatch::test_batch_similarity PASSED
tests/test_similarity.py::TestCosineSimilarityBatch::test_empty_candidates PASSED
tests/test_similarity.py::TestTopKSearch::test_find_top_k PASSED
tests/test_similarity.py::TestTopKSearch::test_find_top_k_more_than_available PASSED
tests/test_similarity.py::TestTopKSearch::test_find_top_k_zero PASSED
tests/test_similarity.py::TestNormalization::test_normalize_vector PASSED
tests/test_similarity.py::TestNormalization::test_normalize_vectors PASSED

======================== 12 passed in 0.15s =========================
```

---

## Example 6: Error Handling

**Command:**
```bash
python scripts/similarity_demo.py "query" --embeddings-file nonexistent.json
```

**Output:**
```
Error loading embeddings: [Errno 2] No such file or directory: 'nonexistent.json'
Generating new embeddings...
Generating 510 sample sentences...
Generating embeddings using openai...
Processed 100/510 texts
...
```

---

## Example 7: Using Different Providers

**Command:**
```bash
python scripts/similarity_demo.py "data science" --provider gemini
```

**Output:**
```
Loading embeddings from embeddings.json...
Loaded 510 embeddings from embeddings.json

Loaded 510 sentences with embeddings
Embedding dimension: 768

Generating embedding for query: 'data science'...
Searching for top-3 most similar sentences...

================================================================================
Query: data science
================================================================================

Top 3 Most Similar Sentences:

1. [Similarity: 0.9123] (Index: 89)
   Data science combines statistics, programming, and domain expertise.

2. [Similarity: 0.8845] (Index: 42)
   Machine learning algorithms can identify patterns in large datasets.

3. [Similarity: 0.8567] (Index: 156)
   Python is a versatile programming language used for web development.
```

**Note:** When using a different provider, the embedding dimensions change (Gemini: 768, OpenAI: 1536), so you may need to regenerate embeddings.

