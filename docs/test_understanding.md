# Understanding Test - Embeddings & Vector Search

## Question 1: Embeddings
What is an embedding?
- A) A text file containing sentences
- B) A numerical vector representing text meaning
- C) A database table
- D) A Python function

**Answer:** B

---

## Question 2: Cosine Similarity
What does cosine similarity measure?
- A) The length of two vectors
- B) The angle between two vectors
- C) The sum of vector elements
- D) The difference between vectors

**Answer:** B

---

## Question 3: Embedding Generation
Which file is responsible for generating embeddings from text?
- A) similarity.py
- B) embeddings.py
- C) similarity_demo.py
- D) test_similarity.py

**Answer:** B

---

## Question 4: Cosine Similarity Formula
What is the formula for cosine similarity?
- A) A + B
- B) (A · B) / (||A|| × ||B||)
- C) A × B
- D) ||A|| + ||B||

**Answer:** B

---

## Question 5: Qdrant Purpose
What is Qdrant used for?
- A) Generating embeddings
- B) Storing and searching vectors efficiently
- C) Calculating cosine similarity
- D) Running unit tests

**Answer:** B

---

## Question 6: Mock Embeddings
Why are mock embeddings used in this project?
- A) They're faster than real embeddings
- B) They provide better semantic understanding
- C) They allow testing without API quota/costs
- D) They're required by Qdrant

**Answer:** C

---

## Question 7: Vector Storage
Where are embeddings stored in the in-memory approach?
- A) Qdrant database
- B) embeddings.json file
- C) Python variables only
- D) Docker container

**Answer:** B

---

## Question 8: Top-K Search
What does "top-k" mean in similarity search?
- A) The k-th most similar result
- B) The k most similar results
- C) Results with similarity > k
- D) Results sorted by index k

**Answer:** B

---

## Question 9: Qdrant Port
What port does Qdrant use for HTTP API?
- A) 6333
- B) 6334
- C) 8080
- D) 3000

**Answer:** A

---

## Question 10: Similarity Score Range
What is the typical range of cosine similarity for embeddings?
- A) -1 to 1
- B) 0 to 1
- C) 0 to 100
- D) -100 to 100

**Answer:** A (theoretical), but B (0 to 1) is typical for normalized embeddings

---

## Practical Questions

### Question 11: Command to Generate Embeddings
Write the command to generate 510 embeddings and save them to a JSON file.

**Answer:** `python embeddings.py`

---

### Question 12: Command to Search In-Memory
Write the command to find top-3 similar sentences to "data science" using in-memory search.

**Answer:** `python similarity_demo.py "data science"`

---

### Question 13: Command to Store in Qdrant
Write the command to store embeddings in Qdrant.

**Answer:** `python vector_db_demo.py store`

---

### Question 14: Command to Search in Qdrant
Write the command to search for "machine learning" in Qdrant with top-5 results.

**Answer:** `python vector_db_demo.py search "machine learning" -k 5`

---

### Question 15: Start Qdrant
Write the command to start Qdrant using Docker Compose.

**Answer:** `docker-compose up -d`

---

## Scenario Questions

### Question 16: Flow Understanding
Describe the complete flow when you run:
```bash
python similarity_demo.py "artificial intelligence"
```

**Expected Answer:**
1. Load embeddings from embeddings.json
2. Generate embedding for query "artificial intelligence"
3. Calculate cosine similarity with all 510 stored embeddings
4. Sort by similarity score
5. Return top-3 most similar sentences

---

### Question 17: Qdrant vs In-Memory
Explain when you would use Qdrant vs in-memory search.

**Expected Answer:**
- **In-Memory**: Small datasets, learning, demos, when you want simple file-based storage
- **Qdrant**: Large datasets (millions of vectors), production systems, when you need scalability and optimized search

---

### Question 18: Embedding Dimensions
What are the typical dimensions for:
- OpenAI embeddings: ?
- Gemini embeddings: ?
- Mock embeddings: ?

**Answer:**
- OpenAI: 1536 (text-embedding-3-small)
- Gemini: 768 (embedding-001)
- Mock: 384

---

## Bonus: Code Understanding

### Question 19: What does this code do?
```python
similarity = cosine_similarity(vec1, vec2)
```

**Answer:** Calculates the cosine similarity between two vectors, returning a value between -1 and 1 indicating how similar they are.

---

### Question 20: What is the purpose of this function?
```python
def find_top_k_similar(query_vector, candidate_vectors, candidate_texts, k=3):
```

**Answer:** Finds the top-k most similar texts to a query vector by calculating cosine similarity for all candidates, sorting by score, and returning the k best matches.

