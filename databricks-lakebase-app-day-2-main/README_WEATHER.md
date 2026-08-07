# Weather Data Harvester - Architecture & Pipeline Documentation

## Overview

This project implements an end-to-end weather data pipeline with semantic search capabilities:
1. **Data Ingestion**: Fetch weather alerts and forecasts from the National Weather Service API
2. **Storage**: Store documents in Lakebase (Databricks-managed Postgres)
3. **Embeddings**: Chunk narrative text and generate vector embeddings
4. **Search**: Semantic search over weather narratives using cosine similarity

---

## Data Source Selection

### National Weather Service (NWS) API

**Why NWS?**
- **Free & Public**: No authentication required, no rate limits for reasonable use
- **Authoritative**: Official weather data from NOAA
- **Rich Narrative Text**: Weather alerts and forecasts contain detailed natural language descriptions ideal for semantic search
- **Structured Data**: Clean JSON API with consistent schema
- **Real-time Updates**: Fresh data for active weather events

**Data Types Collected:**
1. **Active Alerts**: Severe weather warnings, watches, and advisories with detailed event descriptions
2. **Zone Forecasts**: Multi-day text forecasts for specific geographic zones
3. **Hourly Forecasts**: Granular hourly predictions (optional)

**API Endpoints Used:**
- `GET /alerts/active` - Active weather alerts by location
- `GET /points/{lat},{lon}` - Convert coordinates to grid points
- `GET /gridpoints/{office}/{gridX},{gridY}/forecast` - Detailed forecasts

---

## Schema Design

### 1. `weather_documents` Table (Main Data Store)

```sql
CREATE TABLE weather_documents (
    id TEXT PRIMARY KEY,              -- Unique ID from NWS API
    location TEXT NOT NULL,           -- Human-readable location (e.g., "Chicago, IL")
    source_type TEXT NOT NULL,        -- 'alerts', 'forecast', or 'hourly_forecast'
    headline TEXT,                    -- Short summary/title
    narrative_text TEXT,              -- Full narrative description (chunked for embeddings)
    issued_at TIMESTAMPTZ,            -- When the document was issued by NWS
    effective_at TIMESTAMPTZ,         -- When the event/forecast takes effect
    payload JSONB NOT NULL,           -- Full original API response for reference
    synced_at TIMESTAMPTZ DEFAULT now() -- When we fetched this data
);
```

**Design Decisions:**
- **`id` as API identifier**: Preserves NWS document IDs for deduplication and traceability
- **`location` as text**: Human-readable for display; indexed for filtering
- **`source_type` enum**: Enables filtering by document type (alerts vs. forecasts)
- **`narrative_text`**: Core field for chunking and embedding; contains the detailed weather description
- **`payload` as JSONB**: Stores full API response for audit trail and future schema evolution
- **Timestamps**: Track both NWS issue time and our sync time for freshness analysis

### 2. `weather_documents_embeddings` Table (Vector Store)

```sql
CREATE TABLE weather_documents_embeddings (
    id TEXT PRIMARY KEY,              -- Format: {weather_doc_id}_{chunk_index}
    weather_document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,         -- The actual text chunk
    embedding VECTOR(384) NOT NULL,   -- 384-dimensional vector
    model_name TEXT NOT NULL,         -- Embedding model used
    embedded_at TIMESTAMPTZ DEFAULT now(),
    FOREIGN KEY (weather_document_id) REFERENCES weather_documents(id),
    UNIQUE (weather_document_id, chunk_index)
);

CREATE INDEX idx_weather_embeddings_document_id 
    ON weather_documents_embeddings(weather_document_id);
    
CREATE INDEX idx_weather_embeddings_embedding 
    ON weather_documents_embeddings USING ivfflat (embedding vector_cosine_ops);
```

**Design Decisions:**
- **Chunk-level granularity**: Each chunk gets its own embedding for fine-grained retrieval
- **Composite ID**: `{doc_id}_{chunk_index}` ensures uniqueness and readability
- **Foreign key constraint**: Maintains referential integrity with parent documents
- **`chunk_text` stored**: Enables displaying relevant passages without joining back to full document
- **`model_name` tracking**: Supports model versioning and A/B testing
- **IVFFlat index**: Optimizes cosine similarity search for large vector datasets

---

## Chunking Strategy

### Parameters (Configurable via Notebook Widgets)

```python
CHUNK_SIZE = 800        # Characters per chunk
CHUNK_OVERLAP = 100     # Character overlap between consecutive chunks
```

**Rationale:**
- **800 characters**: Balances context (2-4 sentences) with embedding model capacity
- **100-character overlap**: Prevents semantic breaks at chunk boundaries
- **Character-based**: Simple, predictable, and fast (no NLP tokenization required)

**Trade-offs:**
- ✅ Fast processing
- ✅ Predictable chunk sizes
- ✅ Works well for weather narratives (already well-structured)
- ❌ May split mid-sentence
- ❌ Not semantically aware (doesn't preserve paragraph boundaries)

**Alternative Approaches Considered:**
- **Sentence-based chunking**: More expensive, minimal benefit for weather text
- **Paragraph-based**: Weather narratives often have inconsistent paragraph structure
- **Fixed token count**: Requires tokenization overhead; character-based is sufficient

---

## Embedding Model

### Model: `all-MiniLM-L6-v2`

**Specifications:**
- **Dimensions**: 384
- **Model Size**: 80 MB
- **Speed**: ~14,000 sentences/second on CPU
- **Quality**: 68.7 on STS benchmark

**Why This Model?**
1. **Speed**: Fast enough for real-time search queries (<100ms per query)
2. **Efficiency**: Small model size fits in app memory without GPU
3. **Quality**: Sufficient semantic understanding for weather terminology
4. **Compatibility**: Works with pgvector (supports up to 16,000 dimensions)
5. **Open Source**: MIT license, self-hostable

**Alternatives Considered:**
- **all-mpnet-base-v2** (768D): Higher quality but 2x slower and larger
- **OpenAI text-embedding-ada-002** (1536D): Better quality but requires API costs and external dependency
- **BGE-small-en-v1.5** (384D): Comparable quality but less documented

---

## End-to-End Pipeline

### 1. **Data Sync** (Flask App → Lakebase)

**Via Web UI:**
1. Navigate to your deployed app URL
2. Enter locations: `Chicago, IL, Austin, TX, Seattle, WA`
3. Optionally select a source type filter (All / Alerts / Forecasts)
4. Click **Fetch Weather Data**
5. Data is synced to `weather_documents` table

**Programmatic:**
```bash
curl -X POST http://localhost:8000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
```

**What Happens:**
- App calls `WeatherClient.resolve_location()` to geocode locations
- Fetches active alerts via NWS API `/alerts/active?point={lat},{lon}`
- Fetches forecasts via `/gridpoints/{office}/{gridX},{gridY}/forecast`
- Upserts documents to `weather_documents` (deduplicates by `id`)

### 2. **Generate Embeddings** (Databricks Notebook)

**Notebook:** `notebooks/ingest_ticker_news_embeddings_zach.ipynb`

**Steps:**
1. Open the notebook in Databricks
2. **Configure widgets** (or use defaults):
   - `LAKEBASE_PROJECT_NAME`: Your Lakebase project
   - `LAKEBASE_BRANCH_NAME`: Your branch (default: `main`)
   - `EMBEDDING_MODEL_NAME`: `all-MiniLM-L6-v2`
   - `CHUNK_SIZE`: 800
   - `CHUNK_OVERLAP`: 100

3. **Run all cells** in order:
   - **Cell 1-10**: Setup and configuration
   - **Cell 11**: Load weather documents from `weather_documents` table
   - **Cell 14**: Load embedding model (sentence-transformers)
   - **Cell 19**: Chunk `narrative_text` into overlapping segments
   - **Cell 21**: Compute embeddings for each chunk (batched)
   - **Cell 25**: Insert chunk embeddings into `weather_documents_embeddings`

4. **Post-processing** (run in Lakebase Postgres if needed):
   ```sql
   UPDATE weather_documents_embeddings 
   SET embedding = embedding::vector 
   WHERE embedding IS NOT NULL;
   ```
   This casts the PostgreSQL array to pgvector's VECTOR type.

**Expected Output:**
```
✅ Extracted 42 text chunks from 15 weather documents
✅ Computed 42 chunk embeddings using all-MiniLM-L6-v2
✅ Successfully inserted 42 new chunk embeddings
```

### 3. **Semantic Search** (Flask App)

**Via Web UI:**
1. Navigate to the **Semantic Search** section (top of page)
2. Enter a natural language query:
   - `"severe thunderstorm warnings"`
   - `"extreme heat advisories"`
   - `"flooding risks for tomorrow"`
3. Click **Search**
4. Results ranked by semantic similarity (cosine distance)

**Programmatic:**
```bash
curl -X POST http://localhost:8000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "severe thunderstorm warnings", "limit": 10}'
```

**What Happens:**
1. Query text is embedded using the same model (`all-MiniLM-L6-v2`)
2. Cosine similarity computed against all stored embeddings:
   ```sql
   SELECT chunk_text, embedding <=> query_vector AS distance
   FROM weather_documents_embeddings
   ORDER BY embedding <=> query_vector
   LIMIT 10;
   ```
3. Results joined with `weather_documents` to include location, headline, etc.
4. Returned with similarity scores (lower distance = more relevant)

---

## Known Limitations & Future Improvements

### Current Limitations

1. **Manual Embedding Refresh**
   - **Issue**: Embeddings are not automatically generated when new weather data is synced
   - **Workaround**: Must manually run the notebook after each sync
   - **Impact**: New weather data is not searchable until embeddings are generated

2. **US-Only Coverage**
   - **Issue**: NWS API only covers United States locations
   - **Workaround**: None (API limitation)
   - **Impact**: Cannot fetch weather for international locations

3. **Character-Based Chunking**
   - **Issue**: Chunks may split mid-sentence or mid-paragraph
   - **Workaround**: Use 100-character overlap to minimize semantic breaks
   - **Impact**: Occasional awkward chunk boundaries, but minimal quality loss

4. **Cold Start Latency**
   - **Issue**: First search query is slow (~2-3 seconds) due to model loading
   - **Workaround**: Model is lazy-loaded and cached in memory
   - **Impact**: Only affects first search after app restart

5. **No Real-Time Updates**
   - **Issue**: Weather data is not automatically refreshed
   - **Workaround**: User must manually trigger sync via UI or scheduled job
   - **Impact**: Data may become stale during active weather events

6. **Single Embedding Model**
   - **Issue**: Cannot compare different models or use multiple models simultaneously
   - **Workaround**: `model_name` column supports versioning, but requires re-embedding
   - **Impact**: Difficult to A/B test embedding quality

### Potential Improvements

#### 1. **Automated Embedding Pipeline**
Trigger embeddings automatically after sync by invoking a Databricks Job via API

#### 2. **Scheduled Data Refresh**
Add a Databricks Job to sync weather data hourly during active weather seasons

#### 3. **Semantic Chunking**
Replace character-based chunking with sentence-aware chunking using LangChain or spaCy

#### 4. **Hybrid Search (Keyword + Semantic)**
Combine full-text search with vector search for better recall and precision

#### 5. **Multi-Model Support**
Store embeddings from multiple models for comparison and A/B testing

#### 6. **International Coverage**
Integrate OpenWeatherMap, Weather.gov International, or Met Office UK APIs

#### 7. **Reranking**
Add a cross-encoder reranking step for top results to improve relevance

#### 8. **Caching**
Cache frequent queries with LRU cache to reduce embedding computation overhead

---

## Troubleshooting

### Common Issues

#### 1. "No results found" in semantic search
**Cause**: Embeddings table is empty or not properly indexed

**Solution:**
```sql
-- Check if embeddings exist
SELECT COUNT(*) FROM weather_documents_embeddings;

-- Verify vector type
SELECT data_type FROM information_schema.columns 
WHERE table_name = 'weather_documents_embeddings' AND column_name = 'embedding';
-- Should return: USER-DEFINED (udt_name: vector)

-- If type is wrong, cast arrays to vectors
UPDATE weather_documents_embeddings 
SET embedding = embedding::vector;
```

#### 2. "Search returns identical results for different queries"
**Cause**: Embedding model not loaded or query not being embedded

**Solution:** Check app logs for "Loading embedding model: all-MiniLM-L6-v2" and restart the app if needed

#### 3. "ModuleNotFoundError: No module named 'sentence_transformers'"
**Cause**: Missing dependencies after `restartPython()`

**Solution:** Add `%pip install sentence-transformers` at the top of the cell that imports it (Cell 14)

#### 4. Source type filter not working
**Cause**: Frontend dropdown value not being passed to backend

**Solution:** Verify the dropdown is visible and selected value is being sent in the API request

---

## References

* [National Weather Service API](https://www.weather.gov/documentation/services-web-api)
* [pgvector Documentation](https://github.com/pgvector/pgvector)
* [Sentence Transformers](https://www.sbert.net/)
* [Databricks Lakebase](https://docs.databricks.com/lakebase/)
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding model benchmarks
