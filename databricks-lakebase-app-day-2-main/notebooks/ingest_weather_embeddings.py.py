# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Ticker News -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Context Engineering on Databricks** course.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `watchlist` table in Lakebase to find out which ticker
# MAGIC    symbols are currently being tracked.
# MAGIC 2. Fetches recent news for those tickers directly from the Massive
# MAGIC    `/v2/reference/news` endpoint (see `massive_client.py` for the same
# MAGIC    call shape used by the Flask app's `POST /news/sync` route), rate
# MAGIC    limited to stay within the free Massive API tier's strict quota, and
# MAGIC    upserts the results into the `ticker_news_documents` table.
# MAGIC 3. Computes a sentence embedding for each article (title + description)
# MAGIC    using Spark, distributed across the cluster via a pandas UDF, and
# MAGIC    writes them into a `ticker_news_embeddings` table using the
# MAGIC    `pgvector` Postgres extension so downstream RAG / context-engineering
# MAGIC    exercises can run similarity search directly in Postgres.
# MAGIC 4. Fetches the full article body for each `article_url` (via
# MAGIC    `trafilatura`, which strips nav/ads/boilerplate from the raw HTML),
# MAGIC    splits it into overlapping text chunks, embeds each chunk, and writes
# MAGIC    them into a `ticker_news_chunk_embeddings` table - so RAG exercises can
# MAGIC    retrieve fine-grained passages from article bodies, not just
# MAGIC    title/description.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

dbutils.widgets.text("weather_documents_table_name", "weather_documents", "Source table (weather documents)")
dbutils.widgets.text("embeddings_table_name", "weather_documents_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("news_fetch_limit", "50", "Max articles to fetch per ticker")
dbutils.widgets.text("max_requests_per_minute", "5", "Massive API rate limit (free tier is strict)")
dbutils.widgets.text("chunk_size", "800", "Article content chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Article content chunk overlap (chars)")

WEATHER_DOCUMENTS_TABLE_NAME = dbutils.widgets.get("weather_documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
NEWS_FETCH_LIMIT = int(dbutils.widgets.get("news_fetch_limit"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces psycopg3
# MAGIC needs for connection (host/port/dbname/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# DBTITLE 1,Test Psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using OAuth token authentication as user: {db_user}\n")

# Test psycopg3 connection with OAuth token
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=10
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {WEATHER_DOCUMENTS_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {WEATHER_DOCUMENTS_TABLE_NAME}")
    
    cursor.execute(f"SELECT * FROM {WEATHER_DOCUMENTS_TABLE_NAME} LIMIT 5")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    print(f"\nColumns: {colnames}")
    for row in rows:
        print(row)
    
    cursor.close()
    conn.close()
    print("\n✅ psycopg3 with OAuth authentication working correctly!")
except Exception as e:
    import traceback
    print(f"❌ Connection failed: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, you must manually create the required tables
# MAGIC in your Lakebase Postgres database:
# MAGIC
# MAGIC 1. Run `sql/01_setup_news_table.sql` to create `ticker_news_documents`
# MAGIC 2. Run `sql/02_setup_embeddings_table.sql` to create `ticker_news_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC 3. Run `sql/03_setup_chunk_embeddings_table.sql` to create `ticker_news_chunk_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC
# MAGIC This notebook uses psycopg2 with OAuth token authentication for all database operations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch news from Massive for watchlisted tickers
# MAGIC
# MAGIC This ETL is now self-contained: instead of relying on the Flask app's
# MAGIC `POST /news/sync` route to have populated `ticker_news_documents` ahead of
# MAGIC time, the notebook queries the `watchlist` table in Lakebase directly to
# MAGIC find out which tickers are being tracked, then pulls news for exactly
# MAGIC those tickers from Massive itself.
# MAGIC
# MAGIC The free Massive API tier is rate-limited very aggressively, so requests
# MAGIC are made **serially** (not distributed across Spark workers) with a sleep
# MAGIC between calls that enforces `MAX_REQUESTS_PER_MINUTE` (default 5/min).

# COMMAND ----------

# DBTITLE 1,Fetch news and sync using Lakebase SDK
import pandas as pd
import psycopg2

# Load news documents using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    # Query with embedding_text computed
    query = f"""
        SELECT 
            id,
            narrative_text
        FROM {WEATHER_DOCUMENTS_TABLE_NAME}
    """
    
    weather_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(weather_df)} rows from {WEATHER_DOCUMENTS_TABLE_NAME}")
    display(weather_df.head(5))
finally:
    conn.close()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC The `pgvector` extension must be enabled and the destination table
# MAGIC created with the correct vector dimension before inserting embeddings.

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/02_setup_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/02_setup_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk narrative text from weather documents
# MAGIC
# MAGIC To enable fine-grained semantic search, we split each weather document's
# MAGIC `narrative_text` into overlapping chunks. Each chunk is embedded separately,
# MAGIC allowing RAG systems to retrieve specific passages rather than entire documents.
# MAGIC Chunk size and overlap are configurable via widgets (default: 800 chars with
# MAGIC 100 char overlap).

# COMMAND ----------

# DBTITLE 1,Chunk narrative text from weather documents
import pandas as pd

print(f"Chunking narrative text from {len(weather_df)} weather documents...")

# Chunk narrative text from weather documents
out_doc_ids, out_chunk_indexes, out_chunk_texts = [], [], []

for idx, row in weather_df.iterrows():
    doc_id = row['id']
    narrative_text = row['narrative_text']
    
    if not narrative_text or not narrative_text.strip():
        continue

    # Split into overlapping chunks
    for chunk_index, start in enumerate(range(0, len(narrative_text), CHUNK_SIZE - CHUNK_OVERLAP)):
        chunk_text = narrative_text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        out_doc_ids.append(doc_id)
        out_chunk_indexes.append(chunk_index)
        out_chunk_texts.append(chunk_text)
        if start + CHUNK_SIZE >= len(narrative_text):
            break
    
    # Progress update every 10 rows
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(weather_df)} rows")

chunks_df = pd.DataFrame({
    "weather_document_id": out_doc_ids,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Extracted {len(chunks_df)} text chunks from {len(weather_df)} weather documents")
print(f"Average {len(chunks_df) / len(weather_df):.1f} chunks per document")
display(chunks_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings for weather document chunks
# MAGIC
# MAGIC Load the sentence-transformers model (if not already loaded) and compute
# MAGIC embeddings for each text chunk. Processing happens in batches for memory
# MAGIC efficiency, generating one vector per chunk instead of per document.

# COMMAND ----------

# DBTITLE 1,Compute embeddings for weather document chunks
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Model should already be loaded from earlier, but ensure cache is set
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Computing chunk embeddings using {EMBEDDING_MODEL_NAME}...")
# Reuse the model if already loaded, otherwise load it
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute chunk embeddings in batches
batch_size = 32
all_chunk_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_chunk_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

# Create chunk embeddings DataFrame
chunk_embeddings_df = pd.DataFrame({
    "weather_document_id": chunks_df["weather_document_id"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} chunk embeddings using {EMBEDDING_MODEL_NAME}")
display(chunk_embeddings_df.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify weather_documents_embeddings table schema
# MAGIC
# MAGIC The embeddings table must support chunk storage with weather_document_id,
# MAGIC chunk_index, chunk_text, and a foreign key constraint.

# COMMAND ----------

# Before running the cells below, ensure you've manually run the updated
# weather_documents_embeddings table schema with chunk support:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nEnsure the weather_documents_embeddings table has been created with:")
print("  - chunk_index column")
print("  - chunk_text column")
print("  - foreign key to weather_documents(id)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Insert chunk embeddings into weather_documents_embeddings
# MAGIC
# MAGIC Batch insert embeddings with psycopg2, storing each chunk's vector alongside
# MAGIC its text and metadata. Duplicates are skipped via ON CONFLICT.

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using psycopg2
import psycopg2
from datetime import datetime

# Add id (weather_document_id_chunk_index), model_name, and embedded_at columns
chunk_embeddings_df['id'] = chunk_embeddings_df['weather_document_id'] + '_' + chunk_embeddings_df['chunk_index'].astype(str)
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['embedded_at'] = datetime.now()
chunk_embeddings_df['chunk_index'] = chunk_embeddings_df['chunk_index'].astype(int)

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {EMBEDDINGS_TABLE_NAME}...")
    
    # Build connection using psycopg2
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Prepare data tuples for batch insert
        # Format embedding as PostgreSQL array literal: '{val1,val2,...}'
        insert_data = [
            (
                row['id'],
                row['weather_document_id'],
                int(row['chunk_index']),
                row['chunk_text'],
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['embedded_at']
            )
            for row in chunk_embeddings_rows
        ]
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, weather_document_id, chunk_index, chunk_text, embedding, model_name, embedded_at
            ) VALUES (%s, %s, %s, %s, %s::double precision[], %s, %s)
            ON CONFLICT (id) DO NOTHING
        """
        
        # executemany in psycopg2 is much faster than individual INSERTs
        cursor.executemany(insert_sql, insert_data)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new chunk embeddings")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
        print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No chunk embeddings to write.")