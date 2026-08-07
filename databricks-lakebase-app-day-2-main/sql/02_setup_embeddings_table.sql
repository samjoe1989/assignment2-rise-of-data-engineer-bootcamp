-- Setup script for ticker_news_embeddings table
-- Run this manually in your Lakebase Postgres database before running the notebook
-- Replace {{EMBEDDING_DIM}} with your model's dimension (e.g., 384 for all-MiniLM-L6-v2)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_documents_embeddings (
    id TEXT PRIMARY KEY,  -- format: {weather_doc_id}_{chunk_index}
    weather_document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (weather_document_id) REFERENCES weather_documents(id),
    UNIQUE (weather_document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id 
    ON weather_documents_embeddings(weather_document_id);
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding 
    ON weather_documents_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_documents_embeddings'
ORDER BY ordinal_position;