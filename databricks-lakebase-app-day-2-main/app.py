"""
Databricks Weather App:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls weather data from the National Weather Service API via weather_client.py

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os

from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)
_w = WorkspaceClient()

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "weather_documents_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Lazy-load the embedding model
_embedding_model = None

def get_embedding_model():
    """Lazy-load the sentence transformer model for vector search."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def ensure_weather_table():
    """
    Create the weather documents table in Lakebase if it doesn't exist yet.
    
    This table stores weather alerts and forecasts from the National Weather
    Service API for vectorization and semantic search.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL,
            headline TEXT,
            narrative_text TEXT,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )
    # Add effective_at column if it doesn't exist (for existing tables)
    lakebase.run_write(
        f"ALTER TABLE {WEATHER_TABLE_NAME} ADD COLUMN IF NOT EXISTS effective_at TIMESTAMPTZ"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Weather data harvesting UI."""
    return render_template("weather.html")


@app.route("/weather/sync", methods=["POST"])
def sync_weather_from_nws():
    """
    Fetch weather alerts and forecasts from the National Weather Service API
    for a set of locations and upsert them into the weather_documents table.
    
    Body (optional JSON): 
        {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    
    Defaults to a few major cities if no locations are supplied.
    """
    ensure_weather_table()
    client = WeatherClient()
    
    body = request.json if request.is_json else {}
    locations = body.get("locations")
    limit = int(body.get("limit", 50))
    
    total = 0
    resolved_locations = []
    
    for location_str in locations:
        if not isinstance(location_str, str) or not location_str.strip():
            continue
        
        # Resolve location to lat/lon
        resolved = client.resolve_location(location_str)
        if not resolved:
            logger.warning(f"Could not resolve location: {location_str}")
            continue
        
        resolved_locations.append(resolved)
        
        lat, lon = resolved["lat"], resolved["lon"]
        location_name = resolved["location"]
        
        # Get grid point for forecasts
        grid = client.get_grid_point(lat, lon)
        
        # Fetch alerts for this location (lat/lon based)
        alerts = client.get_active_alerts(lat, lon, limit=limit)
        total += _upsert_weather_batch(alerts)
        
        # Fetch forecasts if we have a valid grid point
        if grid:
            forecasts = client.get_forecast(
                office=grid["office"],
                grid_x=grid["gridX"],
                grid_y=grid["gridY"],
                location=location_name,
                limit=limit
            )
            total += _upsert_weather_batch(forecasts)
    
    return jsonify({
        "synced": total,
        "locations": [loc["location"] for loc in resolved_locations]
    })


@app.route("/weather/data", methods=["GET"])
def get_weather_data():
    """
    Query weather data from the database.
    
    Query params:
        locations: Comma-separated list of locations to filter by (optional, supports partial match)
        source_type: Filter by source_type (alerts, forecast, hourly_forecast)
        limit: Maximum number of records to return (default 100)
    """
    ensure_weather_table()
    
    locations_param = request.args.get("locations", "").strip()
    source_type = request.args.get("source_type", "").strip()
    limit = int(request.args.get("limit", 100))
    
    # Build query
    where_clauses = []
    params = []
    
    if locations_param:
        # Split comma-separated locations and create OR conditions
        locations = [loc.strip() for loc in locations_param.split(",") if loc.strip()]
        if locations:
            location_conditions = []
            for loc in locations:
                location_conditions.append("location ILIKE %s")
                params.append(f"%{loc}%")
            where_clauses.append(f"({' OR '.join(location_conditions)})")
    
    if source_type:
        where_clauses.append("source_type = %s")
        params.append(source_type)
    
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    query = f"""
        SELECT 
            id,
            location,
            source_type,
            headline,
            narrative_text,
            issued_at,
            synced_at
        FROM {WEATHER_TABLE_NAME}
        WHERE {where_sql}
        ORDER BY issued_at DESC, synced_at DESC
        LIMIT %s
    """
    params.append(limit)
    
    rows = lakebase.run_query(query, tuple(params))
    
    return jsonify({
        "count": len(rows),
        "data": rows
    })


@app.route("/weather/search", methods=["POST"])
def search_weather_embeddings():
    """
    Semantic search over weather document embeddings using cosine similarity.
    
    Body (JSON):
        {
            "query": "severe thunderstorm warnings",
            "limit": 10
        }
    
    Returns the most similar text chunks from weather_documents_embeddings.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    body = request.json
    query_text = body.get("query", "").strip()
    limit = int(body.get("limit", 10))
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    # Generate embedding for the query
    model = get_embedding_model()
    query_embedding = model.encode(query_text).tolist()
    
    # Format embedding as pgvector literal (requires square brackets)
    embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'
    
    # Search using cosine similarity (<=> operator in pgvector)
    # Lower distance = more similar
    query = f"""
        SELECT 
            e.id,
            e.weather_document_id,
            e.chunk_index,
            e.chunk_text,
            e.model_name,
            e.embedded_at,
            e.embedding <=> %s::vector AS distance,
            w.location,
            w.source_type,
            w.headline,
            w.issued_at
        FROM {EMBEDDINGS_TABLE_NAME} e
        LEFT JOIN {WEATHER_TABLE_NAME} w ON e.weather_document_id = w.id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    
    try:
        rows = lakebase.run_query(query, (embedding_str, embedding_str, limit))
        
        return jsonify({
            "query": query_text,
            "count": len(rows),
            "results": rows
        })
    except Exception as e:
        logger.exception("Error during vector search")
        return jsonify({"error": str(e)}), 500


def _upsert_weather_batch(documents: list[dict]) -> int:
    """
    Upsert weather documents (alerts or forecasts) into the weather table.
    
    Args:
        documents: List of database-formatted weather dicts from WeatherClient.
                   Each dict should have: id, location, source_type, headline,
                   narrative_text, issued_at, effective_at (optional), 
                   payload, synced_at
    """
    import json as _json
    
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, source_type, headline,
                        narrative_text, issued_at, effective_at,
                        payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        doc.get("id"),
                        doc.get("location"),
                        doc.get("source_type"),
                        doc.get("headline"),
                        doc.get("narrative_text"),
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        _json.dumps(doc.get("payload", {})),
                        doc.get("synced_at"),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")
