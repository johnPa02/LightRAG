#!/usr/bin/env python3
"""
Re-embed entities, relations (from Neo4j) and chunks (from JSON) with Voyage AI.

This script:
1. Reads entities/relations from Neo4j
2. Reads chunks from kv_store_text_chunks.json backup
3. Generates new embeddings using Voyage AI (voyage-3-large)
4. Inserts vectors into Qdrant collections

Usage:
    python scripts/reembed_from_neo4j.py --backup-dir ./backup_20241219/default
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

# Install dependencies if needed
try:
    import voyageai
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "voyageai", "-q"])
    import voyageai

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client", "-q"])
    from qdrant_client import QdrantClient, models

try:
    from neo4j import GraphDatabase
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neo4j", "-q"])
    from neo4j import GraphDatabase


# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "pa-Lrmek8QhjK6fhHstsBwDKXeE-5tjmWtDAXfH3Pliw--")
VOYAGE_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lightrag123")

BATCH_SIZE = 64  # Voyage API batch limit
WORKSPACE_ID_FIELD = "__workspace_id__"


def get_entities_from_neo4j(workspace: str) -> List[Dict[str, Any]]:
    """Fetch all entities from Neo4j."""
    print(f"  Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    entities = []
    with driver.session() as session:
        # Get all nodes with workspace label
        result = session.run(f"""
            MATCH (n:`{workspace}`)
            RETURN n.entity_id as entity_id, 
                   n.entity_type as entity_type,
                   n.description as description,
                   n.source_id as source_id
        """)
        for record in result:
            entities.append({
                "entity_id": record["entity_id"] or "",
                "entity_name": record["entity_id"] or "",  # entity_id is the name
                "entity_type": record["entity_type"] or "",
                "description": record["description"] or "",
                "source_id": record["source_id"] or "",
            })
    
    driver.close()
    return entities


def get_relations_from_neo4j(workspace: str) -> List[Dict[str, Any]]:
    """Fetch all relations from Neo4j."""
    print(f"  Fetching relations from Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    relations = []
    with driver.session() as session:
        # Get all relationships
        result = session.run(f"""
            MATCH (a:`{workspace}`)-[r]->(b:`{workspace}`)
            RETURN a.entity_id as src_id,
                   b.entity_id as tgt_id,
                   r.description as description,
                   r.keywords as keywords,
                   r.source_id as source_id
        """)
        for record in result:
            relations.append({
                "src_id": record["src_id"] or "",
                "tgt_id": record["tgt_id"] or "",
                "description": record["description"] or "",
                "keywords": record["keywords"] or "",
                "source_id": record["source_id"] or "",
            })
    
    driver.close()
    return relations


def load_chunks_from_json(file_path: str) -> Dict[str, Any]:
    """Load chunks from JSON backup."""
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} chunks from {os.path.basename(file_path)}")
    return data


def embed_texts_voyage(texts: List[str], client: voyageai.Client) -> List[List[float]]:
    """Embed texts using Voyage AI API."""
    if not texts:
        return []
    
    # Filter out empty strings
    valid_texts = []
    valid_indices = []
    for i, t in enumerate(texts):
        if t and t.strip():
            valid_texts.append(t)
            valid_indices.append(i)
    
    if not valid_texts:
        return [[0.0] * EMBEDDING_DIM] * len(texts)
    
    # Embed in batches
    embeddings_map = {}
    for i in range(0, len(valid_texts), BATCH_SIZE):
        batch = valid_texts[i:i + BATCH_SIZE]
        batch_indices = valid_indices[i:i + BATCH_SIZE]
        
        try:
            result = client.embed(
                texts=batch,
                model=VOYAGE_MODEL,
                input_type="document",
                output_dimension=EMBEDDING_DIM,
            )
            for idx, emb in zip(batch_indices, result.embeddings):
                embeddings_map[idx] = emb
        except Exception as e:
            print(f"  Error embedding batch: {e}")
            for idx in batch_indices:
                embeddings_map[idx] = [0.0] * EMBEDDING_DIM
    
    # Build final list with zeros for empty strings
    final_embeddings = []
    for i in range(len(texts)):
        if i in embeddings_map:
            final_embeddings.append(embeddings_map[i])
        else:
            final_embeddings.append([0.0] * EMBEDDING_DIM)
    
    return final_embeddings


def ensure_collection(client: QdrantClient, collection_name: str, dim: int):
    """Create Qdrant collection if not exists."""
    if client.collection_exists(collection_name):
        # Delete and recreate to ensure correct dimension
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection '{collection_name}'")
    
    print(f"  Creating collection '{collection_name}' (dim={dim})...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    
    # Create workspace index
    client.create_payload_index(
        collection_name=collection_name,
        field_name=WORKSPACE_ID_FIELD,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )


def insert_entities_to_qdrant(
    qdrant_client: QdrantClient,
    entities: List[Dict[str, Any]],
    embeddings: List[List[float]],
    workspace: str,
) -> int:
    """Insert entities with embeddings to Qdrant."""
    points = []
    for entity, embedding in zip(entities, embeddings):
        if not embedding or all(v == 0 for v in embedding):
            continue
        
        payload = {
            WORKSPACE_ID_FIELD: workspace,
            "__id__": entity.get("entity_id", ""),
            "entity_name": entity.get("entity_name", ""),
            "entity_type": entity.get("entity_type", ""),
            "description": entity.get("description", ""),
            "source_id": entity.get("source_id", ""),
        }
        
        points.append(models.PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload,
        ))
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name="lightrag_entities", points=batch)
    
    return len(points)


def insert_relations_to_qdrant(
    qdrant_client: QdrantClient,
    relations: List[Dict[str, Any]],
    embeddings: List[List[float]],
    workspace: str,
) -> int:
    """Insert relations with embeddings to Qdrant."""
    points = []
    for relation, embedding in zip(relations, embeddings):
        if not embedding or all(v == 0 for v in embedding):
            continue
        
        # Create unique relation ID
        rel_id = f"{relation.get('src_id', '')}_{relation.get('tgt_id', '')}"
        
        payload = {
            WORKSPACE_ID_FIELD: workspace,
            "__id__": rel_id,
            "src_id": relation.get("src_id", ""),
            "tgt_id": relation.get("tgt_id", ""),
            "description": relation.get("description", ""),
            "keywords": relation.get("keywords", ""),
            "source_id": relation.get("source_id", ""),
        }
        
        points.append(models.PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload,
        ))
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name="lightrag_relationships", points=batch)
    
    return len(points)


def insert_chunks_to_qdrant(
    qdrant_client: QdrantClient,
    chunks: Dict[str, Any],
    embeddings: List[List[float]],
    workspace: str,
) -> int:
    """Insert chunks with embeddings to Qdrant."""
    points = []
    chunk_list = list(chunks.items())
    
    for (chunk_id, chunk_data), embedding in zip(chunk_list, embeddings):
        if not embedding or all(v == 0 for v in embedding):
            continue
        
        payload = {
            WORKSPACE_ID_FIELD: workspace,
            "__id__": chunk_id,
            "content": chunk_data.get("content", "")[:1000],  # Truncate for payload
            "full_doc_id": chunk_data.get("full_doc_id", ""),
            "chunk_order_index": chunk_data.get("chunk_order_index", 0),
        }
        
        points.append(models.PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload,
        ))
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name="lightrag_chunks", points=batch)
    
    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Re-embed with Voyage AI from Neo4j")
    parser.add_argument("--backup-dir", required=True, help="Path to backup directory (for chunks)")
    parser.add_argument("--workspace", default="default", help="Workspace name")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Re-embed from Neo4j with Voyage AI → Qdrant")
    print("=" * 60)
    print(f"Backup dir: {args.backup_dir}")
    print(f"Workspace: {args.workspace}")
    print(f"Voyage model: {VOYAGE_MODEL}")
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Neo4j URI: {NEO4J_URI}")
    print("=" * 60)
    
    # Initialize clients
    print("\nInitializing clients...")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    qdrant_client = QdrantClient(url=args.qdrant_url)
    
    # Verify connections
    try:
        qdrant_client.get_collections()
        print("  ✓ Qdrant connected")
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {e}")
        sys.exit(1)
    
    # Load data
    print("\n[1/6] Loading entities from Neo4j...")
    entities = get_entities_from_neo4j(args.workspace)
    print(f"  Found {len(entities)} entities")
    
    print("\n[2/6] Loading relations from Neo4j...")
    relations = get_relations_from_neo4j(args.workspace)
    print(f"  Found {len(relations)} relations")
    
    print("\n[3/6] Loading chunks from backup...")
    chunks = load_chunks_from_json(os.path.join(args.backup_dir, "kv_store_text_chunks.json"))
    
    # Process entities
    print("\n[4/6] Embedding and inserting entities...")
    ensure_collection(qdrant_client, "lightrag_entities", EMBEDDING_DIM)
    entity_texts = [f"{e['entity_name']}: {e['description']}" if e['description'] else e['entity_name'] for e in entities]
    print(f"  Embedding {len(entity_texts)} entities...")
    entity_embeddings = embed_texts_voyage(entity_texts, voyage_client)
    count = insert_entities_to_qdrant(qdrant_client, entities, entity_embeddings, args.workspace)
    print(f"  ✓ Inserted {count} entities")
    
    # Process relations
    print("\n[5/6] Embedding and inserting relations...")
    ensure_collection(qdrant_client, "lightrag_relationships", EMBEDDING_DIM)
    relation_texts = []
    for r in relations:
        text = f"{r['src_id']} -> {r['tgt_id']}"
        if r['description']:
            text += f": {r['description']}"
        if r['keywords']:
            text += f" ({r['keywords']})"
        relation_texts.append(text)
    print(f"  Embedding {len(relation_texts)} relations...")
    relation_embeddings = embed_texts_voyage(relation_texts, voyage_client)
    count = insert_relations_to_qdrant(qdrant_client, relations, relation_embeddings, args.workspace)
    print(f"  ✓ Inserted {count} relations")
    
    # Process chunks
    print("\n[6/6] Embedding and inserting chunks...")
    ensure_collection(qdrant_client, "lightrag_chunks", EMBEDDING_DIM)
    chunk_texts = [c.get("content", "") for c in chunks.values()]
    print(f"  Embedding {len(chunk_texts)} chunks...")
    chunk_embeddings = embed_texts_voyage(chunk_texts, voyage_client)
    count = insert_chunks_to_qdrant(qdrant_client, chunks, chunk_embeddings, args.workspace)
    print(f"  ✓ Inserted {count} chunks")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Re-embedding complete!")
    print("=" * 60)
    
    # Show collection stats
    print("\nQdrant collection stats:")
    for name in ["lightrag_entities", "lightrag_relationships", "lightrag_chunks"]:
        info = qdrant_client.get_collection(name)
        print(f"  {name}: {info.points_count} points, dim={info.config.params.vectors.size}")


if __name__ == "__main__":
    main()
