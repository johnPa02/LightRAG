#!/usr/bin/env python3
"""
Re-embed entities, relations, and chunks with Voyage AI and insert to Qdrant.

This script:
1. Reads entities/relations/chunks from KV store backup
2. Generates new embeddings using Voyage AI (voyage-3-large)
3. Inserts vectors into Qdrant collections

Usage:
    python scripts/reembed_with_voyage.py --backup-dir ./backup_20241219/default
"""

import os
import sys
import json
import argparse
import asyncio
from typing import List, Dict, Any, Optional
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


# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "pa-Lrmek8QhjK6fhHstsBwDKXeE-5tjmWtDAXfH3Pliw--")
VOYAGE_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
BATCH_SIZE = 64  # Voyage API limit
WORKSPACE_ID_FIELD = "__workspace_id__"


def load_kv_store(file_path: str) -> Dict[str, Any]:
    """Load KV store JSON file."""
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} items from {os.path.basename(file_path)}")
    return data


def get_text_for_embedding(item: Dict[str, Any], item_type: str) -> str:
    """Extract text to embed from an item."""
    if item_type == "entity":
        # Entity: combine name and description
        name = item.get("entity_name", item.get("name", ""))
        desc = item.get("description", "")
        return f"{name}: {desc}" if desc else name
    
    elif item_type == "relation":
        # Relation: combine src, tgt, description
        src = item.get("src_id", "")
        tgt = item.get("tgt_id", "")
        desc = item.get("description", "")
        keywords = item.get("keywords", "")
        text = f"{src} -> {tgt}"
        if desc:
            text += f": {desc}"
        if keywords:
            text += f" ({keywords})"
        return text
    
    elif item_type == "chunk":
        # Chunk: use content
        return item.get("content", "")
    
    return ""


def embed_texts_voyage(texts: List[str], client: voyageai.Client) -> List[List[float]]:
    """Embed texts using Voyage AI API."""
    if not texts:
        return []
    
    # Voyage API has batch limit, split if needed
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        try:
            result = client.embed(
                texts=batch,
                model=VOYAGE_MODEL,
                input_type="document",
                output_dimension=EMBEDDING_DIM,
            )
            all_embeddings.extend(result.embeddings)
        except Exception as e:
            print(f"  Error embedding batch {i//BATCH_SIZE + 1}: {e}")
            # Return zeros for failed batch
            all_embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))
    
    return all_embeddings


def ensure_collection(client: QdrantClient, collection_name: str, dim: int):
    """Create Qdrant collection if not exists."""
    if client.collection_exists(collection_name):
        print(f"  Collection '{collection_name}' already exists")
        return
    
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


def insert_to_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    items: Dict[str, Any],
    embeddings: List[List[float]],
    workspace: str,
    item_type: str,
):
    """Insert items with embeddings to Qdrant."""
    if not items:
        return 0
    
    points = []
    item_list = list(items.items())
    
    for (item_id, item_data), embedding in zip(item_list, embeddings):
        if not embedding or all(v == 0 for v in embedding):
            continue
        
        # Build payload
        payload = {
            WORKSPACE_ID_FIELD: workspace,
            "__original_id__": item_id,
            "__id__": item_id,
        }
        
        # Add relevant fields based on type
        if item_type == "entity":
            payload["entity_name"] = item_data.get("entity_name", item_data.get("name", ""))
            payload["entity_type"] = item_data.get("entity_type", item_data.get("type", ""))
            payload["description"] = item_data.get("description", "")
            payload["source_id"] = item_data.get("source_id", "")
        elif item_type == "relation":
            payload["src_id"] = item_data.get("src_id", "")
            payload["tgt_id"] = item_data.get("tgt_id", "")
            payload["description"] = item_data.get("description", "")
            payload["keywords"] = item_data.get("keywords", "")
            payload["source_id"] = item_data.get("source_id", "")
        elif item_type == "chunk":
            payload["content"] = item_data.get("content", "")[:1000]  # Truncate for payload
            payload["full_doc_id"] = item_data.get("full_doc_id", "")
            payload["chunk_order_index"] = item_data.get("chunk_order_index", 0)
        
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=payload,
            )
        )
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name=collection_name, points=batch)
    
    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Re-embed with Voyage AI and insert to Qdrant")
    parser.add_argument("--backup-dir", required=True, help="Path to backup directory")
    parser.add_argument("--workspace", default="default", help="Workspace name")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Re-embed with Voyage AI → Qdrant")
    print("=" * 60)
    print(f"Backup dir: {args.backup_dir}")
    print(f"Workspace: {args.workspace}")
    print(f"Voyage model: {VOYAGE_MODEL}")
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Qdrant URL: {args.qdrant_url}")
    print("=" * 60)
    
    # Initialize clients
    print("\nInitializing clients...")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    qdrant_client = QdrantClient(url=args.qdrant_url)
    
    # Verify Qdrant connection
    try:
        qdrant_client.get_collections()
        print("  ✓ Qdrant connected")
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {e}")
        sys.exit(1)
    
    # Load data from backup
    print("\nLoading data from backup...")
    entities = load_kv_store(os.path.join(args.backup_dir, "kv_store_full_entities.json"))
    relations = load_kv_store(os.path.join(args.backup_dir, "kv_store_full_relations.json"))
    chunks = load_kv_store(os.path.join(args.backup_dir, "kv_store_text_chunks.json"))
    
    # Process entities
    print("\n[1/3] Processing entities...")
    ensure_collection(qdrant_client, "lightrag_entities", EMBEDDING_DIM)
    entity_texts = [get_text_for_embedding(e, "entity") for e in entities.values()]
    print(f"  Embedding {len(entity_texts)} entities...")
    entity_embeddings = embed_texts_voyage(entity_texts, voyage_client)
    count = insert_to_qdrant(qdrant_client, "lightrag_entities", entities, entity_embeddings, args.workspace, "entity")
    print(f"  ✓ Inserted {count} entities")
    
    # Process relations
    print("\n[2/3] Processing relations...")
    ensure_collection(qdrant_client, "lightrag_relationships", EMBEDDING_DIM)
    relation_texts = [get_text_for_embedding(r, "relation") for r in relations.values()]
    print(f"  Embedding {len(relation_texts)} relations...")
    relation_embeddings = embed_texts_voyage(relation_texts, voyage_client)
    count = insert_to_qdrant(qdrant_client, "lightrag_relationships", relations, relation_embeddings, args.workspace, "relation")
    print(f"  ✓ Inserted {count} relations")
    
    # Process chunks
    print("\n[3/3] Processing chunks...")
    ensure_collection(qdrant_client, "lightrag_chunks", EMBEDDING_DIM)
    chunk_texts = [get_text_for_embedding(c, "chunk") for c in chunks.values()]
    print(f"  Embedding {len(chunk_texts)} chunks...")
    chunk_embeddings = embed_texts_voyage(chunk_texts, voyage_client)
    count = insert_to_qdrant(qdrant_client, "lightrag_chunks", chunks, chunk_embeddings, args.workspace, "chunk")
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
