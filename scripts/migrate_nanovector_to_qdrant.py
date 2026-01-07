#!/usr/bin/env python3
"""
Migration script: NanoVectorDB → Qdrant

This script migrates vector data from NanoVectorDB (JSON files) to Qdrant.
It reads vdb_entities.json, vdb_relationships.json, vdb_chunks.json and imports to Qdrant.

Usage:
    python scripts/migrate_nanovector_to_qdrant.py [--data-dir PATH] [--workspace WORKSPACE]

Environment variables (or set in .env):
    QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY: Qdrant API key (optional)
"""

import os
import sys
import json
import base64
import zlib
import argparse
from typing import Optional, List
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    print("Error: qdrant-client package not installed. Run: pip install qdrant-client")
    sys.exit(1)


# Configuration
BATCH_SIZE = 100
WORKSPACE_ID_FIELD = "__workspace_id__"

# Collection names (shared across workspaces)
COLLECTION_NAMES = {
    "entities": "lightrag_entities",
    "relationships": "lightrag_relationships",
    "chunks": "lightrag_chunks",
}

# Source file names
SOURCE_FILES = {
    "entities": "vdb_entities.json",
    "relationships": "vdb_relationships.json",
    "chunks": "vdb_chunks.json",
}


def load_nanovector_data(file_path: str) -> dict:
    """Load NanoVectorDB JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return {"embedding_dim": 0, "data": []}
    
    print(f"Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"  - Embedding dim: {data.get('embedding_dim', 'N/A')}")
    print(f"  - Records: {len(data.get('data', []))}")
    return data


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dim: int,
    workspace: str,
):
    """Ensure Qdrant collection exists with proper configuration."""
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists")
        # Ensure workspace index exists
        try:
            collection_info = client.get_collection(collection_name)
            if WORKSPACE_ID_FIELD not in collection_info.payload_schema:
                print(f"  Creating workspace index...")
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=WORKSPACE_ID_FIELD,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                        is_tenant=True,
                    ),
                )
        except Exception as e:
            print(f"  Warning: Could not verify workspace index: {e}")
        return
    
    print(f"Creating collection '{collection_name}' with dim={embedding_dim}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
    )
    
    # Create workspace index for multi-tenancy
    client.create_payload_index(
        collection_name=collection_name,
        field_name=WORKSPACE_ID_FIELD,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )
    print(f"  Collection '{collection_name}' created successfully")


def decode_vector(vector_data) -> Optional[List[float]]:
    """Decode vector from NanoVectorDB format (base64+zlib compressed numpy array)."""
    if not vector_data:
        return None
    
    if isinstance(vector_data, str):
        try:
            decoded = base64.b64decode(vector_data)
            decompressed = zlib.decompress(decoded)
            return np.frombuffer(decompressed, dtype=np.float32).tolist()
        except Exception as e:
            print(f"  Warning: Could not decode vector: {e}")
            return None
    else:
        return vector_data


def detect_embedding_dim(records: list) -> int:
    """Detect actual embedding dimension from first valid vector."""
    for record in records[:10]:  # Check first 10 records
        vector_data = record.get("vector")
        if vector_data:
            vector = decode_vector(vector_data)
            if vector:
                return len(vector)
    return 0


def migrate_collection(
    client: QdrantClient,
    collection_name: str,
    data: dict,
    workspace: str,
    clear_existing: bool = False,
):
    """Migrate data to Qdrant collection."""
    records = data.get("data", [])
    
    if not records:
        print(f"No records to migrate for '{collection_name}'")
        return 0
    
    # Detect actual embedding dimension from vectors (not from metadata)
    embedding_dim = detect_embedding_dim(records)
    if embedding_dim == 0:
        print(f"Warning: Could not detect embedding dimension for '{collection_name}'")
        return 0
    
    print(f"  Detected actual embedding dim: {embedding_dim}")
    
    # Ensure collection exists
    ensure_collection(client, collection_name, embedding_dim, workspace)
    
    # Clear existing data for this workspace if requested
    if clear_existing:
        print(f"Clearing existing data for workspace '{workspace}' in '{collection_name}'...")
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=WORKSPACE_ID_FIELD,
                                match=models.MatchValue(value=workspace),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            print(f"  Warning: Could not clear existing data: {e}")
    
    # Migrate in batches
    total = len(records)
    migrated = 0
    skipped = 0
    
    print(f"Migrating {total} records to '{collection_name}'...")
    
    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        points = []
        
        for record in batch:
            # Decode vector using helper function
            vector = decode_vector(record.get("vector"))
            if not vector:
                skipped += 1
                continue
            
            # Build payload (all fields except vector)
            payload = {k: v for k, v in record.items() if k != "vector"}
            
            # Add workspace identifier
            payload[WORKSPACE_ID_FIELD] = workspace
            
            # Use existing ID in payload, generate UUID for Qdrant point ID
            original_id = record.get("__id__", str(uuid4()))
            payload["__original_id__"] = original_id
            
            # Generate a valid UUID for Qdrant (Qdrant requires UUID or integer)
            point_id = str(uuid4())
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
        
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points,
            )
            migrated += len(points)
        
        print(f"  Processed {min(i + BATCH_SIZE, total)}/{total} ({migrated} migrated)")
    
    return migrated


def migrate_to_qdrant(
    data_dir: str,
    workspace: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str] = None,
    clear_existing: bool = False,
):
    """Main migration function."""
    print(f"\n{'='*60}")
    print("Migration: NanoVectorDB → Qdrant")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Workspace: {workspace}")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Clear existing: {clear_existing}")
    print(f"{'='*60}\n")
    
    # Connect to Qdrant
    print("Connecting to Qdrant...")
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    
    # Verify connection
    try:
        client.get_collections()
        print("Connected successfully!\n")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return False
    
    # Migrate each collection
    total_migrated = 0
    
    for data_type, source_file in SOURCE_FILES.items():
        file_path = os.path.join(data_dir, source_file)
        collection_name = COLLECTION_NAMES[data_type]
        
        print(f"\n--- Migrating {data_type} ---")
        data = load_nanovector_data(file_path)
        
        if data.get("data"):
            migrated = migrate_collection(
                client=client,
                collection_name=collection_name,
                data=data,
                workspace=workspace,
                clear_existing=clear_existing,
            )
            total_migrated += migrated
            print(f"  ✓ Migrated {migrated} {data_type}")
        else:
            print(f"  - Skipped (no data)")
    
    # Verify migration
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    
    for data_type, collection_name in COLLECTION_NAMES.items():
        try:
            if client.collection_exists(collection_name):
                # Count points for this workspace
                count = client.count(
                    collection_name=collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=WORKSPACE_ID_FIELD,
                                match=models.MatchValue(value=workspace),
                            )
                        ]
                    ),
                )
                print(f"  {collection_name}: {count.count} points")
            else:
                print(f"  {collection_name}: (not created)")
        except Exception as e:
            print(f"  {collection_name}: Error - {e}")
    
    print(f"\n{'='*60}")
    print(f"Migration completed! Total: {total_migrated} records")
    print(f"{'='*60}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NanoVectorDB to Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate default workspace
    python scripts/migrate_nanovector_to_qdrant.py --data-dir ./data/rag_storage/default

    # Migrate with custom workspace
    python scripts/migrate_nanovector_to_qdrant.py --data-dir ./data/rag_storage/default --workspace myworkspace

    # Clear existing data before migration
    python scripts/migrate_nanovector_to_qdrant.py --data-dir ./data/rag_storage/default --clear
        """
    )
    
    parser.add_argument(
        "--data-dir",
        default="./data/rag_storage/default",
        help="Path to directory containing vdb_*.json files"
    )
    parser.add_argument(
        "--workspace",
        default="default",
        help="Workspace name for data isolation in Qdrant"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data for the workspace before migration"
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant API key (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    success = migrate_to_qdrant(
        data_dir=args.data_dir,
        workspace=args.workspace,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        clear_existing=args.clear,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
