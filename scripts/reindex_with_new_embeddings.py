#!/usr/bin/env python3
"""
Re-index documents with new embedding dimension.

This script:
1. Clears Neo4j graph data for workspace
2. Clears Qdrant collections (if any)
3. Clears KV stores 
4. Re-inserts documents from backup to generate new embeddings

Usage:
    python scripts/reindex_with_new_embeddings.py --backup-dir ./backup_20241219/default
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def clear_neo4j(workspace: str):
    """Clear Neo4j data for workspace."""
    print(f"\n[1/4] Clearing Neo4j data for workspace '{workspace}'...")
    
    from neo4j import GraphDatabase
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "lightrag123")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # Delete all nodes with workspace label
        result = session.run(f"MATCH (n:`{workspace}`) DETACH DELETE n RETURN count(n) as deleted")
        deleted = result.single()["deleted"]
        print(f"  Deleted {deleted} nodes from Neo4j")
    
    driver.close()
    print("  Neo4j cleared!")


async def clear_qdrant(workspace: str):
    """Clear Qdrant collections for workspace."""
    print(f"\n[2/4] Clearing Qdrant data for workspace '{workspace}'...")
    
    try:
        from qdrant_client import QdrantClient, models
        
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)
        
        collections = ["lightrag_entities", "lightrag_relationships", "lightrag_chunks"]
        for coll_name in collections:
            if client.collection_exists(coll_name):
                # Delete points for this workspace
                try:
                    client.delete(
                        collection_name=coll_name,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="__workspace_id__",
                                        match=models.MatchValue(value=workspace),
                                    )
                                ]
                            )
                        ),
                    )
                    print(f"  Cleared workspace data from {coll_name}")
                except Exception as e:
                    print(f"  Warning clearing {coll_name}: {e}")
            else:
                print(f"  Collection {coll_name} doesn't exist (will be created on insert)")
        
        print("  Qdrant cleared!")
    except ImportError:
        print("  Qdrant client not installed, skipping...")


async def clear_kv_stores(data_dir: str):
    """Clear KV store files."""
    print(f"\n[3/4] Clearing KV stores in {data_dir}...")
    
    kv_files = [
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_entity_chunks.json",
        "kv_store_relation_chunks.json",
        "kv_store_text_chunks.json",
        "kv_store_doc_status.json",
    ]
    
    for filename in kv_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            # Write empty dict
            with open(filepath, "w") as f:
                json.dump({}, f)
            print(f"  Cleared {filename}")
    
    print("  KV stores cleared!")


async def reinsert_documents(backup_dir: str, lightrag_url: str):
    """Re-insert documents via LightRAG API."""
    print(f"\n[4/4] Re-inserting documents from {backup_dir}...")
    
    import aiohttp
    
    # Load documents from backup
    docs_file = os.path.join(backup_dir, "kv_store_full_docs.json")
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    print(f"  Found {len(docs)} documents to re-insert")
    
    async with aiohttp.ClientSession() as session:
        for i, (doc_id, doc_data) in enumerate(docs.items(), 1):
            content = doc_data.get("content", "")
            if not content:
                print(f"  [{i}/{len(docs)}] Skipping {doc_id[:20]}... (no content)")
                continue
            
            print(f"  [{i}/{len(docs)}] Inserting {doc_id[:20]}... ({len(content)} chars)")
            
            try:
                async with session.post(
                    f"{lightrag_url}/documents/text",
                    json={"text": content},
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 min timeout
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"    ✓ Success: {result.get('message', 'OK')}")
                    else:
                        error = await resp.text()
                        print(f"    ✗ Failed ({resp.status}): {error[:100]}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
    
    print("\n  Re-insertion complete!")


async def main():
    parser = argparse.ArgumentParser(description="Re-index documents with new embeddings")
    parser.add_argument("--backup-dir", required=True, help="Path to backup directory")
    parser.add_argument("--data-dir", default="./data/rag_storage/default", help="LightRAG data directory")
    parser.add_argument("--workspace", default="default", help="Workspace name")
    parser.add_argument("--lightrag-url", default="http://localhost:9621", help="LightRAG API URL")
    parser.add_argument("--skip-clear", action="store_true", help="Skip clearing existing data")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Re-index Documents with New Embeddings (dim=3072)")
    print("=" * 60)
    print(f"Backup dir: {args.backup_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Workspace: {args.workspace}")
    print(f"LightRAG URL: {args.lightrag_url}")
    print("=" * 60)
    
    if not args.skip_clear:
        # Clear existing data
        await clear_neo4j(args.workspace)
        await clear_qdrant(args.workspace)
        await clear_kv_stores(args.data_dir)
    else:
        print("\nSkipping data clearing...")
    
    # Re-insert documents
    await reinsert_documents(args.backup_dir, args.lightrag_url)
    
    print("\n" + "=" * 60)
    print("✓ Re-indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
