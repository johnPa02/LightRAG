#!/usr/bin/env python3
"""
Script to delete appendix chunks from both JSON storage and Qdrant.

Target: Delete Thông tư 01/2025/TT-BYT PHỤ LỤC I-VI chunks
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "lightrag_vdb_chunks"

# Target chunk IDs to delete (PHỤ LỤC I-IV only)
APPENDIX_CHUNK_IDS = [
    "chunk-8ec5c05262b2e1d73ffa63ded3ec1b94",  # PHỤ LỤC I - Danh mục bệnh khám tại cấp chuyên sâu
    "chunk-94f1e306917bad92d0e339740c967688",  # PHỤ LỤC II - Danh mục bệnh khám tại cấp cơ bản
    "chunk-102f18107d61ca3da8d4267d6262e083",  # PHỤ LỤC III - Danh mục bệnh sử dụng phiếu chuyển 1 năm
    "chunk-5c72a7425552de12e45963578904bfa9",  # PHỤ LỤC IV - Danh mục bệnh chuyển cấp ban đầu để quản lý
]

def load_kv_store(file_path: Path) -> Dict[str, Any]:
    """Load the KV store JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_kv_store(file_path: Path, data: Dict[str, Any]) -> None:
    """Save the KV store JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_chunks_to_delete(kv_store: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List chunks that will be deleted."""
    chunks_to_delete = []
    for chunk_id in APPENDIX_CHUNK_IDS:
        if chunk_id in kv_store:
            chunk = kv_store[chunk_id]
            content_preview = chunk.get('content', '')[:100] + '...' if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
            chunks_to_delete.append({
                'chunk_id': chunk_id,
                'content_preview': content_preview,
                'tokens': chunk.get('tokens', 0),
                'file_path': chunk.get('file_path', 'N/A')
            })
        else:
            print(f"Warning: Chunk {chunk_id} not found in KV store")
    return chunks_to_delete

def delete_from_json(kv_store_path: Path, dry_run: bool = True) -> int:
    """Delete chunks from JSON KV store."""
    kv_store = load_kv_store(kv_store_path)
    deleted_count = 0
    
    for chunk_id in APPENDIX_CHUNK_IDS:
        if chunk_id in kv_store:
            if not dry_run:
                del kv_store[chunk_id]
            deleted_count += 1
            print(f"  {'Would delete' if dry_run else 'Deleted'}: {chunk_id[:30]}...")
    
    if not dry_run and deleted_count > 0:
        save_kv_store(kv_store_path, kv_store)
        print(f"\nSaved updated KV store to {kv_store_path}")
    
    return deleted_count

def delete_from_qdrant(dry_run: bool = True) -> int:
    """Delete chunks from Qdrant vector store."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Collection '{COLLECTION_NAME}' not found in Qdrant.")
            print(f"Available collections: {collection_names}")
            return 0
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"\nQdrant collection '{COLLECTION_NAME}': {collection_info.points_count} points")
        
        # Search for points with matching chunk IDs
        deleted_count = 0
        points_to_delete = []
        
        for chunk_id in APPENDIX_CHUNK_IDS:
            # Search by scrolling to find points with chunk_id in payload
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter={
                    "must": [
                        {"key": "chunk_id", "match": {"value": chunk_id}}
                    ]
                },
                limit=10
            )
            
            points, next_page = results
            if points:
                for point in points:
                    points_to_delete.append(point.id)
                    print(f"  {'Would delete' if dry_run else 'Deleting'} point ID: {point.id} (chunk: {chunk_id[:30]}...)")
                    deleted_count += 1
        
        # Also try searching by _id field
        if deleted_count == 0:
            print("\nTrying to search by '_id' field...")
            for chunk_id in APPENDIX_CHUNK_IDS:
                results = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter={
                        "must": [
                            {"key": "_id", "match": {"value": chunk_id}}
                        ]
                    },
                    limit=10
                )
                
                points, next_page = results
                if points:
                    for point in points:
                        points_to_delete.append(point.id)
                        print(f"  {'Would delete' if dry_run else 'Deleting'} point ID: {point.id} (chunk: {chunk_id[:30]}...)")
                        deleted_count += 1
        
        # If still nothing found, try searching in payload as id
        if deleted_count == 0:
            print("\nTrying to search by 'id' field in payload...")
            for chunk_id in APPENDIX_CHUNK_IDS:
                results = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter={
                        "must": [
                            {"key": "id", "match": {"value": chunk_id}}
                        ]
                    },
                    limit=10
                )
                
                points, next_page = results
                if points:
                    for point in points:
                        points_to_delete.append(point.id)
                        print(f"  {'Would delete' if dry_run else 'Deleting'} point ID: {point.id} (chunk: {chunk_id[:30]}...)")
                        deleted_count += 1
        
        # Actually delete if not dry run
        if not dry_run and points_to_delete:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=points_to_delete)
            )
            print(f"\nDeleted {len(points_to_delete)} points from Qdrant")
        
        return deleted_count
        
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Delete appendix chunks from storage')
    parser.add_argument('--execute', action='store_true', help='Actually delete (default is dry-run)')
    parser.add_argument('--list', action='store_true', help='Only list chunks to be deleted')
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    # Path to KV store
    kv_store_path = project_root / 'data' / 'rag_storage_dev' / 'default' / 'kv_store_text_chunks.json'
    
    print("=" * 60)
    print("DELETE APPENDIX CHUNKS FROM THÔNG TƯ 01/2025/TT-BYT")
    print("=" * 60)
    
    if args.list:
        print("\nChunks to be deleted:")
        print("-" * 60)
        kv_store = load_kv_store(kv_store_path)
        chunks = list_chunks_to_delete(kv_store)
        for chunk in chunks:
            print(f"\nID: {chunk['chunk_id']}")
            print(f"File: {chunk['file_path']}")
            print(f"Tokens: {chunk['tokens']}")
            print(f"Preview: {chunk['content_preview']}")
        print(f"\nTotal: {len(chunks)} chunks")
        return
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("   Use --execute to actually delete")
    else:
        print("\n🔴 EXECUTE MODE - Changes will be permanent!")
    
    # Delete from JSON
    print("\n" + "=" * 40)
    print("STEP 1: Delete from JSON KV Store")
    print("=" * 40)
    json_deleted = delete_from_json(kv_store_path, dry_run)
    print(f"\n{'Would delete' if dry_run else 'Deleted'} {json_deleted} chunks from JSON")
    
    # Delete from Qdrant
    print("\n" + "=" * 40)
    print("STEP 2: Delete from Qdrant")
    print("=" * 40)
    qdrant_deleted = delete_from_qdrant(dry_run)
    print(f"\n{'Would delete' if dry_run else 'Deleted'} {qdrant_deleted} points from Qdrant")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"JSON chunks: {json_deleted}")
    print(f"Qdrant points: {qdrant_deleted}")
    
    if dry_run:
        print("\n⚠️  This was a dry run. Run with --execute to actually delete.")

if __name__ == "__main__":
    main()
