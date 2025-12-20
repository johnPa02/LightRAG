#!/usr/bin/env python3
"""
Re-embed all data using Voyage AI SDK with output_dimension=2048
"""
import json
import os
import sys
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
import voyageai
import time

# Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lightrag123")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "pa-Lrmek8QhjK6fhHstsBwDKXeE-5tjmWtDAXfH3Pliw--")
BACKUP_DIR = "/root/projects/LightRAG/backup_20241219/default"
WORKSPACE = "default"
EMBEDDING_DIM = 1024
BATCH_SIZE = 64  # Voyage limit

def get_voyage_embeddings(vo_client, texts, model="voyage-3-large", dim=2048):
    """Get embeddings from Voyage AI with output_dimension"""
    if not texts:
        return []
    
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        result = vo_client.embed(batch, model=model, output_dimension=dim)
        all_embeddings.extend(result.embeddings)
        if i > 0 and i % 200 == 0:
            print(f"  Embedded {i}/{len(texts)}...")
            time.sleep(0.1)  # Rate limit
    
    return all_embeddings


def main():
    print("=== Re-embed with Voyage AI (dim=2048) ===\n")
    
    # Init clients
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL)
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Delete old collections
    for name in ["lightrag_vdb_entities", "lightrag_vdb_relationships", "lightrag_vdb_chunks"]:
        if qdrant.collection_exists(name):
            qdrant.delete_collection(name)
            print(f"Deleted old collection: {name}")
    
    # Create new collections
    for name in ["lightrag_vdb_entities", "lightrag_vdb_relationships", "lightrag_vdb_chunks"]:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
        qdrant.create_payload_index(
            collection_name=name,
            field_name="__workspace_id__",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        print(f"Created collection: {name}")
    
    print()
    
    # === ENTITIES ===
    print("Processing entities from Neo4j...")
    with neo4j_driver.session() as session:
        result = session.run(f"MATCH (n:`{WORKSPACE}`) RETURN n")
        entities = []
        for record in result:
            node = record["n"]
            entities.append({
                "id": node.get("entity_id", node.get("id", "")),
                "entity_name": node.get("entity_id", ""),
                "entity_type": node.get("entity_type", ""),
                "description": node.get("description", ""),
                "source_id": node.get("source_id", ""),
            })
    
    print(f"Found {len(entities)} entities")
    
    # Create embeddings for entities - use same format as LightRAG: "{entity_name}\n{description}"
    entity_texts = [
        f"{e['entity_name']}\n{e['description']}"
        for e in entities
    ]
    
    print("Embedding entities...")
    entity_embeddings = get_voyage_embeddings(vo, entity_texts, dim=EMBEDDING_DIM)
    
    # Insert to Qdrant
    points = []
    for i, (entity, emb) in enumerate(zip(entities, entity_embeddings)):
        import uuid
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "__workspace_id__": WORKSPACE,
                "__id__": entity["id"],
                "entity_name": entity["entity_name"],
                "entity_type": entity["entity_type"],
                "description": entity["description"],
                "source_id": entity["source_id"],
            }
        ))
    
    # Batch insert
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        qdrant.upsert(collection_name="lightrag_vdb_entities", points=batch)
    print(f"Inserted {len(points)} entities to Qdrant")
    
    # === RELATIONSHIPS ===
    print("\nProcessing relationships from Neo4j...")
    with neo4j_driver.session() as session:
        result = session.run(f"""
            MATCH (a:`{WORKSPACE}`)-[r]->(b:`{WORKSPACE}`)
            RETURN a.entity_id as src, type(r) as rel_type, b.entity_id as tgt,
                   r.description as description, r.source_id as source_id,
                   r.weight as weight, r.keywords as keywords
        """)
        relationships = []
        for record in result:
            relationships.append({
                "src_id": record["src"],
                "tgt_id": record["tgt"],
                "description": record["description"] or "",
                "keywords": record["keywords"] or "",
                "weight": record["weight"] or 1.0,
                "source_id": record["source_id"] or "",
            })
    
    print(f"Found {len(relationships)} relationships")
    
    # Create embeddings - use same format as LightRAG: "{keywords}\t{src}\n{tgt}\n{description}"
    rel_texts = [
        f"{r['keywords']}\t{r['src_id']}\n{r['tgt_id']}\n{r['description']}"
        for r in relationships
    ]
    
    print("Embedding relationships...")
    rel_embeddings = get_voyage_embeddings(vo, rel_texts, dim=EMBEDDING_DIM)
    
    # Insert to Qdrant
    points = []
    for rel, emb in zip(relationships, rel_embeddings):
        import uuid
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "__workspace_id__": WORKSPACE,
                "__id__": f"{rel['src_id']}_{rel['tgt_id']}",
                "src_id": rel["src_id"],
                "tgt_id": rel["tgt_id"],
                "description": rel["description"],
                "keywords": rel["keywords"],
                "weight": rel["weight"],
                "source_id": rel["source_id"],
            }
        ))
    
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        qdrant.upsert(collection_name="lightrag_vdb_relationships", points=batch)
    print(f"Inserted {len(points)} relationships to Qdrant")
    
    # === CHUNKS ===
    print("\nProcessing chunks from backup...")
    chunks_file = os.path.join(BACKUP_DIR, "kv_store_text_chunks.json")
    with open(chunks_file, "r") as f:
        chunks_data = json.load(f)
    
    chunks = []
    for chunk_id, data in chunks_data.items():
        if isinstance(data, dict):
            chunks.append({
                "id": chunk_id,
                "content": data.get("content", ""),
                "full_doc_id": data.get("full_doc_id", ""),
            })
    
    print(f"Found {len(chunks)} chunks")
    
    # Create embeddings
    chunk_texts = [c["content"][:2000] for c in chunks]
    
    print("Embedding chunks...")
    chunk_embeddings = get_voyage_embeddings(vo, chunk_texts, dim=EMBEDDING_DIM)
    
    # Insert to Qdrant
    points = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        import uuid
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "__workspace_id__": WORKSPACE,
                "__id__": chunk["id"],
                "content": chunk["content"][:1000],
                "full_doc_id": chunk["full_doc_id"],
            }
        ))
    
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        qdrant.upsert(collection_name="lightrag_vdb_chunks", points=batch)
    print(f"Inserted {len(points)} chunks to Qdrant")
    
    # Verify
    print("\n=== Final Status ===")
    for name in ["lightrag_vdb_entities", "lightrag_vdb_relationships", "lightrag_vdb_chunks"]:
        info = qdrant.get_collection(name)
        print(f"{name}: {info.points_count} points, dim={info.config.params.vectors.size}")
    
    neo4j_driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
