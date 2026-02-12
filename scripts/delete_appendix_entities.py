#!/usr/bin/env python3
"""
Script to delete appendix (Phụ lục) entities from Thông tư 01/2025/TT-BYT
in Neo4j and related chunks from Qdrant.
"""

import asyncio
import os
import sys
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Configuration for DEV environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")  # DEV port
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lightrag123")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6335"))  # DEV port

# Appendix patterns to match
APPENDIX_PATTERNS = [
    "Phụ lục I - Thông tư 01/2025/TT-BYT",
    "Phụ lục II - Thông tư 01/2025/TT-BYT", 
    "Phụ lục III - Thông tư 01/2025/TT-BYT",
    "Phụ lục IV - Thông tư 01/2025/TT-BYT",
    "Phụ lục V - Thông tư 01/2025/TT-BYT",
    "PHỤ LỤC V - Thông tư 01/2025/TT-BYT",
    "Phụ lục VI - Thông tư 01/2025/TT-BYT",
    "Ghi chú - Phụ lục IV - Thông tư 01/2025/TT-BYT",
    "Ghi chú Phụ lục I - Thông tư 01/2025/TT-BYT",
    "Khoản ghi chú - Phụ lục III - Thông tư 01/2025/TT-BYT",
]


def list_appendix_entities(driver):
    """List all entities matching appendix patterns in Neo4j."""
    entities = []
    
    with driver.session() as session:
        # Query to find entities with names containing "Phụ lục" and "01/2025"
        query = """
        MATCH (n)
        WHERE n.entity_id IS NOT NULL 
        AND (
            (toLower(n.entity_id) CONTAINS 'phụ lục' AND toLower(n.entity_id) CONTAINS '01/2025')
            OR (toLower(n.entity_id) CONTAINS 'phu luc' AND toLower(n.entity_id) CONTAINS '01/2025')
        )
        RETURN n.entity_id AS entity_id, labels(n) AS labels
        ORDER BY n.entity_id
        """
        
        result = session.run(query)
        for record in result:
            entities.append({
                "entity_id": record["entity_id"],
                "labels": record["labels"]
            })
    
    return entities


def delete_appendix_entities(driver, dry_run=True):
    """Delete appendix entities from Neo4j."""
    deleted_entities = []
    deleted_relationships = []
    
    with driver.session() as session:
        # Find and delete entities matching the patterns
        query = """
        MATCH (n)
        WHERE n.entity_id IS NOT NULL 
        AND (
            (toLower(n.entity_id) CONTAINS 'phụ lục' AND toLower(n.entity_id) CONTAINS '01/2025')
            OR (toLower(n.entity_id) CONTAINS 'phu luc' AND toLower(n.entity_id) CONTAINS '01/2025')
        )
        RETURN n.entity_id AS entity_id, labels(n) AS labels
        """
        
        result = session.run(query)
        entities_to_delete = [record["entity_id"] for record in result]
        
        if not dry_run and entities_to_delete:
            # Delete relationships first
            delete_rels_query = """
            MATCH (n)-[r]-()
            WHERE n.entity_id IS NOT NULL 
            AND (
                (toLower(n.entity_id) CONTAINS 'phụ lục' AND toLower(n.entity_id) CONTAINS '01/2025')
                OR (toLower(n.entity_id) CONTAINS 'phu luc' AND toLower(n.entity_id) CONTAINS '01/2025')
            )
            DELETE r
            RETURN count(r) AS deleted_rels
            """
            rels_result = session.run(delete_rels_query)
            for record in rels_result:
                deleted_relationships.append(record["deleted_rels"])
            
            # Delete nodes
            delete_nodes_query = """
            MATCH (n)
            WHERE n.entity_id IS NOT NULL 
            AND (
                (toLower(n.entity_id) CONTAINS 'phụ lục' AND toLower(n.entity_id) CONTAINS '01/2025')
                OR (toLower(n.entity_id) CONTAINS 'phu luc' AND toLower(n.entity_id) CONTAINS '01/2025')
            )
            DELETE n
            RETURN count(n) AS deleted_nodes
            """
            nodes_result = session.run(delete_nodes_query)
            for record in nodes_result:
                deleted_entities.append(record["deleted_nodes"])
        
        return entities_to_delete, deleted_entities, deleted_relationships


def main():
    dry_run = "--execute" not in sys.argv
    
    print("=" * 60)
    print("DELETE APPENDIX ENTITIES FROM THÔNG TƯ 01/2025/TT-BYT")
    print("=" * 60)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("   Use --execute to actually delete the data\n")
    else:
        print("\n🔴 EXECUTION MODE - Data will be deleted!\n")
    
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Verify connection
        driver.verify_connectivity()
        print("✅ Connected to Neo4j\n")
        
        # List entities
        print("📋 Searching for appendix entities...")
        entities = list_appendix_entities(driver)
        
        if not entities:
            print("❌ No appendix entities found matching the patterns")
            return
        
        print(f"\n📌 Found {len(entities)} entities to delete:\n")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity['entity_id']}")
            print(f"     Labels: {entity['labels']}")
        
        if not dry_run:
            print("\n🗑️  Deleting entities...")
            entities_to_delete, deleted_nodes, deleted_rels = delete_appendix_entities(driver, dry_run=False)
            print(f"\n✅ Deleted {sum(deleted_nodes)} nodes")
            print(f"✅ Deleted {sum(deleted_rels)} relationships")
        else:
            print("\n⚠️  Run with --execute to delete these entities")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        driver.close()
        print("\n✅ Neo4j connection closed")


if __name__ == "__main__":
    main()
