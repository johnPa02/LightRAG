#!/usr/bin/env python3
"""
Migration script: NetworkX GraphML → Neo4j

This script migrates graph data from NetworkX (GraphML format) to Neo4j.
It reads the graph_chunk_entity_relation.graphml file and imports nodes/edges to Neo4j.

Usage:
    python scripts/migrate_networkx_to_neo4j.py [--graphml-path PATH] [--workspace WORKSPACE]

Environment variables (or set in .env):
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USERNAME: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (required)
    NEO4J_DATABASE: Neo4j database (default: neo4j)
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j package not installed. Run: pip install neo4j")
    sys.exit(1)


# Configuration
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100


def parse_graphml(xml_file: str) -> dict:
    """Parse GraphML file and extract nodes and edges."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # GraphML namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        # Parse nodes
        for node in root.findall(".//node", namespace):
            node_id = node.get("id")
            if node_id:
                node_id = node_id.strip('"')
            
            node_data = {
                "id": node_id,
                "entity_type": "",
                "description": "",
                "source_id": "",
            }
            
            # Extract node attributes from data elements
            for data_elem in node.findall("./data", namespace):
                key = data_elem.get("key")
                text = data_elem.text if data_elem.text else ""
                
                if key == "d0":  # entity_id
                    node_data["entity_id"] = text.strip('"')
                elif key == "d1":  # entity_type
                    node_data["entity_type"] = text.strip('"')
                elif key == "d2":  # description
                    node_data["description"] = text
                elif key == "d3":  # source_id
                    node_data["source_id"] = text
                elif key == "d4":  # file_path
                    node_data["file_path"] = text

            data["nodes"].append(node_data)

        # Parse edges
        for edge in root.findall(".//edge", namespace):
            source = edge.get("source")
            target = edge.get("target")
            
            if source:
                source = source.strip('"')
            if target:
                target = target.strip('"')

            edge_data = {
                "source": source,
                "target": target,
                "weight": 1.0,
                "description": "",
                "keywords": "",
                "source_id": "",
            }
            
            # Extract edge attributes
            for data_elem in edge.findall("./data", namespace):
                key = data_elem.get("key")
                text = data_elem.text if data_elem.text else ""
                
                if key == "d5":  # weight
                    try:
                        edge_data["weight"] = float(text)
                    except (ValueError, TypeError):
                        edge_data["weight"] = 1.0
                elif key == "d6":  # description
                    edge_data["description"] = text
                elif key == "d7":  # keywords or relation_id
                    edge_data["relation_id"] = text
                elif key == "d8":  # source_id
                    edge_data["source_id"] = text
                elif key == "d9":  # keywords
                    edge_data["keywords"] = text
                elif key == "d10":  # file_path
                    edge_data["file_path"] = text

            data["edges"].append(edge_data)

        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")
        return data

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return {"nodes": [], "edges": []}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"nodes": [], "edges": []}


def process_in_batches(tx, query: str, data: list, batch_size: int, param_name: str = "batch"):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tx.run(query, {param_name: batch})
        print(f"  Processed {min(i + batch_size, len(data))}/{len(data)}")


def migrate_to_neo4j(
    graphml_path: str,
    workspace: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    clear_existing: bool = False,
):
    """Migrate GraphML data to Neo4j."""
    
    print(f"\n{'='*60}")
    print(f"Migration: NetworkX GraphML → Neo4j")
    print(f"{'='*60}")
    print(f"GraphML file: {graphml_path}")
    print(f"Workspace: {workspace}")
    print(f"Neo4j URI: {neo4j_uri}")
    print(f"Neo4j Database: {neo4j_database}")
    print(f"{'='*60}\n")

    # Check if file exists
    if not os.path.exists(graphml_path):
        print(f"Error: GraphML file not found: {graphml_path}")
        return False

    # Parse GraphML
    print("Step 1: Parsing GraphML file...")
    graph_data = parse_graphml(graphml_path)
    
    if not graph_data["nodes"]:
        print("Warning: No nodes found in GraphML file")
        return False

    # Connect to Neo4j
    print("\nStep 2: Connecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    try:
        # Verify connection
        driver.verify_connectivity()
        print("Connected successfully!")

        # LightRAG uses workspace directly as label (no prefix)
        workspace_label = workspace
        
        with driver.session(database=neo4j_database) as session:
            # Optionally clear existing data for this workspace
            if clear_existing:
                print(f"\nStep 2.1: Clearing existing data for workspace '{workspace}'...")
                session.run(f"MATCH (n:`{workspace_label}`) DETACH DELETE n")
                print("Cleared existing data.")

            # Create indexes
            print(f"\nStep 3: Creating indexes for workspace '{workspace}'...")
            try:
                session.run(f"""
                    CREATE INDEX IF NOT EXISTS FOR (n:`{workspace_label}`) ON (n.entity_id)
                """)
                print("Index created on entity_id")
            except Exception as e:
                print(f"Index may already exist: {e}")

            # Import nodes
            print(f"\nStep 4: Importing {len(graph_data['nodes'])} nodes...")
            create_nodes_query = f"""
            UNWIND $batch AS node
            MERGE (n:`{workspace_label}` {{entity_id: node.id}})
            SET n.entity_type = node.entity_type,
                n.description = node.description,
                n.source_id = node.source_id,
                n.displayName = node.id
            """
            
            session.execute_write(
                process_in_batches,
                create_nodes_query,
                graph_data["nodes"],
                BATCH_SIZE_NODES,
                "batch"
            )
            print("Nodes imported successfully!")

            # Import edges
            print(f"\nStep 5: Importing {len(graph_data['edges'])} edges...")
            create_edges_query = f"""
            UNWIND $batch AS edge
            MATCH (source:`{workspace_label}` {{entity_id: edge.source}})
            MATCH (target:`{workspace_label}` {{entity_id: edge.target}})
            MERGE (source)-[r:DIRECTED]-(target)
            SET r.weight = edge.weight,
                r.description = edge.description,
                r.keywords = edge.keywords,
                r.source_id = edge.source_id
            """
            
            session.execute_write(
                process_in_batches,
                create_edges_query,
                graph_data["edges"],
                BATCH_SIZE_EDGES,
                "batch"
            )
            print("Edges imported successfully!")

            # Verify migration
            print("\nStep 6: Verifying migration...")
            result = session.run(f"""
                MATCH (n:`{workspace_label}`)
                WITH count(n) as node_count
                OPTIONAL MATCH ()-[r]->()
                WHERE startNode(r):`{workspace_label}`
                RETURN node_count, count(r) as edge_count
            """)
            record = result.single()
            print(f"Neo4j now contains: {record['node_count']} nodes, {record['edge_count']} edges")

        print(f"\n{'='*60}")
        print("Migration completed successfully!")
        print(f"{'='*60}\n")
        return True

    except Exception as e:
        print(f"Error during migration: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        driver.close()


def find_graphml_files(base_path: str) -> list:
    """Find all GraphML files in the given path."""
    graphml_files = []
    
    if os.path.isfile(base_path) and base_path.endswith('.graphml'):
        return [base_path]
    
    if os.path.isdir(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.graphml'):
                    graphml_files.append(os.path.join(root, file))
    
    return graphml_files


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NetworkX GraphML to Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate specific file with default workspace
    python scripts/migrate_networkx_to_neo4j.py --graphml-path ./data/rag_storage/graph_chunk_entity_relation.graphml

    # Migrate with custom workspace
    python scripts/migrate_networkx_to_neo4j.py --graphml-path ./dickens/graph_chunk_entity_relation.graphml --workspace dickens

    # Clear existing data before migration
    python scripts/migrate_networkx_to_neo4j.py --graphml-path ./data/rag_storage/graph_chunk_entity_relation.graphml --clear
        """
    )
    
    parser.add_argument(
        "--graphml-path",
        default="./data/rag_storage/graph_chunk_entity_relation.graphml",
        help="Path to GraphML file or directory containing GraphML files"
    )
    parser.add_argument(
        "--workspace",
        default="default",
        help="Workspace name (used as label prefix in Neo4j)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data for the workspace before migration"
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password"
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database"
    )

    args = parser.parse_args()

    if not args.neo4j_password:
        print("Error: NEO4J_PASSWORD environment variable or --neo4j-password argument required")
        sys.exit(1)

    # Find GraphML files
    graphml_files = find_graphml_files(args.graphml_path)
    
    if not graphml_files:
        print(f"No GraphML files found at: {args.graphml_path}")
        sys.exit(1)

    print(f"Found {len(graphml_files)} GraphML file(s) to migrate")
    
    success_count = 0
    for graphml_file in graphml_files:
        # Derive workspace from directory name if not specified
        workspace = args.workspace
        if workspace == "default" and len(graphml_files) > 1:
            workspace = os.path.basename(os.path.dirname(graphml_file))
        
        success = migrate_to_neo4j(
            graphml_path=graphml_file,
            workspace=workspace,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            clear_existing=args.clear,
        )
        
        if success:
            success_count += 1

    print(f"\nMigration summary: {success_count}/{len(graphml_files)} files migrated successfully")
    sys.exit(0 if success_count == len(graphml_files) else 1)


if __name__ == "__main__":
    main()
