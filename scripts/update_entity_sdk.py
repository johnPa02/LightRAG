#!/usr/bin/env python3
"""
Script to update entity description using LightRAG SDK directly
Run this inside the Docker container
"""
import asyncio
import os
import sys

# Set environment variables
os.environ["NEO4J_URI"] = "bolt://neo4j:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "lightrag123"

# Add lightrag to path
sys.path.insert(0, '/app')

from lightrag import LightRAG
from lightrag.utils import setup_logger

# Setup logger
setup_logger("lightrag", level="INFO")

WORKING_DIR = "/app/data/rag_storage"

async def update_dieu_26():
    """Update entity Điều 26 - Luật Doanh nghiệp 2020"""
    
    print("Initializing LightRAG...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
    )
    await rag.initialize_storages()
    print("LightRAG initialized!")
    
    # New description with more keywords
    new_description = """Điều 26 Luật Doanh nghiệp 2020 quy định về trình tự, thủ tục đăng ký doanh nghiệp.
Điều này quy định thủ tục đăng ký thành lập doanh nghiệp tư nhân, thủ tục đăng kí công ty TNHH 1 thành viên, 
thủ tục đăng ký công ty TNHH một thành viên, thủ tục đăng ký công ty TNHH hai thành viên trở lên, 
thủ tục đăng ký công ty hợp danh, thủ tục đăng ký công ty cổ phần.
Bao gồm quy định về hồ sơ đăng ký doanh nghiệp, nộp hồ sơ tại Cơ quan đăng ký kinh doanh, 
thời hạn cấp Giấy chứng nhận đăng ký doanh nghiệp trong 03 ngày làm việc.
Phương thức đăng ký: trực tiếp, qua bưu chính, hoặc qua mạng thông tin điện tử."""

    try:
        print("\n=== Editing entity Điều 26 - Luật Doanh nghiệp 2020 ===")
        updated_entity = await rag.aedit_entity(
            entity_name="Điều 26 - Luật Doanh nghiệp 2020",
            updated_data={
                "description": new_description
            }
        )
        print(f"✓ Entity updated successfully!")
        print(f"  New description: {updated_entity.get('description', '')[:200]}...")
        
    except Exception as e:
        print(f"✗ Error updating entity: {e}")
        import traceback
        traceback.print_exc()
    
    # Verify by checking the entity
    print("\n=== Verifying entity in graph ===")
    try:
        node = await rag.chunk_entity_relation_graph.get_node("Điều 26 - Luật Doanh nghiệp 2020")
        if node:
            print(f"Entity found!")
            print(f"Description: {node.get('description', '')[:300]}...")
        else:
            print("Entity not found!")
    except Exception as e:
        print(f"Error verifying: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(update_dieu_26())
