#!/usr/bin/env python3
"""
Script để tạo relation giữa các Điều (hồ sơ đăng ký) và Mẫu số (biểu mẫu) tương ứng
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag import LightRAG

async def create_form_relations():
    """Tạo relations giữa Điều về hồ sơ đăng ký và Mẫu biểu mẫu tương ứng"""
    
    # Initialize LightRAG with same config as server
    rag = LightRAG(
        working_dir=os.environ.get("RAG_WORKING_DIR", "./data/rag_storage"),
    )
    await rag.initialize_storages()
    
    # Define relations to create
    # Format: (source_entity, target_entity, description, keywords)
    relations = [
        # Điều 21 Luật DN 2020 - Hồ sơ đăng ký công ty TNHH một thành viên -> Mẫu số 2
        (
            "Điều 21 - Luật Doanh nghiệp 2020",
            "Mẫu số 2 - Thông tư 68/2025/TT-BTC",
            "Điều 21 Luật Doanh nghiệp 2020 quy định hồ sơ đăng ký công ty TNHH một thành viên sử dụng Mẫu số 2 Thông tư 68/2025/TT-BTC",
            "hồ sơ đăng ký, công ty TNHH một thành viên, biểu mẫu, giấy đề nghị đăng ký doanh nghiệp"
        ),
        # Điều 19 Luật DN 2020 - Hồ sơ đăng ký doanh nghiệp tư nhân -> Mẫu số 1
        (
            "Điều 19 - Luật Doanh nghiệp 2020",
            "Mẫu số 1 - Phụ lục I - Thông tư 68/2025/TT-BTC",
            "Điều 19 Luật Doanh nghiệp 2020 quy định hồ sơ đăng ký doanh nghiệp tư nhân sử dụng Mẫu số 1 Thông tư 68/2025/TT-BTC",
            "hồ sơ đăng ký, doanh nghiệp tư nhân, biểu mẫu, giấy đề nghị đăng ký doanh nghiệp"
        ),
        # Điều 20 Luật DN 2020 - Hồ sơ đăng ký công ty hợp danh -> Mẫu số 3
        (
            "Điều 20 - Luật Doanh nghiệp 2020",
            "Mẫu số 3 - Phụ lục I - Thông tư 68/2025/TT-BTC",
            "Điều 20 Luật Doanh nghiệp 2020 quy định hồ sơ đăng ký công ty hợp danh sử dụng Mẫu số 3 Thông tư 68/2025/TT-BTC",
            "hồ sơ đăng ký, công ty hợp danh, biểu mẫu, giấy đề nghị đăng ký doanh nghiệp"
        ),
        # Điều 22 Luật DN 2020 - Hồ sơ đăng ký công ty TNHH 2+ thành viên -> Mẫu số 4
        (
            "Điều 22 - Luật Doanh nghiệp 2020",
            "Mẫu số 4 - Phụ lục I - Thông tư 68/2025/TT-BTC",
            "Điều 22 Luật Doanh nghiệp 2020 quy định hồ sơ đăng ký công ty TNHH hai thành viên trở lên sử dụng Mẫu số 4 Thông tư 68/2025/TT-BTC",
            "hồ sơ đăng ký, công ty TNHH hai thành viên trở lên, biểu mẫu, giấy đề nghị đăng ký doanh nghiệp"
        ),
        # Điều 23 Luật DN 2020 - Hồ sơ đăng ký công ty cổ phần -> Mẫu số 5
        (
            "Điều 23 - Luật Doanh nghiệp 2020",
            "Mẫu số 5 - Phụ lục I - Thông tư 68/2025/TT-BTC",
            "Điều 23 Luật Doanh nghiệp 2020 quy định hồ sơ đăng ký công ty cổ phần sử dụng Mẫu số 5 Thông tư 68/2025/TT-BTC",
            "hồ sơ đăng ký, công ty cổ phần, biểu mẫu, giấy đề nghị đăng ký doanh nghiệp"
        ),
    ]
    
    print(f"Creating {len(relations)} form relations...")
    
    for src, tgt, desc, keywords in relations:
        try:
            result = rag.create_relation(src, tgt, {
                "description": desc,
                "keywords": keywords,
                "weight": 3.0  # High weight for form relations
            })
            print(f"✓ Created relation: {src} -> {tgt}")
        except Exception as e:
            print(f"✗ Failed to create relation {src} -> {tgt}: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(create_form_relations())
