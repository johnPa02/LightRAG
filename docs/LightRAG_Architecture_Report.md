# LightRAG Architecture Report
## Hệ thống RAG cho Văn bản Pháp luật Việt Nam

**Ngày tạo:** 29/12/2025  
**Phiên bản:** 1.4.9.9  
**Tác giả:** LightRAG Development Team

---

## Mục lục

1. [Tổng quan Kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Quy trình Indexing (Nhập dữ liệu)](#2-quy-trình-indexing-nhập-dữ-liệu)
3. [Quy trình Query (Truy vấn)](#3-quy-trình-query-truy-vấn)
4. [Các Thành phần Lưu trữ](#4-các-thành-phần-lưu-trữ)
5. [Xử lý Đặc thù Văn bản Pháp luật](#5-xử-lý-đặc-thù-văn-bản-pháp-luật)
6. [Các Cải tiến Đã Thực hiện](#6-các-cải-tiến-đã-thực-hiện)
7. [Cấu hình và Tham số](#7-cấu-hình-và-tham-số)

---

## 1. Tổng quan Kiến trúc

### 1.1 Mô hình Hybrid RAG

LightRAG sử dụng kiến trúc **Hybrid RAG** kết hợp:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KEYWORD EXTRACTION (LLM)                      │
│  Query → {high_level_keywords, low_level_keywords}               │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  VECTOR SEARCH   │ │   GRAPH SEARCH   │ │   CHUNK SEARCH   │
│   (Entities)     │ │   (Relations)    │ │   (Direct)       │
│     Qdrant       │ │      Neo4j       │ │     Qdrant       │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                │               │               │
                └───────────────┼───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT AGGREGATION                           │
│  - Entity ranking & boosting                                     │
│  - Chunk merging & deduplication                                 │
│  - Cross-reference resolution                                    │
│  - Amendment injection                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM RESPONSE GENERATION                       │
│  System Prompt + Context + Query → Response                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Các Thành phần Chính

| Thành phần | Công nghệ | Chức năng |
|------------|-----------|-----------|
| **API Server** | FastAPI + Uvicorn | REST API endpoints |
| **Vector DB** | Qdrant | Lưu trữ embeddings cho entities và chunks |
| **Graph DB** | Neo4j | Lưu trữ knowledge graph (entities + relations) |
| **KV Store** | JSON Files | Lưu trữ metadata, cache |
| **LLM** | OpenAI/Compatible | Entity extraction, keyword extraction, response generation |
| **Embedding** | OpenAI/Compatible | Text → Vector conversion |

---

## 2. Quy trình Indexing (Nhập dữ liệu)

### 2.1 Pipeline Tổng quan

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Document  │ → │   Chunking  │ → │   Entity    │ → │   Storage   │
│   Input     │    │             │    │  Extraction │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.2 Bước 1: Document Chunking

**File:** `lightrag/operate.py` - function `chunking_by_token_size()`

```python
# Tham số mặc định
chunk_token_size = 1200      # Số tokens tối đa mỗi chunk
chunk_overlap_token_size = 100  # Overlap giữa các chunks
```

**Quy trình:**
1. Đọc document text
2. Tokenize bằng tiktoken
3. Chia thành chunks với overlap
4. Mỗi chunk được gán `chunk_id` duy nhất (hash của content)

### 2.3 Bước 2: Entity & Relation Extraction

**File:** `lightrag/operate.py` - function `extract_entities()`

**Prompt:** `lightrag/prompt.py` - `PROMPTS["entity_extraction"]`

LLM được gọi để extract từ mỗi chunk:

```json
{
  "entities": [
    {
      "entity_name": "Điều 9 - Nghị định 153/2020/NĐ-CP",
      "entity_type": "article",
      "description": "Điều 9 quy định về điều kiện chào bán trái phiếu",
      "source_id": "chunk-abc123..."
    }
  ],
  "relationships": [
    {
      "src_id": "Điều 9 - Nghị định 153/2020/NĐ-CP",
      "tgt_id": "Nghị định 153/2020/NĐ-CP",
      "description": "thuộc về",
      "keywords": "điều khoản, văn bản"
    }
  ]
}
```

**Entity Types cho Văn bản Pháp luật:**
- `LawDocument`: Văn bản pháp luật (Luật, Nghị định, Thông tư...)
- `Article`: Điều
- `Clause`: Khoản
- `Point`: Điểm
- `Other`: Khái niệm khác không thuộc các loại trên

**Relationship Types:**
- `IS_PART_OF_DOCUMENT`: Article → LawDocument (Điều thuộc Văn bản)
- `IS_PART_OF_ARTICLE`: Clause → Article (Khoản thuộc Điều)
- `IS_PART_OF_CLAUSE`: Point → Clause (Điểm thuộc Khoản)
- `REFERENCES`: Tham chiếu chéo đến Luật/Điều/Khoản/Điểm khác
- `GUIDED_BY`: Được hướng dẫn bởi Thông tư hoặc văn bản cấp dưới
- `AMENDS`: Sửa đổi văn bản khác
- `REPEALS`: Bãi bỏ/thay thế văn bản khác

### 2.4 Bước 3: Storage

**Entities được lưu vào:**
1. **Neo4j** (Graph): Node với properties (entity_id, description, source_id, file_path...)
2. **Qdrant** (Vector): Embedding của entity name + description

**Relations được lưu vào:**
1. **Neo4j** (Graph): Edge giữa 2 nodes
2. **Qdrant** (Vector): Embedding của content được tạo từ:
   ```
   [keywords]\t[src_entity_name]
   [tgt_entity_name]
   [description]
   ```
   Ví dụ: `"IS_PART_OF_ARTICLE\tKhoản 1 - Điều 24\nĐiều 24\nKhoản 1 thuộc Điều 24"`
   
   **Lưu ý:** Nhờ content chứa cả tên entities, relation có thể được tìm kiếm bằng tên entity, không chỉ bằng mô tả mối quan hệ.

**Chunks được lưu vào:**
1. **Qdrant** (Vector): Embedding của chunk content
2. **KV Store** (JSON): Full text content

---

## 3. Quy trình Query (Truy vấn)

### 3.1 Tổng quan Pipeline

Khi người dùng gửi câu hỏi, hệ thống thực hiện qua **6 giai đoạn** chính:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GĐ1: KEYWORD EXTRACTION                                                     │
│  Query → LLM phân tích → {high_level, low_level} keywords                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GĐ2: MULTI-SOURCE SEARCH                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │ Entity VDB  │  │ Relation VDB│  │  Chunk VDB  │                          │
│  │  (Qdrant)   │  │  (Qdrant)   │  │  (Qdrant)   │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
│         │                │                │                                   │
│         └────────────────┴────────────────┘                                   │
│                          │                                                    │
│                    Neo4j Graph                                                │
│               (lấy thông tin bổ sung)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GĐ3: RANKING & BOOSTING                                                     │
│  - Entities khớp query keywords: +5000 điểm                                  │
│  - Entities bổ sung (supplementary): +1000 điểm                             │
│  - Entities từ graph relations: theo degree                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GĐ4: TOKEN TRUNCATION                                                       │
│  - Cắt entities: max 6,000 tokens                                            │
│  - Cắt relations: max 6,000 tokens                                           │
│  - Cắt chunks: max ~34,000 tokens (tùy context)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GĐ5: CHUNK MERGING & ENRICHMENT                                             │
│  - Merge chunks từ nhiều nguồn (vector, entity, relation)                   │
│  - Resolve cross-references (Điều X của Luật Y)                             │
│  - Inject amendment content (nội dung sửa đổi/bổ sung)                      │
│  - Deduplicate                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GĐ6: RESPONSE GENERATION                                                    │
│  Context + Query → LLM → Response                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Giai đoạn 1: Keyword Extraction

**Mục đích:** Trích xuất từ khóa từ câu hỏi để phục vụ tìm kiếm

**Cách hoạt động:**
1. Query được gửi đến LLM với prompt yêu cầu phân tích
2. LLM trả về 2 loại keywords:
   - **High-level keywords:** Khái niệm trừu tượng, chủ đề tổng quát (ví dụ: "điều kiện thành lập doanh nghiệp")
   - **Low-level keywords:** Entities cụ thể, tên văn bản (ví dụ: "Điều 9 Nghị định 153/2020/NĐ-CP")

**Quy tắc đặc biệt cho văn bản pháp luật:**
- Nếu query là trích dẫn trực tiếp (direct legal citation), giữ nguyên thành 1 keyword
- Ví dụ: "Điều 9 Nghị định 153/2020/NĐ-CP" → **KHÔNG TÁCH** thành "Điều 9" + "Nghị định 153/2020/NĐ-CP"

**Ví dụ:**
| Query | High-level | Low-level |
|-------|------------|-----------|
| "Điều 9 Nghị định 153/2020/NĐ-CP" | [] | ["Điều 9 Nghị định 153/2020/NĐ-CP"] |
| "Điều kiện chào bán trái phiếu riêng lẻ" | ["điều kiện chào bán", "trái phiếu riêng lẻ"] | ["trái phiếu doanh nghiệp"] |

---

### 3.3 Giai đoạn 2: Multi-Source Search

**Mục đích:** Tìm kiếm thông tin liên quan từ nhiều nguồn dữ liệu

**3 chế độ tìm kiếm (Query Mode):**

| Mode | Nguồn dữ liệu | Keywords sử dụng | Use case |
|------|---------------|------------------|----------|
| **local** | Entity VDB + Neo4j | low_level | Truy vấn cụ thể về 1 điều khoản |
| **global** | Relation VDB + Neo4j | high_level | Truy vấn về mối quan hệ giữa các khái niệm |
| **mix** | Tất cả + Chunk VDB | Cả hai | Mặc định, tổng hợp nhiều nguồn |

**Quy trình search trong mode "mix":**

```
1. ENTITY SEARCH (low_level keywords)
   ├── Vector search trong Qdrant (entities_vdb)
   ├── Lấy top-k entities gần nhất (cosine similarity > 0.2)
   └── Enrich từ Neo4j: lấy description, relations, metadata

2. RELATION SEARCH (high_level keywords)
   ├── Vector search trong Qdrant (relationships_vdb)
   └── Lấy các relations liên quan đến keywords

3. CHUNK SEARCH (full query)
   ├── Vector search trong Qdrant (chunks_vdb)
   └── Lấy text chunks có nội dung gần với query
```

**Tham số quan trọng:**
- `top_k`: 40 (số entities/relations tối đa từ vector search)
- `chunk_top_k`: 80 (số chunks tối đa từ vector search)
- `cosine_better_than_threshold`: 0.2 (ngưỡng similarity tối thiểu)

---

### 3.4 Giai đoạn 3: Ranking & Boosting

**Mục đích:** Sắp xếp entities theo độ quan trọng để đảm bảo entities liên quan nhất không bị cắt

**Vấn đề cần giải quyết:**
- Sau search, có thể có hàng nghìn entities
- Token limit chỉ cho phép giữ lại ~16 entities
- Cần đảm bảo entity người dùng hỏi về PHẢI được giữ lại

**Cơ chế Boosting:**

| Loại entity | Điểm cộng | Giải thích |
|-------------|-----------|------------|
| **Query-matched** | +5000 | Entity có tên khớp với query keyword |
| **Supplementary** | +1000 | Entity được lấy từ graph relations |
| **Regular** | +0 | Entity chỉ từ vector search |

**Quy trình:**
1. **Chuẩn hóa tên entity** để so sánh (bỏ dấu "-", chuẩn hóa spaces)
2. So sánh với query keywords đã chuẩn hóa
3. Nếu khớp, cộng thêm 5000 điểm vào rank
4. Sắp xếp theo thứ tự: Query-matched > Supplementary > Regular > By degree

**Ví dụ:**
```
Query: "Điều 9 Nghị định 153/2020/NĐ-CP"
Keyword (normalized): "điều 9 nghị định 153/2020/nđ-cp"

Entity tìm được: "Điều 9 - Nghị định 153/2020/NĐ-CP"
Entity (normalized): "điều 9 nghị định 153/2020/nđ-cp"

→ MATCH! → Boost +5000 → Entity này sẽ được giữ lại
```

---

### 3.5 Giai đoạn 4: Token Truncation

**Mục đích:** Cắt giảm dữ liệu để fit vào context window của LLM

**Giới hạn token:**
| Component | Default Limit | Có thể điều chỉnh |
|-----------|--------------|-------------------|
| Entities | 6,000 tokens | max_entity_tokens |
| Relations | 6,000 tokens | max_relation_tokens |
| Chunks | ~34,000 tokens | max_total_tokens - entities - relations |

**Quy trình:**
1. Entities/Relations đã được sắp xếp theo rank (GĐ3)
2. Duyệt từ đầu danh sách, tính token count
3. Khi vượt quá limit, cắt bỏ phần còn lại
4. Entities có rank cao (đã boost) nằm ở đầu → được giữ lại

**Lưu ý:**
- Với 6,000 tokens, thường giữ được 15-20 entities
- Nếu có quá nhiều entities, những entity ít liên quan sẽ bị loại
- Query-matched entities luôn được ưu tiên giữ lại nhờ boost

---

### 3.6 Giai đoạn 5: Chunk Merging & Enrichment

**Mục đích:** Tổng hợp và làm giàu chunks từ nhiều nguồn, đảm bảo context đầy đủ

#### 5.1 Nguồn chunks

| Nguồn | Mô tả | Ưu tiên |
|-------|-------|---------|
| Vector chunks | Từ search trực tiếp | Cao |
| Entity chunks | Chunks chứa entities đã match | Trung bình |
| Relation chunks | Chunks chứa relations đã match | Trung bình |
| Cross-reference chunks | Chunks được tham chiếu từ annotations | Cao |
| Amendment chunks | Chunks chứa nội dung sửa đổi | Cao nhất |

#### 5.2 Cross-Reference Resolution

Khi chunk chứa tham chiếu đến Điều/Khoản khác, hệ thống tự động tìm và inject nội dung:

**Các pattern được detect:**

| Pattern | Ví dụ | Hành động |
|---------|-------|-----------|
| Guidance reference | "được hướng dẫn bởi Điều 49 Nghị định 168/2025/NĐ-CP" | Tìm chunk chứa Điều 49 NĐ 168 |
| Amendment annotation | "[Điều này được sửa đổi bởi Khoản 10 Điều 1 Luật DN sửa đổi 2025]" | Tìm chunk chứa Khoản 10 |
| Internal reference | "quy định tại khoản 2 Điều 115 của Luật này" | Tìm chunk Điều 115 trong cùng file |

**Quy trình resolve:**
1. Parse content để detect cross-reference patterns
2. Xác định Điều/Khoản được tham chiếu
3. Tìm chunk phù hợp trong text_chunks_db
4. Inject chunk đó vào context (với flag is_cross_reference=True)

#### 5.3 Amendment Injection

Khi người dùng hỏi về một Điều đã được sửa đổi:

```
Chunk gốc: "Điều 23. Góp vốn vào công ty
          [Điều này được bổ sung bởi Khoản 10 Điều 1 Luật DN sửa đổi 2025]
          1. Góp vốn là việc..."

Hệ thống tự động:
1. Detect annotation "[...được bổ sung bởi Khoản 10...]"
2. Tìm chunk chứa "Khoản 10 Điều 1 Luật DN sửa đổi 2025"
3. Inject chunk đó ngay sau chunk Điều 23
4. Mark với is_amendment_content=True để ưu tiên
```

#### 5.4 Round-Robin Merging

Chunks từ các nguồn được merge xen kẽ để đảm bảo diversity:

```
Final chunks = [
    vector_chunk_1,    ← từ vector search
    entity_chunk_1,    ← từ entity match
    relation_chunk_1,  ← từ relation match
    vector_chunk_2,
    entity_chunk_2,
    ...
]
```

#### 5.5 Deduplication

- Chunks trùng lặp (cùng chunk_id) được loại bỏ
- Giữ chunk có priority cao hơn (amendment > cross-ref > vector > entity)

---

### 3.7 Giai đoạn 6: Response Generation

**Mục đích:** Sinh câu trả lời từ context đã xây dựng

#### 6.1 Context Assembly

Toàn bộ context được format thành 2 phần:

**Phần 1: Knowledge Graph Data**
```
### Entities
- **Điều 9 - Nghị định 153/2020/NĐ-CP** (article): Điều 9 quy định về điều kiện chào bán trái phiếu

### Relationships
- Điều 9 → thuộc về → Nghị định 153/2020/NĐ-CP
```

**Phần 2: Document Chunks**
```
[1] **Nghị_định_153-2020-NĐ-CP.txt**
Điều 9. Điều kiện chào bán trái phiếu
1. Đối với trái phiếu không chuyển đổi...

[2] **Nghị_định_153-2020-NĐ-CP.txt**
a) Là công ty cổ phần hoặc công ty trách nhiệm hữu hạn...
```

#### 6.2 LLM Call

**System prompt** yêu cầu LLM:
- Trả lời dựa trên context được cung cấp
- Trích dẫn đầy đủ, chính xác nội dung pháp luật
- Ghi rõ Điều, Khoản, Điểm khi trích dẫn
- Ghi chú nguồn bằng [1], [2]...
- Nếu không đủ thông tin trong context, nói rõ ràng

#### 6.3 Caching

- Response được cache theo hash của query + parameters
- Nếu query giống nhau, trả về cache (tiết kiệm API cost)
- Cache có thể được bật/tắt qua config

---

## 4. Các Thành phần Lưu trữ

### 4.1 Qdrant Collections

| Collection | Content được Embedding | Searchable by |
|------------|------------------------|---------------|
| `lightrag_vdb_entities` | `entity_name + "\n" + description` | Tên entity, mô tả |
| `lightrag_vdb_relationships` | `keywords + "\t" + src_id + "\n" + tgt_id + "\n" + description` | Tên entities, loại quan hệ, mô tả |
| `lightrag_vdb_chunks` | Chunk content (full text) | Nội dung văn bản |

**Entity Payload:**
```json
{
  "entity_name": "Điều 9 - Nghị định 153/2020/NĐ-CP",
  "content": "Điều 9 - Nghị định 153/2020/NĐ-CP\nĐiều 9 quy định về điều kiện chào bán trái phiếu",
  "source_id": "chunk-abc123...",
  "description": "Điều 9 quy định về điều kiện chào bán trái phiếu",
  "entity_type": "Article",
  "file_path": "Nghị_định_153-2020-NĐ-CP.txt",
  "created_at": 1766734340,
  "workspace_id": "default"
}
```

**Relationship Payload:**
```json
{
  "src_id": "Khoản 2 - Điều 9 - Luật NSNN",
  "tgt_id": "Khoản 3 - Điều 40 - Luật NSNN",
  "content": "REFERENCES\tKhoản 2 - Điều 9 - Luật NSNN\nKhoản 3 - Điều 40 - Luật NSNN\nKhoản 2 Điều 9 viện dẫn quy định tại khoản 3 Điều 40",
  "keywords": "REFERENCES",
  "description": "Khoản 2 Điều 9 viện dẫn quy định tại khoản 3 Điều 40",
  "source_id": "chunk-abc123...",
  "file_path": "Luật_NSNN.txt"
}
```

**Chunk Payload:**
```json
{
  "content": "Điều 9. Điều kiện chào bán trái phiếu\n1. Đối với trái phiếu không chuyển đổi...",
  "chunk_id": "chunk-abc123...",
  "file_path": "Nghị_định_153-2020-NĐ-CP.txt",
  "created_at": 1766734340,
  "workspace_id": "default"
}
```

### 4.2 Neo4j Schema

**Nodes:**
```cypher
(:default:article {
  entity_id: "Điều 9 - Nghị định 153/2020/NĐ-CP",
  description: "Điều 9 quy định về điều kiện chào bán trái phiếu",
  source_id: "chunk-abc123<SEP>chunk-def456",
  file_path: "Nghị_định_153-2020-NĐ-CP.txt",
  entity_type: "article",
  created_at: 1766734340
})
```

**Relationships:**
```cypher
(:article)-[:RELATES_TO {
  description: "thuộc về",
  keywords: "điều khoản, văn bản",
  weight: 1.0,
  source_id: "chunk-abc123"
}]->(:document)
```

### 4.3 KV Store (JSON Files)

| File | Content |
|------|---------|
| `kv_store_full_docs.json` | Original document texts |
| `kv_store_text_chunks.json` | Chunk content + metadata |
| `kv_store_entity_chunks.json` | Entity → chunk_ids mapping |
| `kv_store_full_entities.json` | Entity full data |
| `kv_store_full_relations.json` | Relation full data |
| `kv_store_llm_response_cache.json` | LLM response cache |

---

## 5. Xử lý Đặc thù Văn bản Pháp luật

### 5.1 Entity Naming Convention

```python
# Format chuẩn cho legal entities
"Điều 9 - Nghị định 153/2020/NĐ-CP"
"Khoản 1 - Điều 9 - Nghị định 153/2020/NĐ-CP"
"Điểm a - Khoản 1 - Điều 9 - Nghị định 153/2020/NĐ-CP"
```

### 5.2 Cross-Reference Patterns

**File:** `lightrag/operate.py` - function `_resolve_cross_references()`

| Pattern | Regex | Example |
|---------|-------|---------|
| Guidance | `được hướng dẫn bởi (Điều \d+)` | "được hướng dẫn bởi Điều 49 NĐ 168" |
| Amendment | `được (sửa đổi\|bổ sung) bởi (Khoản \d+)` | "[Khoản này được bổ sung bởi Khoản 10...]" |
| Internal | `(khoản \d+) (Điều \d+) của Luật này` | "quy định tại khoản 2 Điều 115 của Luật này" |
| Direct | `theo (Điều \d+) (Nghị định\|Luật)` | "theo Điều 7 Nghị định 01/2021" |

### 5.3 Amendment Tracking

```python
# Annotations được inject vào chunks khi indexing
"[Điều này được sửa đổi bởi Khoản X Điều Y Văn bản Z có hiệu lực từ ngày DD/MM/YYYY]"

# Khi query, system:
# 1. Detect annotation trong chunk
# 2. Tìm chunk chứa nội dung sửa đổi
# 3. Inject ngay sau chunk gốc
```

### 5.4 Supplementary Entity Boosting

```python
# Entities được link qua relations "bổ sung", "sửa đổi" được boost
if relation_desc contains ["bổ sung", "sửa đổi", "thay thế"]:
    linked_entity["is_supplementary"] = True
    linked_entity["rank"] += 1000  # Boost priority
```

---

## 6. Các Cải tiến Đã Thực hiện

### 6.1 Keyword Extraction Improvements

**Vấn đề:** LLM tách "Điều 9 Nghị định 153/2020/NĐ-CP" thành ["Điều 9", "Nghị định 153/2020/NĐ-CP"] gây nhiễu.

**Giải pháp:** Cập nhật prompt với rule:
```
**CRITICAL RULE for Legal Citations:**
- When query IS or CONTAINS a specific legal citation like "Điều X Nghị định/Luật Y", 
  keep the FULL citation as ONE keyword.
- DO NOT split into separate parts.
```

### 6.2 Query Match Boosting

**Vấn đề:** Entities trực tiếp liên quan đến query bị đẩy xuống sau supplementary entities.

**Giải pháp:** Thêm `_query_match_boost` flag với priority cao nhất:
```python
# Entities khớp với query keywords được boost +5000
# Sort order: query_match > supplementary > rank
```

### 6.3 Normalized Matching

**Vấn đề:** Query "Điều 9 Nghị định 153" không match entity "Điều 9 - Nghị định 153" do dấu gạch ngang.

**Giải pháp:** Normalize trước khi compare:
```python
def normalize_for_match(s):
    s = re.sub(r'\s*[-–—]\s*', ' ', s.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return s
```

### 6.4 Internal Reference Resolution

**Vấn đề:** "khoản 2 Điều 115 của Luật này" không được resolve đúng.

**Giải pháp:** Thêm Pattern 2b với `source_file_path` filtering:
```python
# Pattern 2b: Internal refs "Điều X của Luật này"
if "của luật này" in ref_text.lower():
    # Filter chunks from same source file only
    chunk_file = chunk.get("file_path", "")
    if source_file_path and source_file_path in chunk_file:
        # Match internal reference
```

---

## 7. Cấu hình và Tham số

### 7.1 Environment Variables

```bash
# LLM Configuration
LLM_MODEL=gpt-4o-mini
LLM_BINDING=openai
OPENAI_API_KEY=sk-xxx

# Embedding Configuration  
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BINDING=openai
EMBEDDING_DIM=1536

# Storage Configuration
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_VECTOR_STORAGE=QdrantVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage

# Qdrant
QDRANT_URL=http://localhost:6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=lightrag123
```

### 7.2 Query Parameters

```python
@dataclass
class QueryParam:
    mode: str = "mix"          # local|global|hybrid|mix
    top_k: int = 40            # Number of results per search
    max_tokens: int = 50000    # Total context tokens
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 6000
    hop_depth: int = 2         # Graph traversal depth
    enable_rerank: bool = False
    response_type: str = "Multiple Paragraphs"
```

### 7.3 Indexing Parameters

```python
# Chunking
chunk_token_size = 1200
chunk_overlap_token_size = 100

# Entity Extraction
entity_extract_max_gleaning = 1
max_async = 16  # Parallel LLM calls
```

### 7.4 Vector Search Parameters

```python
# Qdrant
cosine_better_than_threshold = 0.2  # Minimum similarity score
top_k_entities = 40
top_k_chunks = 80
```

---

## Appendix A: File Structure

```
lightrag/
├── __init__.py
├── lightrag.py          # Main LightRAG class
├── operate.py           # Core operations (indexing, querying)
├── prompt.py            # All LLM prompts
├── base.py              # Base classes and types
├── constants.py         # Default constants
├── utils.py             # Utility functions
├── kg/
│   ├── neo4j_impl.py    # Neo4j storage implementation
│   ├── qdrant_impl.py   # Qdrant storage implementation
│   └── ...
├── llm/
│   ├── openai.py        # OpenAI LLM binding
│   └── ...
└── api/
    ├── lightrag_server.py  # FastAPI server
    └── routers/
```

## Appendix B: API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Execute RAG query |
| `/documents/text` | POST | Insert text document |
| `/documents/file` | POST | Upload file |
| `/health` | GET | Health check |
| `/documents` | GET | List documents |

---

*Report generated for LightRAG Vietnamese Legal Document System*
