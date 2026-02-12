# VietDoc-Analyzer System Flow Report
## Báo cáo Tổng quan Hoạt động Hệ thống

**Ngày tạo:** 03/02/2026  

---

## 1. Tổng quan Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Frontend)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND (VietDoc-Analyzer)                           │
│                          POST /api/conversations/ask                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Xác thực người dùng                                                      │
│  • Quản lý session memory                                                   │
│  • Kết hợp long-term memory                                                 │
│  • Gọi LightRAG API                                                         │
│  • Streaming Response (SSE)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                         │
         ▼                    ▼                         ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────────────────────────┐
│  SimpleMem    │    │   PostgreSQL   │    │         LightRAG Server          │
│  (Long-term   │    │   (Session     │    │ POST /api/healthcare/query/stream│
│   Memory)     │    │    Memory)     │    │                                  │
└───────────────┘    └────────────────┘    └──────────────────────────────────┘
                                                          │
                          ┌───────────────────────────────┼───────────────────┐
                          ▼                               ▼                   ▼
                   ┌────────────┐                ┌────────────┐      ┌─────────────┐
                   │   Neo4j    │                │   Qdrant   │      │ External AI │
                   │  (Graph)   │                │  (Vector)  │      │   Services  │
                   └────────────┘                └────────────┘      └─────────────┘
```

---

## 2. Luồng xử lý Chat chính

### Khi người dùng gửi câu hỏi:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 1: NHẬN REQUEST TỪ FRONTEND                                            │
├────────────────────────────────────────────────────────────────────────────┤
│ POST /api/conversations/ask                                                 │
│ {                                                                           │
│   "conversationId": 123,        ← ID cuộc trò chuyện                        │
│   "question": "Điều 9 NĐ 153?", ← Câu hỏi của người dùng                    │
│   "clinicalContext": "..."      ← Ngữ cảnh lâm sàng (tùy chọn)              │
│ }                                                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2: LẤY LONG-TERM MEMORY (Ký ức dài hạn về người dùng)                  │
├────────────────────────────────────────────────────────────────────────────┤
│ → Gọi SimpleMem Server: POST /users/{userId}/context                        │
│ → Lấy các thông tin đã lưu về người dùng từ các phiên trước                │
│ → Ví dụ: hoàn cảnh cá nhân, câu hỏi thường gặp, preferences                │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3: KẾT HỢP MEMORY + GỌI LIGHTRAG                                       │
├────────────────────────────────────────────────────────────────────────────┤
│ → Kết hợp: Long-term memory + Session memory + Câu hỏi                      │
│ → Gọi LightRAG: POST /api/healthcare/query/stream                           │
│ → Streaming response về cho Frontend                                        │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 4: LƯU KẾT QUẢ + CẬP NHẬT MEMORY (Chạy song song, không chờ đợi)       │
├────────────────────────────────────────────────────────────────────────────┤
│ [4A] Lưu câu hỏi + trả lời vào PostgreSQL                                  │
│ [4B] Cập nhật Session Memory (tóm tắt progressively)                        │
│ [4C] Lưu vào Long-term Memory (SimpleMem)                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Luồng xử lý trong LightRAG

### 3.1 Luồng xử lý truy vấn

```
┌────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: PHÂN TÍCH KEYWORDS                                             │
├────────────────────────────────────────────────────────────────────────────┤
│ Query: "Điều 9 Nghị định 153/2020/NĐ-CP" → LLM Phân tích                    │
│ → High-level: []  (Không có khái niệm trừu tượng)                           │
│ → Low-level: ["Điều 9 Nghị định 153/2020/NĐ-CP"]                            │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: TÌM KIẾM SONG SONG (3 NGUỒN)                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Qdrant]              [Neo4j]                [Qdrant]                     │
│  Entity Search         Graph Expansion        Chunk Search                  │
│  → Tìm Điều 9          → Tìm các điều         → Tìm văn bản                │
│                           liên quan              chứa nội dung              │
│                                                                             │
│  ⚡ Plus: Appendix API và Perplexity Search (nếu bật)                       │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: XỬ LÝ ĐẶC THÙ PHÁP LUẬT                                        │
├────────────────────────────────────────────────────────────────────────────┤
│ ✅ Multi-hop Expansion: Tìm các điều sửa đổi/bổ sung qua Graph              │
│ ✅ Amendment Injection: Tự động thêm nội dung sửa đổi                       │
│ ✅ Cross-reference Resolution: Giải quyết tham chiếu chéo                   │
│ ✅ Priority Ranking: Ưu tiên kết quả khớp với câu hỏi                       │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4: TẠO CÂU TRẢ LỜI                                                │
├────────────────────────────────────────────────────────────────────────────┤
│ Context (Entities + Relations + Chunks) → LLM → Response (Streaming)       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tính năng Đặc biệt

### 4.1 🔍 Perplexity Web Search (Bật/Tắt)

**Mục đích:** Bổ sung thông tin từ Internet cho câu hỏi cần dữ liệu mới/cập nhật.

| Trạng thái | Mô tả |
|------------|-------|
| **TẮT (`use_perplexity: false`)** | Chỉ dùng dữ liệu nội bộ (văn bản pháp luật đã index) |
| **BẬT (`use_perplexity: true`)** | Tra cứu thêm từ Internet qua Perplexity AI |

**Trong request:**
```json
{
  "query": "Điều 9 NĐ 153 có gì mới năm 2026?",
  "use_perplexity": true   ← Bật tìm kiếm web
}
```

**Luồng hoạt động khi BẬT:**
```
Query ─────┬──────────────────────────────────────────────────→ Graph + Vector Search
           │                                                            │
           └──→ Perplexity API ───→ Kết quả Web ───────────────────────┼
                                                                        │
                                                        ┌───────────────┘
                                                        ▼
                        [Kết hợp RAG Context + Web Context] → LLM → Response
```

> **Lưu ý:** Kết quả từ Perplexity chỉ để tham khảo. Ưu tiên nguồn chính thống từ văn bản pháp luật.

---

### 4.2 📊 Appendix API (Tra cứu Phụ lục)

**Mục đích:** Tra cứu bảng mã, danh mục ICD, phụ lục y tế được lưu riêng.

**Điều kiện kích hoạt:**
- Query có liên quan đến phụ lục (ví dụ: hỏi về mã bệnh, STT dịch vụ...)
- Intent API xác định độ tin cậy > 50%

**Cấu hình:**
```env
APPENDIX_INTENT_API_URL=http://host:port/intent     # API phân loại intent
APPENDIX_VECTOR_API_URL=http://host:port/search     # API tìm kiếm phụ lục
APPENDIX_CONFIDENCE_THRESHOLD=0.5                    # Ngưỡng độ tin cậy
```

**Luồng hoạt động:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 1: KIỂM TRA INTENT                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Query: "Mã ICD cho bệnh tiểu đường là gì?" → Intent API                     │
│ → matched: true, confidence: 0.85, appendices: ["Phụ lục mã ICD-10"]       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (nếu confidence > 0.5)
┌─────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2: TÌM KIẾM VECTOR TRONG PHỤ LỤC                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ → Gọi Vector API với document_ids từ bước 1                                  │
│ → Trả về chunks chứa thông tin phụ lục liên quan                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3: INJECT VÀO CONTEXT                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Appendix content được thêm vào đầu context với nhãn đặc biệt:               │
│ "---Appendix Content (Supplementary Information)---"                         │
│                                                                              │
│ LLM được hướng dẫn: ƯU TIÊN dữ liệu từ Appendix cho câu hỏi tra cứu        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 🧠 Memory System (Hệ thống Ghi nhớ)

**Hai loại memory:**

| Loại | Lưu trữ | Phạm vi | Mục đích |
|------|---------|---------|----------|
| **Session Memory** | PostgreSQL | Per conversation | Tóm tắt cuộc hội thoại hiện tại |
| **Long-term Memory** | LanceDB (SimpleMem) | Per user | Ghi nhớ thông tin về user qua nhiều phiên |

**Session Memory - Progressive Summarization:**
```
Turn 1: User hỏi về quy định BHYT → Tóm tắt: "User quan tâm BHYT"
Turn 2: User hỏi chi tiết Điều 5 → Tóm tắt: "User quan tâm BHYT, đặc biệt Điều 5"
Turn 3: User hỏi về case cụ thể → Tóm tắt cập nhật...
```

**Long-term Memory - Atomic Facts:**
```json
{
  "memories": [
    {
      "entry_id": "b0454743-35e0-4fc9-b1c8-a06846b08d07",
      "lossless_restatement": "Người dùng đang bị u não.",
      "keywords": [
        "u não",
        "bệnh"
      ],
      "timestamp": "2026-02-02T14:04:31",
      "location": null,
      "persons": [
        "người dùng"
      ],
      "entities": [
        "u não"
      ],
      "topic": "Tình trạng sức khỏe"
    },
    {
      "entry_id": "9a5a1827-19ed-4b6e-abd5-16ee9b870671",
      "lossless_restatement": "Chồng của người dùng là liệt sĩ.",
      "keywords": [
        "chồng",
        "liệt sĩ"
      ],
      "timestamp": "2026-02-02T14:08:31",
      "location": null,
      "persons": [
        "chồng người dùng"
      ],
      "entities": [
        "liệt sĩ"
      ],
      "topic": "Thân nhân liệt sĩ"
    }
  ],
  "total": 2
}
```

---

## 5. Sequence Diagram (Luồng hoàn chỉnh)

```
┌──────┐    ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
│Client│    │ Backend │   │SimpleMem │   │ LightRAG │   │Perplexity│   │Appendix │
└──┬───┘    └────┬────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬────┘
   │             │              │               │               │              │
   │ POST /ask   │              │               │               │              │
   │────────────>│              │               │               │              │
   │             │              │               │               │              │
   │             │ Get context  │               │               │              │
   │             │─────────────>│               │               │              │
   │             │  memories    │               │               │              │
   │             │<─────────────│               │               │              │
   │             │              │               │               │              │
   │ SSE:connect │              │               │               │              │
   │<────────────│              │               │               │              │
   │             │              │               │               │              │
   │             │ POST /query/stream ────────────────────────────────────────>│
   │             │              │               │               │              │
   │             │              │    ┌──────────┴──────────┐    │              │
   │             │              │    │    SONG SONG        │    │              │
   │             │              │    │ • Graph search      │    │              │
   │             │              │    │ • Vector search     │<───┘              │
   │             │              │    │ • Perplexity (nếu)  │<──────────────────┘
   │             │              │    │ • Appendix (nếu)    │   Appendix result
   │             │              │    └──────────┬──────────┘
   │             │              │               │
   │             │ stream chunk │               │
   │             │<─────────────────────────────│
   │ SSE:content │              │               │
   │<────────────│              │               │
   │             │              │               │
   │             │ (tiếp tục stream...)         │
   │             │              │               │
   │ SSE: done   │              │               │
   │<────────────│              │               │
   │             │              │               │
   │             │ Store memory (async)         │
   │             │─────────────>│               │
```

---

## 6. Công nghệ sử dụng

| Component | Technology | Mô tả |
|-----------|------------|-------|
| **Backend** | Node.js + Express | VietDoc-Analyzer Backend |
| **LightRAG** | Python + FastAPI | RAG Engine + API Server |
| **Vector DB** | Qdrant | Lưu embeddings (entities, chunks) |
| **Graph DB** | Neo4j | Lưu knowledge graph |
| **Memory (Short-term)** | PostgreSQL | Session memory |
| **Memory (Long-term)** | LanceDB | User facts (SimpleMem) |
| **Embedding** | Voyage AI | Vietnamese-optimized embeddings |
| **LLM** | OpenAI GPT-4.1 | Response generation |
| **Web Search** | Perplexity Sonar | Bổ sung từ Internet |

---

## 7. Performance Metrics

| Thao tác | Thời gian | Ghi chú |
|----------|-----------|---------|
| Lấy Long-term Memory | ~2s | Voyage AI embedding + LanceDB search |
| LightRAG Query (Streaming) | 5-15s | Tùy độ phức tạp |
| Perplexity Search | ~2s | Khi bật |
| Appendix Search | ~1s | Khi match intent |
| Session Memory Update | ~1s | Async, không block |
