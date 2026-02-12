# VietDoc-Analyzer Backend Chat Flow Report

## Tổng quan

Hệ thống chat của VietDoc-Analyzer sử dụng kiến trúc **multi-layer memory** kết hợp với **RAG (Retrieval-Augmented Generation)** để trả lời câu hỏi về văn bản pháp lý y tế.

## Kiến trúc Memory

```
┌──────────────────────────────────────────────────────────────────┐
│                         MEMORY LAYERS                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐   ┌─────────────────────────────────┐   │
│  │ Session Memory      │   │ Long-term Memory (SimpleMem)    │   │
│  │ (PostgreSQL)        │   │ (LanceDB + Voyage AI)           │   │
│  ├─────────────────────┤   ├─────────────────────────────────┤   │
│  │ • Per conversation  │   │ • Per user (cross-session)      │   │
│  │ • Progressive       │   │                             │   │
│  │   summary           │   │ • Atomic facts extraction       │   │
│  │ • Last 3 turns      │   │ • Vietnamese healthcare focus   │   │
│  └─────────────────────┘   └─────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Luồng xử lý Chat (`askQuestion`)

### Phase 1: Initialization & Context Gathering

```
┌─────────────────────────────────────────────────────────────────┐
│ [1] REQUEST RECEIVED                                             │
│     POST /api/conversations/ask                                  │
│     Body: { conversationId?, question, clinicalContext? }        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [2] USER VALIDATION                                              │
│     • Lấy user từ session                                        │
│     • Kiểm tra quyền truy cập                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [3] CONVERSATION HANDLING                                        │
│     IF conversationId không có:                                  │
│       → Tạo conversation mới                                     │
│     ELSE:                                                        │
│       → Load session memory từ conversation hiện tại             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [4] CREATE QUERY RECORD                                          │
│     • Lưu câu hỏi vào PostgreSQL                                 │
│     • Ghi nhận clinical context (nếu có)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [5] BUILD CONVERSATION HISTORY                                   │
│     • Lấy 3 turns gần nhất từ conversation                       │
│     • Format: [{ role: "user"|"assistant", content }]            │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Memory Retrieval (Parallel)

```
┌─────────────────────────────────────────────────────────────────┐
│ [6] LONG-TERM MEMORY RETRIEVAL                                   │
│                                                                  │
│     ┌───────────────────────────────────────────────────────┐   │
│     │ SimpleMem Server (http://171.244.137.76:8100)         │   │
│     │                                                        │   │
│     │ POST /users/{userId}/context                          │   │
│     │ Body: { question: "..." }                             │   │
│     │                                                        │   │
│     │ Response:                                              │   │
│     │ {                                                      │   │
│     │   context: "=== KÝ ỨC DÀI HẠN VỀ NGƯỜI DÙNG ===       │   │
│     │            [Tóm tắt liên quan]                         │   │
│     │            ...                                         │   │
│     │            [Chi tiết ký ức đã lưu]                     │   │
│     │            - Ký ức 1: ...                              │   │
│     │            - Ký ức 2: ...",                            │   │
│     │   memory_count: 3                                      │   │
│     │ }                                                      │   │
│     └───────────────────────────────────────────────────────┘   │
│                                                                  │
│     Latency: ~2 seconds (Voyage AI embedding + LanceDB search)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [7] COMBINE MEMORIES                                             │
│                                                                  │
│     combinedMemory = longTermMemoryContext                       │
│                    + "\n\n[TÓM TẮT PHIÊN HỘI THOẠI HIỆN TẠI]\n"  │
│                    + currentSessionMemory                        │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: RAG Query (Streaming)

```
┌─────────────────────────────────────────────────────────────────┐
│ [8] SETUP SSE STREAMING                                          │
│                                                                  │
│     Headers:                                                     │
│     • Content-Type: text/event-stream                            │
│     • Cache-Control: no-cache, no-transform                      │
│     • Connection: keep-alive                                     │
│     • X-Accel-Buffering: no (Nginx bypass)                       │
│                                                                  │
│     res.write(": connected\n\n")  // Initial handshake           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [9] RAG API CALL                                                 │
│                                                                  │
│     ┌───────────────────────────────────────────────────────┐   │
│     │ LightRAG Server (http://171.244.137.76:9622)          │   │
│     │                                                        │   │
│     │ POST /api/healthcare/query/stream                     │   │
│     │ Body: {                                                │   │
│     │   query: "...",                                        │   │
│     │   mode: "mix",                                         │   │
│     │   stream: true,                                        │   │
│     │   include_references: true,                            │   │
│     │   include_chunk_content: true,                         │   │
│     │   history_turns: 3,                                    │   │
│     │   conversation_history: [...],                         │   │
│     │   session_memory: combinedMemory  ← Memory injected    │   │
│     │ }                                                      │   │
│     └───────────────────────────────────────────────────────┘   │
│                                                                  │
│     Response: NDJSON Stream                                      │
│     {"response": "Theo quy định..."}                             │
│     {"response": " tại Điều 5..."}                               │
│     {"references": [...]}                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [10] STREAM PROCESSING                                           │
│                                                                  │
│     WHILE (chunks from RAG API):                                 │
│       • Parse JSON chunk                                         │
│       • IF chunk.response:                                       │
│           fullAnswer += chunk.response                           │
│           res.write(`data: {"type":"content","content":"..."}`)  │
│           res.flush() ← Force immediate send                     │
│       • IF chunk.references:                                     │
│           ragReferences = chunk.references                       │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Post-Processing (Fire-and-Forget)

```
┌─────────────────────────────────────────────────────────────────┐
│ [11] SAVE ANSWER TO DATABASE                                     │
│                                                                  │
│     await storage.createAnswer({                                 │
│       queryId,                                                   │
│       content: fullAnswer,                                       │
│       confidenceLevel: "high"|"medium"|"low",                    │
│       isRefusal: boolean,                                        │
│       latencyMs,                                                 │
│       modelInfo: { model, usedExternalRag, ragReferences }       │
│     })                                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────────┐               ┌───────────────────────┐
│ [12A] UPDATE SESSION  │               │ [12B] STORE LONG-TERM │
│       MEMORY          │               │       MEMORY          │
│       (ASYNC)         │               │       (ASYNC)         │
├───────────────────────┤               ├───────────────────────┤
│                       │               │                       │
│ OpenAI GPT-4.1        │               │ SimpleMem Server      │
│                       │               │                       │
│ Prompt:               │               │ POST /users/{id}/qa   │
│ "Progressively        │               │ Body: {               │
│  summarize..."        │               │   question,           │
│                       │               │   answer,             │
│ Input:                │               │   clinical_context    │
│ • Current summary     │               │ }                     │
│ • New Q&A pair        │               │                       │
│                       │               │ → Extract atomic facts│
│ Output:               │               │ → Voyage AI embedding │
│ • New progressive     │               │ → Store to LanceDB    │
│   summary             │               │                       │
│                       │               │                       │
│ → Update PostgreSQL   │               │                       │
│   conversations       │               │                       │
│   .session_memory     │               │                       │
└───────────────────────┘               └───────────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [13] FINALIZE RESPONSE                                           │
│                                                                  │
│     res.write(`data: {                                           │
│       "type": "done",                                            │
│       "conversationId": 123,                                     │
│       "queryId": 456,                                            │
│       "answerId": 789,                                           │
│       "references": [...]                                        │
│     }`)                                                          │
│                                                                  │
│     res.end()                                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Sequence Diagram

```
┌──────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│Client│    │ Backend │    │SimpleMem │    │ LightRAG │    │ OpenAI  │
└──┬───┘    └────┬────┘    └────┬─────┘    └────┬─────┘    └────┬────┘
   │             │              │               │               │
   │ POST /ask   │              │               │               │
   │────────────>│              │               │               │
   │             │              │               │               │
   │             │ GET context  │               │               │
   │             │─────────────>│               │               │
   │             │              │               │               │
   │             │ memories     │               │               │
   │             │<─────────────│               │               │
   │             │              │               │               │
   │ SSE: connected             │               │               │
   │<────────────│              │               │               │
   │             │              │               │               │
   │             │ POST /query/stream           │               │
   │             │─────────────────────────────>│               │
   │             │              │               │               │
   │             │ chunk 1      │               │               │
   │             │<─────────────────────────────│               │
   │ SSE: content│              │               │               │
   │<────────────│              │               │               │
   │             │              │               │               │
   │             │ chunk 2      │               │               │
   │             │<─────────────────────────────│               │
   │ SSE: content│              │               │               │
   │<────────────│              │               │               │
   │             │              │               │               │
   │             │ references   │               │               │
   │             │<─────────────────────────────│               │
   │             │              │               │               │
   │ SSE: done   │              │               │               │
   │<────────────│              │               │               │
   │             │              │               │               │
   │             │ Store Q&A (async)            │               │
   │             │─────────────>│               │               │
   │             │              │               │               │
   │             │ Summarize (async)            │               │
   │             │─────────────────────────────────────────────>│
   │             │              │               │               │
```

## API Endpoints

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/conversations/ask` | Đặt câu hỏi (streaming SSE) |
| GET | `/api/conversations` | Lấy danh sách conversations |
| POST | `/api/conversations` | Tạo conversation mới |
| GET | `/api/conversations/:id` | Chi tiết conversation + queries |
| DELETE | `/api/conversations/:id` | Xóa conversation |

### Memory Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations/memory/stats` | Thống kê long-term memory |
| DELETE | `/api/conversations/memory/clear` | Xóa long-term memory |
| GET | `/api/conversations/memory/test` | Test SimpleMem connection |

## External Services

### 1. SimpleMem Server

- **URL**: `http://171.244.137.76:8100`
- **Purpose**: Long-term memory storage & retrieval
- **Technology**: Python FastAPI + LanceDB + Voyage AI

| Endpoint | Purpose |
|----------|---------|
| `POST /users/{id}/context` | Lấy context liên quan từ memory |
| `POST /users/{id}/qa` | Lưu Q&A pair vào memory |
| `GET /users/{id}/stats` | Thống kê memory |
| `DELETE /users/{id}/memories` | Xóa tất cả memories |

### 2. LightRAG Server

- **URL**: `http://171.244.137.76:9622`
- **Purpose**: Document retrieval & answer generation
- **Technology**: Python + Neo4j + Qdrant

| Endpoint | Purpose |
|----------|---------|
| `POST /api/healthcare/query/stream` | Query với streaming response |

### 3. OpenAI API

- **Purpose**: Session memory summarization
- **Model**: `gpt-4.1`
- **Usage**: Progressive summarization của conversation

## Data Models

### PostgreSQL Tables

```sql
-- conversations table
CREATE TABLE conversations (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  hospital_id INTEGER REFERENCES hospitals(id),
  title VARCHAR(255),
  session_memory TEXT,  -- Progressive summary
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

-- queries table
CREATE TABLE queries (
  id SERIAL PRIMARY KEY,
  conversation_id INTEGER REFERENCES conversations(id),
  user_id INTEGER REFERENCES users(id),
  question TEXT,
  clinical_context TEXT,
  created_at TIMESTAMP
);

-- answers table
CREATE TABLE answers (
  id SERIAL PRIMARY KEY,
  query_id INTEGER REFERENCES queries(id),
  content TEXT,
  confidence_level VARCHAR(10),  -- high, medium, low
  is_refusal BOOLEAN,
  refusal_reason TEXT,
  latency_ms INTEGER,
  model_info JSONB,
  created_at TIMESTAMP
);
```

### SimpleMem Memory Entry

```json
{
  "entry_id": "uuid",
  "lossless_restatement": "Người dùng bị u não.",
  "keywords": ["u não", "bệnh"],
  "timestamp": "2026-02-02T16:17:41",
  "persons": ["người dùng"],
  "topic": "Tình trạng sức khỏe"
}
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| SimpleMem context retrieval | ~2s | Voyage AI embedding + LanceDB search |
| LightRAG streaming | 3-10s | Depends on document complexity |
| Session memory update | ~1s | GPT-4.1 summarization (async) |
| Long-term memory store | ~2s | Fact extraction + embedding (async) |

## Error Handling

1. **SimpleMem unavailable**: Fallback to session memory only
2. **LightRAG error**: Return error SSE event
3. **OpenAI error**: Session memory not updated, logged but not blocking

## Configuration

```env
# RAG API
RAG_API_URL=http://171.244.137.76:9622/api/healthcare/query/stream

# SimpleMem
SIMPLEMEM_API_URL=http://171.244.137.76:8100

# OpenAI (for session memory)
AI_INTEGRATIONS_OPENAI_API_KEY=sk-...
AI_INTEGRATIONS_OPENAI_BASE_URL=https://api.openai.com/v1
```

---

*Generated: February 2026*
*Author: VietDoc-Analyzer Development Team*
