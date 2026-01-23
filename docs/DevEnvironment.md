# Môi Trường Development

Hướng dẫn thiết lập và quản lý môi trường DEV tách biệt với Production.

## Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRODUCTION                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │  LightRAG   │    │   Neo4j     │    │   Qdrant    │                      │
│  │  :9621      │───▶│  :7474/7687 │    │  :6333/6334 │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│         │                  │                  │                              │
│         └──────────────────┼──────────────────┘                              │
│                            │                                                 │
│                      ┌─────▼─────┐                                           │
│                      │   SYNC    │  ./scripts/sync_databases.sh              │
│                      └─────┬─────┘                                           │
│                            │                                                 │
│         ┌──────────────────┼──────────────────┐                              │
│         │                  │                  │                              │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐                      │
│  │ LightRAG    │    │  Neo4j      │    │   Qdrant    │                      │
│  │ DEV :9622   │───▶│  DEV :7475  │    │  DEV :6335  │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│                              DEVELOPMENT                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Port Mapping

| Service | Production | Development |
|---------|------------|-------------|
| **LightRAG API/WebUI** | 9621 | 9622 |
| **Neo4j HTTP** | 7474 | 7475 |
| **Neo4j Bolt** | 7687 | 7688 |
| **Qdrant HTTP** | 6333 | 6335 |
| **Qdrant gRPC** | 6334 | 6336 |

## Cấu Trúc Thư Mục

```
LightRAG/
├── data/
│   ├── inputs/              # Production input files
│   ├── inputs_dev/          # Development input files
│   ├── rag_storage/         # Production KV storage
│   ├── rag_storage_dev/     # Development KV storage
│   ├── neo4j_backup/        # Neo4j export files
│   └── qdrant_backup/       # Qdrant snapshots
├── .env                     # Production environment
├── .env.dev                 # Development environment
├── docker-compose.yml       # Production stack
└── docker-compose-dev.yml   # Development stack
```

## Khởi Động Môi Trường

### Production
```bash
docker compose up -d
```

### Development
```bash
docker compose -f docker-compose-dev.yml up -d
```

### Cả hai cùng lúc
```bash
docker compose up -d
docker compose -f docker-compose-dev.yml up -d
```

## Sync Dữ Liệu

### Sync từ Production → Development
```bash
# Sync toàn bộ (files + Neo4j + Qdrant)
./scripts/sync_databases.sh prod-to-dev --all

# Sync từng phần
./scripts/sync_databases.sh prod-to-dev --neo4j
./scripts/sync_databases.sh prod-to-dev --qdrant
./scripts/sync_databases.sh prod-to-dev --files
```

### Sync từ Development → Production
```bash
# ⚠️ CẨN THẬN: Sẽ ghi đè dữ liệu production!

# Sync toàn bộ
./scripts/sync_databases.sh dev-to-prod --all

# Sync từng phần
./scripts/sync_databases.sh dev-to-prod --neo4j
./scripts/sync_databases.sh dev-to-prod --qdrant
./scripts/sync_databases.sh dev-to-prod --files
```

### Chỉ xem thông tin (không sync)
```bash
./scripts/sync_databases.sh status
```

## URLs Truy Cập

### Production
- **LightRAG API**: http://localhost:9621
- **LightRAG WebUI**: http://localhost:9621
- **Neo4j Browser**: http://localhost:7474
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Development
- **LightRAG API**: http://localhost:9622
- **LightRAG WebUI**: http://localhost:9622
- **Neo4j Browser**: http://localhost:7475
- **Qdrant Dashboard**: http://localhost:6335/dashboard

## Workflow Phát Triển

### 1. Clone data từ prod để test
```bash
# Khởi động môi trường dev
docker compose -f docker-compose-dev.yml up -d

# Sync data từ prod
./scripts/sync_databases.sh prod-to-dev --all

# Restart để load data mới
docker restart lightrag-dev
```

### 2. Test với dữ liệu mới
```bash
# Thêm file mới vào dev
cp my_new_document.txt data/inputs_dev/default/__enqueued__/

# API sẽ tự động xử lý file mới
```

### 3. Đẩy thay đổi lên prod (sau khi test thành công)
```bash
# ⚠️ Backup prod trước
./scripts/sync_databases.sh backup-prod

# Sync dev lên prod
./scripts/sync_databases.sh dev-to-prod --all

# Restart prod
docker restart lightrag
```

## Backup & Restore

### Tạo backup Production
```bash
./scripts/sync_databases.sh backup-prod
# Output: data/backups/prod_YYYYMMDD_HHMMSS/
```

### Tạo backup Development
```bash
./scripts/sync_databases.sh backup-dev
# Output: data/backups/dev_YYYYMMDD_HHMMSS/
```

### Restore từ backup
```bash
./scripts/sync_databases.sh restore --backup data/backups/prod_20260122_120000 --target prod
./scripts/sync_databases.sh restore --backup data/backups/dev_20260122_120000 --target dev
```

## Troubleshooting

### Container không start
```bash
# Check logs
docker logs lightrag-dev
docker logs neo4j-dev
docker logs qdrant-dev

# Rebuild
docker compose -f docker-compose-dev.yml up -d --build
```

### Neo4j sync failed
```bash
# Kiểm tra kết nối
docker exec neo4j-dev cypher-shell -u neo4j -p lightrag123 "RETURN 1"

# Xem số nodes
docker exec neo4j-dev cypher-shell -u neo4j -p lightrag123 "MATCH (n) RETURN count(n)"
```

### Qdrant sync failed
```bash
# Kiểm tra collections
curl http://localhost:6335/collections

# Xem chi tiết collection
curl http://localhost:6335/collections/lightrag_vdb_entities
```

### Reset môi trường dev hoàn toàn
```bash
# Stop và xóa containers + volumes
docker compose -f docker-compose-dev.yml down -v

# Xóa data folders
rm -rf data/inputs_dev data/rag_storage_dev

# Khởi động lại
docker compose -f docker-compose-dev.yml up -d

# Sync lại từ prod
./scripts/sync_databases.sh prod-to-dev --all
```

## Lưu Ý Quan Trọng

1. **Không bao giờ** sync dev-to-prod mà không backup trước
2. **Kiểm tra kỹ** data trước khi sync lên prod
3. **Restart container** sau khi sync để load data mới
4. Dev và Prod dùng **network riêng biệt**, không ảnh hưởng lẫn nhau
5. Mount source code (`./lightrag`) chỉ có ở dev để hot-reload
