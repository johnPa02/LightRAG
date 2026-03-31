# Multi-Domain Docker Environment

Tổ chức Docker cho 2 domain (Healthcare, Business) × 2 môi trường (Prod, Dev).

## Quick Start

```bash
cd docker

# Healthcare
make hc-prod          # Start HC prod (:9621)
make hc-dev           # Start HC dev  (:9622)

# Business
make biz-prod         # Start Biz prod (:9631)
make biz-dev          # Start Biz dev  (:9632)

# Utilities
make status           # Show all running containers
make down             # Stop everything
make help             # Show all commands
```

## Port Allocation

| Stack      | LightRAG | Neo4j HTTP | Neo4j Bolt | Qdrant HTTP | Qdrant gRPC |
|------------|----------|------------|------------|-------------|-------------|
| HC Prod    | 9621     | 7474       | 7687       | 6333        | 6334        |
| HC Dev     | 9622     | 7475       | 7688       | 6335        | 6336        |
| Biz Prod   | 9631     | 7484       | 7697       | 6343        | 6344        |
| Biz Dev    | 9632     | 7485       | 7698       | 6345        | 6346        |

## Architecture

```
docker compose -p <project-name> -f docker-compose.<domain>.yml --env-file envs/<domain>.<env>.env up -d
```

Each stack gets isolated **volumes** and **networks** via Docker Compose project name prefix:
- `hc-prod_neo4j_data`, `hc-dev_neo4j_data`
- `biz-prod_qdrant_data`, `biz-dev_qdrant_data`

## File Structure

```
docker/
├── docker-compose.healthcare.yml   # HC stack (neo4j + qdrant + lightrag)
├── docker-compose.business.yml     # Biz stack (neo4j + qdrant + lightrag)
├── envs/
│   ├── healthcare.prod.env         # HC production config
│   ├── healthcare.dev.env          # HC development config
│   ├── business.prod.env           # Biz production config
│   └── business.dev.env            # Biz development config
├── Makefile                        # Shortcut commands
└── README.md                       # This file
```

## Environment Files

Env files chứa API keys → đã được thêm vào `.gitignore`.
Khi deploy lên server mới, copy env files hoặc tạo mới từ template.

## Migration from Old Setup

File cũ (`docker-compose.yml`, `docker-compose-dev.yml`, `.env`, `.env.dev`) vẫn giữ nguyên.
Có thể chạy song song trong thời gian chuyển đổi.
