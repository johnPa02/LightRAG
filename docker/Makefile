.PHONY: hc-prod hc-dev biz-prod biz-dev hc-down biz-down down status help

COMPOSE = docker compose
HC_FILE = -f docker-compose.healthcare.yml
BIZ_FILE = -f docker-compose.business.yml

# ============================================
# Healthcare Domain
# ============================================
hc-prod: ## Start Healthcare PROD (port 9621)
	$(COMPOSE) -p hc-prod $(HC_FILE) --env-file envs/healthcare.prod.env up -d

hc-dev: ## Start Healthcare DEV (port 9622)
	$(COMPOSE) -p hc-dev $(HC_FILE) --env-file envs/healthcare.dev.env up -d

hc-prod-down: ## Stop Healthcare PROD
	$(COMPOSE) -p hc-prod $(HC_FILE) --env-file envs/healthcare.prod.env down

hc-dev-down: ## Stop Healthcare DEV
	$(COMPOSE) -p hc-dev $(HC_FILE) --env-file envs/healthcare.dev.env down

hc-prod-logs: ## Tail Healthcare PROD logs
	$(COMPOSE) -p hc-prod $(HC_FILE) --env-file envs/healthcare.prod.env logs -f lightrag

hc-dev-logs: ## Tail Healthcare DEV logs
	$(COMPOSE) -p hc-dev $(HC_FILE) --env-file envs/healthcare.dev.env logs -f lightrag

# ============================================
# Business Domain
# ============================================
biz-prod: ## Start Business PROD (port 9631)
	$(COMPOSE) -p biz-prod $(BIZ_FILE) --env-file envs/business.prod.env up -d

biz-dev: ## Start Business DEV (port 9632)
	$(COMPOSE) -p biz-dev $(BIZ_FILE) --env-file envs/business.dev.env up -d

biz-prod-down: ## Stop Business PROD
	$(COMPOSE) -p biz-prod $(BIZ_FILE) --env-file envs/business.prod.env down

biz-dev-down: ## Stop Business DEV
	$(COMPOSE) -p biz-dev $(BIZ_FILE) --env-file envs/business.dev.env down

biz-prod-logs: ## Tail Business PROD logs
	$(COMPOSE) -p biz-prod $(BIZ_FILE) --env-file envs/business.prod.env logs -f lightrag

biz-dev-logs: ## Tail Business DEV logs
	$(COMPOSE) -p biz-dev $(BIZ_FILE) --env-file envs/business.dev.env logs -f lightrag

# ============================================
# All Domains
# ============================================
all-prod: hc-prod biz-prod ## Start all PROD
all-dev: hc-dev biz-dev ## Start all DEV

down: ## Stop everything
	$(COMPOSE) -p hc-prod $(HC_FILE) --env-file envs/healthcare.prod.env down 2>/dev/null; \
	$(COMPOSE) -p hc-dev $(HC_FILE) --env-file envs/healthcare.dev.env down 2>/dev/null; \
	$(COMPOSE) -p biz-prod $(BIZ_FILE) --env-file envs/business.prod.env down 2>/dev/null; \
	$(COMPOSE) -p biz-dev $(BIZ_FILE) --env-file envs/business.dev.env down 2>/dev/null; true

# ============================================
# Utilities
# ============================================
status: ## Show running containers for all domains
	@echo "=== Healthcare ===" && \
	docker ps --filter "name=hc-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" && \
	echo "" && \
	echo "=== Business ===" && \
	docker ps --filter "name=biz-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

config-hc-prod: ## Validate Healthcare PROD config
	$(COMPOSE) -p hc-prod $(HC_FILE) --env-file envs/healthcare.prod.env config

config-biz-prod: ## Validate Business PROD config
	$(COMPOSE) -p biz-prod $(BIZ_FILE) --env-file envs/business.prod.env config

build: ## Rebuild all LightRAG images
	$(COMPOSE) -p hc-prod $(HC_FILE) --env-file envs/healthcare.prod.env build
	$(COMPOSE) -p biz-prod $(BIZ_FILE) --env-file envs/business.prod.env build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================
# Qwen Test
# ============================================
QWEN_FILE = -f docker-compose.qwen-test.yml

qwen-test: ## Start Qwen Test (port 9633, uses biz-dev DBs)
	$(COMPOSE) -p qwen-test $(QWEN_FILE) --env-file envs/business.qwen-test.env up -d --build

qwen-test-down: ## Stop Qwen Test
	$(COMPOSE) -p qwen-test $(QWEN_FILE) --env-file envs/business.qwen-test.env down

qwen-test-logs: ## Tail Qwen Test logs
	$(COMPOSE) -p qwen-test $(QWEN_FILE) --env-file envs/business.qwen-test.env logs -f lightrag

