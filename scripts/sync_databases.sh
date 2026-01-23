#!/bin/bash
# =============================================================================
# Database Sync Tool for LightRAG
# =============================================================================
# Sync data between Production and Development environments
#
# Usage:
#   ./scripts/sync_databases.sh <command> [options]
#
# Commands:
#   prod-to-dev    Sync from Production to Development
#   dev-to-prod    Sync from Development to Production (⚠️ careful!)
#   status         Show current status of both environments
#   backup-prod    Create backup of Production
#   backup-dev     Create backup of Development
#   restore        Restore from backup
#
# Options:
#   --all          Sync everything (files + neo4j + qdrant)
#   --neo4j        Sync only Neo4j
#   --qdrant       Sync only Qdrant
#   --files        Sync only files (inputs + rag_storage)
#   --force        Skip confirmation prompts
#
# Examples:
#   ./scripts/sync_databases.sh prod-to-dev --all
#   ./scripts/sync_databases.sh dev-to-prod --neo4j --force
#   ./scripts/sync_databases.sh status
#   ./scripts/sync_databases.sh backup-prod
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# =============================================================================
# Configuration
# =============================================================================

# Production
PROD_NEO4J_CONTAINER="neo4j"
PROD_NEO4J_HOST="localhost"
PROD_NEO4J_HTTP_PORT="7474"
PROD_NEO4J_BOLT_PORT="7687"
PROD_QDRANT_URL="http://localhost:6333"
PROD_INPUTS_DIR="data/inputs"
PROD_STORAGE_DIR="data/rag_storage"

# Development
DEV_NEO4J_CONTAINER="neo4j-dev"
DEV_NEO4J_HOST="localhost"
DEV_NEO4J_HTTP_PORT="7475"
DEV_NEO4J_BOLT_PORT="7688"
DEV_QDRANT_URL="http://localhost:6335"
DEV_INPUTS_DIR="data/inputs_dev"
DEV_STORAGE_DIR="data/rag_storage_dev"

# Neo4j credentials
NEO4J_USER="neo4j"
NEO4J_PASS="lightrag123"

# Backup directory
BACKUP_DIR="data/backups"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

confirm() {
    if [ "$FORCE" = true ]; then
        return 0
    fi
    echo -e "${YELLOW}$1${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

check_container() {
    if ! docker ps | grep -q "$1"; then
        log_error "Container '$1' is not running!"
        return 1
    fi
    return 0
}

get_neo4j_stats() {
    local container=$1
    local nodes=$(docker exec $container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
        "MATCH (n) RETURN count(n) as c" 2>/dev/null | tail -1 | tr -d ' ')
    local rels=$(docker exec $container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
        "MATCH ()-[r]->() RETURN count(r) as c" 2>/dev/null | tail -1 | tr -d ' ')
    echo "$nodes nodes, $rels relationships"
}

get_qdrant_stats() {
    local url=$1
    local collections=$(curl -s "$url/collections" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cols = data.get('result', {}).get('collections', [])
    total = 0
    names = []
    for c in cols:
        names.append(c['name'])
    print(f\"{len(names)} collections: {', '.join(names)}\")
except:
    print('N/A')
" 2>/dev/null)
    echo "$collections"
}

get_files_stats() {
    local dir=$1
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f | wc -l)
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "$count files ($size)"
    else
        echo "0 files (not exists)"
    fi
}

# =============================================================================
# Neo4j Sync Functions
# =============================================================================

sync_neo4j() {
    local source_container=$1
    local target_container=$2
    local direction=$3
    
    log_info "Syncing Neo4j ($direction)..."
    
    check_container $source_container || return 1
    check_container $target_container || return 1
    
    # Get labels from source
    local labels=$(docker exec $source_container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
        "CALL db.labels() YIELD label RETURN label" 2>/dev/null | tail -n +2)
    
    # Export from source
    log_info "Exporting from $source_container..."
    local export_file="sync_$(date +%Y%m%d_%H%M%S).json"
    docker exec $source_container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
        "CALL apoc.export.json.all('$export_file', {useTypes: true})" >/dev/null 2>&1
    
    # Copy export file
    log_info "Copying export file..."
    mkdir -p data/neo4j_backup
    docker cp $source_container:/var/lib/neo4j/import/$export_file data/neo4j_backup/
    docker cp data/neo4j_backup/$export_file $target_container:/var/lib/neo4j/import/
    
    # Create constraints in target
    log_info "Creating constraints in $target_container..."
    for label in $labels; do
        label=$(echo "$label" | tr -d '"' | tr -d ' ')
        if [ -n "$label" ]; then
            docker exec $target_container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:$label) REQUIRE n.neo4jImportId IS UNIQUE" 2>/dev/null || true
        fi
    done
    
    # Clear target database
    log_info "Clearing target database..."
    docker exec $target_container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
        "MATCH (n) DETACH DELETE n" 2>/dev/null || true
    
    # Import to target
    log_info "Importing to $target_container..."
    docker exec $target_container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
        "CALL apoc.import.json('$export_file')" >/dev/null 2>&1
    
    # Cleanup
    docker exec $source_container rm -f /var/lib/neo4j/import/$export_file 2>/dev/null || true
    docker exec $target_container rm -f /var/lib/neo4j/import/$export_file 2>/dev/null || true
    rm -f data/neo4j_backup/$export_file
    
    log_success "Neo4j sync completed!"
}

# =============================================================================
# Qdrant Sync Functions
# =============================================================================

sync_qdrant() {
    local source_url=$1
    local target_url=$2
    local direction=$3
    
    log_info "Syncing Qdrant ($direction)..."
    
    # Get collections from source
    local collections=$(curl -s "$source_url/collections" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'result' in data and 'collections' in data['result']:
    for c in data['result']['collections']:
        print(c['name'])
" 2>/dev/null)
    
    if [ -z "$collections" ]; then
        log_warn "No collections found in source"
        return 0
    fi
    
    mkdir -p data/qdrant_backup
    
    for collection in $collections; do
        log_info "Syncing collection: $collection"
        
        # Create snapshot
        local snapshot_result=$(curl -s -X POST "$source_url/collections/$collection/snapshots")
        local snapshot_name=$(echo "$snapshot_result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'result' in data and 'name' in data['result']:
    print(data['result']['name'])
" 2>/dev/null)
        
        if [ -n "$snapshot_name" ]; then
            # Download snapshot
            curl -s "$source_url/collections/$collection/snapshots/$snapshot_name" \
                -o "data/qdrant_backup/${collection}.snapshot"
            
            # Delete existing collection in target
            curl -s -X DELETE "$target_url/collections/$collection" >/dev/null 2>&1
            
            # Get collection config
            local config=$(curl -s "$source_url/collections/$collection" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'result' in data and 'config' in data['result']:
    config = data['result']['config']
    params = config.get('params', {})
    result = {
        'vectors': params.get('vectors', {'size': 1024, 'distance': 'Cosine'})
    }
    print(json.dumps(result))
" 2>/dev/null)
            
            # Create collection with config
            curl -s -X PUT "$target_url/collections/$collection" \
                -H "Content-Type: application/json" \
                -d "$config" >/dev/null 2>&1
            
            # Upload snapshot
            curl -s -X POST "$target_url/collections/$collection/snapshots/upload" \
                -H "Content-Type: multipart/form-data" \
                -F "snapshot=@data/qdrant_backup/${collection}.snapshot" >/dev/null 2>&1 || {
                    # Alternative: recover from snapshot
                    log_info "  Using snapshot recovery..."
                    # Copy snapshot to target container
                    if [ "$direction" = "prod-to-dev" ]; then
                        docker cp "data/qdrant_backup/${collection}.snapshot" qdrant-dev:/qdrant/snapshots/
                    else
                        docker cp "data/qdrant_backup/${collection}.snapshot" qdrant:/qdrant/snapshots/
                    fi
                }
            
            log_success "  Collection $collection synced"
        else
            log_warn "  Could not create snapshot for $collection"
        fi
    done
    
    log_success "Qdrant sync completed!"
}

# =============================================================================
# Files Sync Functions
# =============================================================================

sync_files() {
    local source_inputs=$1
    local target_inputs=$2
    local source_storage=$3
    local target_storage=$4
    local direction=$5
    
    log_info "Syncing files ($direction)..."
    
    # Create directories
    mkdir -p "$target_inputs" "$target_storage"
    
    # Sync inputs
    if [ -d "$source_inputs" ]; then
        log_info "Syncing inputs..."
        rsync -av --delete "$source_inputs/" "$target_inputs/"
        log_success "Inputs synced"
    fi
    
    # Sync rag_storage
    if [ -d "$source_storage" ]; then
        log_info "Syncing rag_storage..."
        rsync -av --delete "$source_storage/" "$target_storage/"
        log_success "rag_storage synced"
    fi
    
    log_success "Files sync completed!"
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup() {
    local env=$1  # prod or dev
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/${env}_${timestamp}"
    
    log_info "Creating backup for $env environment..."
    mkdir -p "$backup_path"
    
    if [ "$env" = "prod" ]; then
        local neo4j_container=$PROD_NEO4J_CONTAINER
        local qdrant_url=$PROD_QDRANT_URL
        local inputs_dir=$PROD_INPUTS_DIR
        local storage_dir=$PROD_STORAGE_DIR
    else
        local neo4j_container=$DEV_NEO4J_CONTAINER
        local qdrant_url=$DEV_QDRANT_URL
        local inputs_dir=$DEV_INPUTS_DIR
        local storage_dir=$DEV_STORAGE_DIR
    fi
    
    # Backup Neo4j
    if check_container $neo4j_container 2>/dev/null; then
        log_info "Backing up Neo4j..."
        docker exec $neo4j_container cypher-shell -u $NEO4J_USER -p $NEO4J_PASS \
            "CALL apoc.export.json.all('backup.json', {useTypes: true})" >/dev/null 2>&1
        docker cp $neo4j_container:/var/lib/neo4j/import/backup.json "$backup_path/neo4j.json"
        docker exec $neo4j_container rm -f /var/lib/neo4j/import/backup.json
    fi
    
    # Backup Qdrant
    log_info "Backing up Qdrant..."
    mkdir -p "$backup_path/qdrant"
    local collections=$(curl -s "$qdrant_url/collections" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for c in data.get('result', {}).get('collections', []):
        print(c['name'])
except:
    pass
" 2>/dev/null)
    
    for collection in $collections; do
        local snapshot_result=$(curl -s -X POST "$qdrant_url/collections/$collection/snapshots" 2>/dev/null)
        local snapshot_name=$(echo "$snapshot_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('result', {}).get('name', ''))
except:
    pass
" 2>/dev/null)
        if [ -n "$snapshot_name" ]; then
            curl -s "$qdrant_url/collections/$collection/snapshots/$snapshot_name" \
                -o "$backup_path/qdrant/${collection}.snapshot" 2>/dev/null
        fi
    done
    
    # Backup files
    log_info "Backing up files..."
    if [ -d "$inputs_dir" ]; then
        cp -r "$inputs_dir" "$backup_path/inputs"
    fi
    if [ -d "$storage_dir" ]; then
        cp -r "$storage_dir" "$backup_path/rag_storage"
    fi
    
    log_success "Backup created: $backup_path"
    echo "$backup_path"
}

# =============================================================================
# Status Function
# =============================================================================

show_status() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║              LightRAG Environment Status                       ║${NC}"
    echo -e "${BOLD}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo ""
    
    # Production
    echo -e "${CYAN}┌─────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│  PRODUCTION                                                     │${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────────┤${NC}"
    
    if check_container $PROD_NEO4J_CONTAINER 2>/dev/null; then
        echo -e "│  Neo4j:    ${GREEN}●${NC} Running  $(get_neo4j_stats $PROD_NEO4J_CONTAINER)"
    else
        echo -e "│  Neo4j:    ${RED}○${NC} Stopped"
    fi
    
    if curl -s "$PROD_QDRANT_URL/collections" >/dev/null 2>&1; then
        echo -e "│  Qdrant:   ${GREEN}●${NC} Running  $(get_qdrant_stats $PROD_QDRANT_URL)"
    else
        echo -e "│  Qdrant:   ${RED}○${NC} Stopped"
    fi
    
    echo -e "│  Inputs:   $(get_files_stats $PROD_INPUTS_DIR)"
    echo -e "│  Storage:  $(get_files_stats $PROD_STORAGE_DIR)"
    echo -e "${CYAN}└─────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # Development
    echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│  DEVELOPMENT                                                    │${NC}"
    echo -e "${YELLOW}├─────────────────────────────────────────────────────────────────┤${NC}"
    
    if check_container $DEV_NEO4J_CONTAINER 2>/dev/null; then
        echo -e "│  Neo4j:    ${GREEN}●${NC} Running  $(get_neo4j_stats $DEV_NEO4J_CONTAINER)"
    else
        echo -e "│  Neo4j:    ${RED}○${NC} Stopped"
    fi
    
    if curl -s "$DEV_QDRANT_URL/collections" >/dev/null 2>&1; then
        echo -e "│  Qdrant:   ${GREEN}●${NC} Running  $(get_qdrant_stats $DEV_QDRANT_URL)"
    else
        echo -e "│  Qdrant:   ${RED}○${NC} Stopped"
    fi
    
    echo -e "│  Inputs:   $(get_files_stats $DEV_INPUTS_DIR)"
    echo -e "│  Storage:  $(get_files_stats $DEV_STORAGE_DIR)"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

COMMAND=""
SYNC_NEO4J=false
SYNC_QDRANT=false
SYNC_FILES=false
FORCE=false
BACKUP_PATH=""
TARGET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        prod-to-dev|dev-to-prod|status|backup-prod|backup-dev|restore)
            COMMAND=$1
            shift
            ;;
        --all)
            SYNC_NEO4J=true
            SYNC_QDRANT=true
            SYNC_FILES=true
            shift
            ;;
        --neo4j)
            SYNC_NEO4J=true
            shift
            ;;
        --qdrant)
            SYNC_QDRANT=true
            shift
            ;;
        --files)
            SYNC_FILES=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --backup)
            BACKUP_PATH=$2
            shift 2
            ;;
        --target)
            TARGET=$2
            shift 2
            ;;
        -h|--help)
            head -40 "$0" | tail -35
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    status)
        show_status
        ;;
    
    prod-to-dev)
        echo ""
        echo -e "${BOLD}Syncing: PRODUCTION → DEVELOPMENT${NC}"
        echo ""
        
        if [ "$SYNC_FILES" = true ]; then
            sync_files "$PROD_INPUTS_DIR" "$DEV_INPUTS_DIR" "$PROD_STORAGE_DIR" "$DEV_STORAGE_DIR" "prod-to-dev"
            echo ""
        fi
        
        if [ "$SYNC_NEO4J" = true ]; then
            sync_neo4j "$PROD_NEO4J_CONTAINER" "$DEV_NEO4J_CONTAINER" "prod-to-dev"
            echo ""
        fi
        
        if [ "$SYNC_QDRANT" = true ]; then
            sync_qdrant "$PROD_QDRANT_URL" "$DEV_QDRANT_URL" "prod-to-dev"
            echo ""
        fi
        
        log_success "Sync completed! Remember to restart lightrag-dev: docker restart lightrag-dev"
        ;;
    
    dev-to-prod)
        echo ""
        echo -e "${RED}${BOLD}⚠️  WARNING: Syncing DEVELOPMENT → PRODUCTION${NC}"
        echo -e "${RED}This will OVERWRITE production data!${NC}"
        echo ""
        
        if ! confirm "Are you sure you want to continue?"; then
            log_info "Aborted."
            exit 0
        fi
        
        # Auto backup production first
        log_info "Creating backup of production first..."
        create_backup "prod"
        echo ""
        
        if [ "$SYNC_FILES" = true ]; then
            sync_files "$DEV_INPUTS_DIR" "$PROD_INPUTS_DIR" "$DEV_STORAGE_DIR" "$PROD_STORAGE_DIR" "dev-to-prod"
            echo ""
        fi
        
        if [ "$SYNC_NEO4J" = true ]; then
            sync_neo4j "$DEV_NEO4J_CONTAINER" "$PROD_NEO4J_CONTAINER" "dev-to-prod"
            echo ""
        fi
        
        if [ "$SYNC_QDRANT" = true ]; then
            sync_qdrant "$DEV_QDRANT_URL" "$PROD_QDRANT_URL" "dev-to-prod"
            echo ""
        fi
        
        log_success "Sync completed! Remember to restart lightrag: docker restart lightrag"
        ;;
    
    backup-prod)
        create_backup "prod"
        ;;
    
    backup-dev)
        create_backup "dev"
        ;;
    
    restore)
        if [ -z "$BACKUP_PATH" ] || [ -z "$TARGET" ]; then
            log_error "Usage: $0 restore --backup <path> --target <prod|dev>"
            exit 1
        fi
        log_warn "Restore functionality coming soon..."
        ;;
    
    *)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  prod-to-dev    Sync from Production to Development"
        echo "  dev-to-prod    Sync from Development to Production"
        echo "  status         Show current status"
        echo "  backup-prod    Create backup of Production"
        echo "  backup-dev     Create backup of Development"
        echo ""
        echo "Options:"
        echo "  --all          Sync everything"
        echo "  --neo4j        Sync only Neo4j"
        echo "  --qdrant       Sync only Qdrant"
        echo "  --files        Sync only files"
        echo "  --force        Skip confirmation"
        echo ""
        echo "Examples:"
        echo "  $0 prod-to-dev --all"
        echo "  $0 dev-to-prod --neo4j --force"
        echo "  $0 status"
        exit 1
        ;;
esac
