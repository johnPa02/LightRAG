import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4_1_complete, openai_embed
from lightrag.utils import logger, set_verbose_debug, EmbeddingFunc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
WORKING_DIR = "./law"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    # Configure embedding function with text-embedding-3-large
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,  # text-embedding-3-large dimension
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
        ),
    )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=gpt_4_1_complete,
        chunk_token_size=3000,  # Increase token limit per chunk for legal articles
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status

    return rag


async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        # Clear old data files (UNCOMMENT TO REBUILD FROM SCRATCH)
        # files_to_delete = [
        #     "graph_chunk_entity_relation.graphml",
        #     "kv_store_doc_status.json",
        #     "kv_store_full_docs.json",
        #     "kv_store_text_chunks.json",
        #     "vdb_chunks.json",
        #     "vdb_entities.json",
        #     "vdb_relationships.json",
        # ]

        # for file in files_to_delete:
        #     file_path = os.path.join(WORKING_DIR, file)
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #         print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        # test_text = ["This is a test string for embedding."]
        # embedding = await rag.embedding_func(test_text)
        # embedding_dim = embedding.shape[1]
        # print("\n=======================")
        # print("Test embedding function")
        # print("========================")
        # print(f"Test dict: {test_text}")
        # print(f"Detected embedding dimension: {embedding_dim}\n\n")

        # Insert document (UNCOMMENT TO REBUILD FROM SCRATCH)
        with open("./law.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(
                f.read(),
                split_by_character="\n\n",  # Split by double newline (paragraph)
                split_by_character_only=True,  # Keep each paragraph as one chunk
            )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================") 
        print(
            await rag.aquery(
                "Hồ sơ thay đổi người đứng đầu văn phòng đại diện theo Nghị định 168 gồm những gì?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
