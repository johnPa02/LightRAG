from __future__ import annotations
from functools import partial
from pathlib import Path

import asyncio
import json
import json_repair
import re
from typing import Any, AsyncIterator, overload, Literal
from collections import Counter, defaultdict

from lightrag.exceptions import (
    PipelineCancelledException,
    ChunkTokenLimitExceededError,
)
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    Tokenizer,
    is_float_regex,
    sanitize_and_normalize_extracted_text,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    use_llm_func_with_cache,
    update_chunk_cache_list,
    remove_think_tags,
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    process_chunks_unified,
    safe_vdb_operation_with_exception,
    create_prefixed_exception,
    fix_tuple_delimiter_corruption,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    apply_source_ids_limit,
    merge_source_ids,
    make_relation_chunk_key,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
    QueryContextResult,
)
from lightrag.prompt import PROMPTS
from lightrag.domains.base import DomainConfig, get_prompt
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)
from lightrag.kg.shared_storage import get_storage_keyed_lock
import time
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def _truncate_entity_identifier(
    identifier: str, limit: int, chunk_key: str, identifier_role: str
) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""

    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]  # Show first 20 characters as preview
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    seperator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = seperator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit * 0.9)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens ({summary_token_count}) exceeds 90% of embedding limit "
                f"({embedding_token_limit}) for {description_type}: {description_name}"
            )

    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 feilds on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.warning(
                f"Entity extraction error: entity name became empty after cleaning. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if (
        len(record_attributes) != 5 or "relation" not in record_attributes[0]
    ):  # treat "relationship" and "relation" interchangeable
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on REALTION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.warning(
                f"Relationship extraction error: source entity became empty after cleaning. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.warning(
                f"Relationship extraction error: target entity became empty after cleaning. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])

        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],
    relationships_to_rebuild: dict[tuple[str, str], list[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> list of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> list of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        entity_chunks_storage: KV storage maintaining full chunk IDs per entity
        relation_chunks_storage: KV storage maintaining full chunk IDs per relation
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    # cached_results： chunk_id -> [list of (extraction_result, create_time) from LLM cache sorted by create_time of the first extraction_result]
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = "No cached extraction results found, cannot rebuild"
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
        return

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for result in results:
                entities, relationships = await _rebuild_from_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    chunk_id=chunk_id,
                    extraction_result=result[0],
                    timestamp=result[1],
                )

                # Merge entities and relationships from this extraction result
                # Compare description lengths and keep the better version for the same chunk_id
                for entity_name, entity_list in entities.items():
                    if entity_name not in chunk_entities[chunk_id]:
                        # New entity for this chunk_id
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    elif len(chunk_entities[chunk_id][entity_name]) == 0:
                        # Empty list, add the new entities
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_entities[chunk_id][entity_name][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(entity_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new entity that has longer description
                            chunk_entities[chunk_id][entity_name] = list(entity_list)
                        # Otherwise keep existing version

                # Compare description lengths and keep the better version for the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if rel_key not in chunk_relationships[chunk_id]:
                        # New relationship for this chunk_id
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    elif len(chunk_relationships[chunk_id][rel_key]) == 0:
                        # Empty list, add the new relationships
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_relationships[chunk_id][rel_key][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(rel_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new relationship that has longer description
                            chunk_relationships[chunk_id][rel_key] = list(rel_list)
                        # Otherwise keep existing version

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_entities_count += 1
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                    rebuilt_relationships_count += 1
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                # Task completed successfully, retrieve result to mark as processed
                task.result()
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Re-raise the first exception to notify the caller
        raise first_exception

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text, where:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # Read from storage
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    for chunk_data in chunk_data_list:
        if chunk_data and isinstance(chunk_data, dict):
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_entry in cache_data_list:
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk and collect earliest times
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]

    # Sort cached_results by the earliest create_time of each chunk
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results  # each item: list(extraction_result, create_time)


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        record_delimiter (str): Delimiter for records
        completion_delimiter (str): Delimiter for completion
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if completion_delimiter not in result:
        logger.warning(
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
        )

    # Split LLL output result to records by "\n"
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to seperate record instead of "\n"
    fixed_records = []
    for record in records:
        record = record.strip()
        if record is None:
            continue
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for entity_record in entity_records:
            if not entity_record.startswith("entity") and not entity_record.startswith(
                "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                if not entity_relation_record.startswith(
                    "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records = fixed_records + [entity_relation_record]

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record seperators instead new-line"
        )

    for record in fixed_records:
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core.lower():
            # change delimiter_core to lower case, and fix again
            delimiter_core = delimiter_core.lower()
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if entity_data is not None:
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            continue

        # Try to parse as relationship
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


async def _rebuild_from_extraction_result(
    text_chunks_storage: BaseKVStorage,
    extraction_result: str,
    chunk_id: str,
    timestamp: int,
) -> tuple[dict, dict]:
    """Parse cached extraction result using the same logic as extract_entities

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path from storage
    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )

    # Call the shared processing function
    return await _process_extraction_result(
        extraction_result,
        chunk_id,
        timestamp,
        file_path,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: list[str],
    chunk_entities: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single entity from cached extraction results"""

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str,
        entity_type: str,
        file_paths: list[str],
        source_chunk_ids: list[str],
        truncation_info: str = "",
    ):
        try:
            # Update entity in graph storage (critical path)
            updated_entity_data = {
                **current_entity,
                "description": final_description,
                "entity_type": entity_type,
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }
            await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

            # Update entity in vector database (equally critical)
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_content = f"{entity_name}\n{final_description}"

            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Use safe operation wrapper - VDB failure must throw exception
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if entity_chunks_storage is not None and normalized_chunk_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # Collect all entity data from relevant (limited) chunks
    all_entity_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths_list = []
    seen_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    max_file_paths = global_config.get("max_file_paths")
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)
    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: list[str],
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if relation_chunks_storage is not None and normalized_chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from relevant chunks
    all_relationship_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )

    if not all_relationship_data:
        logger.warning(f"No relation data found for `{src}-{tgt}`")
        return

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths_list = []
    seen_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_path = rel_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths")
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))

    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )

    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Generate final description from relations or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else current_relationship.get("file_path", "unknown_source"),
        "truncate": truncation_info,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else current_relationship.get("description", "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    for node_id in {src, tgt}:
        if not (await knowledge_graph_inst.has_node(node_id)):
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            if entities_vdb is not None:
                entity_vdb_id = compute_mdhash_id(node_id, prefix="ent-")
                entity_content = f"{node_id}\n{node_description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    # Sort src and tgt to ensure consistent ordering (smaller string first)
    if src > tgt:
        src, tgt = tgt, src
    try:
        rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

        # Delete old vector records first (both directions to be safe)
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Insert new vector record
        rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{src}-{tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = f"Failed to rebuild relationship storage for `{src}-{tgt}`: {e}"
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{src}`~`{tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)

    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 1. Get existing node data from knowledge graph
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        already_entity_types.append(already_node.get("entity_type", ""))
        already_source_ids.extend(already_node.get("source_id", "").split(GRAPH_FIELD_SEP))
        already_file_paths.extend(already_node.get("file_path", "").split(GRAPH_FIELD_SEP))
        already_description.extend(already_node.get("description", "").split(GRAPH_FIELD_SEP))

    new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

    existing_full_source_ids = []
    if entity_chunks_storage is not None:
        stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merging new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if entity_chunks_storage is not None and full_source_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_entity")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_nodes = []
        for dp in nodes_data:
            source_id = dp.get("source_id")
            # Skip descriptions sourced from chunks dropped by the limitation cap
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            filtered_nodes.append(dp)
        nodes_data = filtered_nodes
    else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
        nodes_data = list(nodes_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not nodes_data
    ):
        if already_node:
            logger.info(
                f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
            )
            existing_node_data = dict(already_node)
            return existing_node_data
        else:
            logger.error(f"Internal Error: already_node missing for `{entity_name}`")
            raise ValueError(
                f"Internal Error: already_node missing for `{entity_name}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize entity type by highest count
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 7. Deduplicate nodes by description, keeping first occurrence in the same document
    unique_nodes = {}
    for dp in nodes_data:
        desc = dp.get("description")
        if not desc:
            continue
        if desc not in unique_nodes:
            unique_nodes[desc] = dp

    # Sort description by timestamp, then by description length when timestamps are the same
    sorted_nodes = sorted(
        unique_nodes.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_nodes]

    # Combine already_description with sorted new sorted descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Entity {entity_name} has no description")
        raise ValueError(f"Entity {entity_name} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during entity summary")

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        entity_name,
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Indicating file_path has been truncated before

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in nodes_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )
        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10.Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(nodes_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
        truncate=truncation_info,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    if entity_vdb is not None:
        entity_vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
        entity_content = f"{entity_name}\n{description}"
        data_for_vdb = {
            entity_vdb_id: {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "content": entity_content,
                "source_id": source_id,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
            operation_name="entity_upsert",
            entity_name=entity_name,
            max_retries=3,
            retry_delay=0.1,
        )
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    if src_id == tgt_id:
        return None

    already_edge = None
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # 1. Get existing edge data from graph storage
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 1.0 if missing
            already_weights.append(already_edge.get("weight", 1.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    already_edge["source_id"].split(GRAPH_FIELD_SEP)
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    already_edge["file_path"].split(GRAPH_FIELD_SEP)
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.extend(
                    already_edge["description"].split(GRAPH_FIELD_SEP)
                )

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]

    storage_key = make_relation_chunk_key(src_id, tgt_id)
    existing_full_source_ids = []
    if relation_chunks_storage is not None:
        stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merge new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if relation_chunks_storage is not None and full_source_ids:
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_relation")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{src_id}`~`{tgt_id}`",
    )
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_edges = []
        for dp in edges_data:
            source_id = dp.get("source_id")
            # Skip relationship fragments sourced from chunks dropped by keep oldest cap
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            filtered_edges.append(dp)
        edges_data = filtered_edges
    else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
        edges_data = list(edges_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not edges_data
    ):
        if already_edge:
            logger.info(
                f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
            )
            existing_edge_data = dict(already_edge)
            return existing_edge_data
        else:
            logger.error(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )
            raise ValueError(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize weight by summing new edges and existing weights
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)

    # 6.2 Finalize keywords by merging existing and new keywords
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for keyword_str in already_keywords:
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    # 7. Deduplicate by description, keeping first occurrence in the same document
    unique_edges = {}
    for dp in edges_data:
        description_value = dp.get("description")
        if not description_value:
            continue
        if description_value not in unique_edges:
            unique_edges[description_value] = dp

    # Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
    sorted_edges = sorted(
        unique_edges.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_edges]

    # Combine already_description with sorted new descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Relation {src_id}~{tgt_id} has no description")
        raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during relation summary"
                )

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Relation",
        f"({src_id}, {tgt_id})",
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS limit
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Track if already_file_paths contains placeholder

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        # Check if this is a placeholder record
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in edges_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths")

    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10. Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    for need_insert_id in [src_id, tgt_id]:
        # Optimization: Use get_node instead of has_node + get_node
        existing_node = await knowledge_graph_inst.get_node(need_insert_id)

        if existing_node is None:
            # Node doesn't exist - create new node
            node_created_at = int(time.time())
            node_data = {
                "entity_id": need_insert_id,
                "source_id": source_id,
                "description": description,
                "entity_type": "UNKNOWN",
                "file_path": file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(need_insert_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None:
                chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                if chunk_ids:
                    await entity_chunks_storage.upsert(
                        {
                            need_insert_id: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )

            if entity_vdb is not None:
                entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                entity_content = f"{need_insert_id}\n{description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": need_insert_id,
                        "source_id": source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="added_entity_upsert",
                    entity_name=need_insert_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

            # Track entities added during edge processing
            if added_entities is not None:
                entity_data = {
                    "entity_name": need_insert_id,
                    "entity_type": "UNKNOWN",
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": node_created_at,
                }
                added_entities.append(entity_data)
        else:
            # Node exists - update its source_ids by merging with new source_ids
            updated = False  # Track if any update occurred

            # 1. Get existing full source_ids from entity_chunks_storage
            existing_full_source_ids = []
            if entity_chunks_storage is not None:
                stored_chunks = await entity_chunks_storage.get_by_id(need_insert_id)
                if stored_chunks and isinstance(stored_chunks, dict):
                    existing_full_source_ids = [
                        chunk_id
                        for chunk_id in stored_chunks.get("chunk_ids", [])
                        if chunk_id
                    ]

            # If not in entity_chunks_storage, get from graph database
            if not existing_full_source_ids:
                if existing_node.get("source_id"):
                    existing_full_source_ids = existing_node["source_id"].split(
                        GRAPH_FIELD_SEP
                    )

            # 2. Merge with new source_ids from this relationship
            new_source_ids_from_relation = [
                chunk_id for chunk_id in source_ids if chunk_id
            ]
            merged_full_source_ids = merge_source_ids(
                existing_full_source_ids, new_source_ids_from_relation
            )

            # 3. Save merged full list to entity_chunks_storage (conditional)
            if (
                entity_chunks_storage is not None
                and merged_full_source_ids != existing_full_source_ids
            ):
                updated = True
                await entity_chunks_storage.upsert(
                    {
                        need_insert_id: {
                            "chunk_ids": merged_full_source_ids,
                            "count": len(merged_full_source_ids),
                        }
                    }
                )

            # 4. Apply source_ids limit for graph and vector db
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            max_source_limit = global_config.get("max_source_ids_per_entity")
            limited_source_ids = apply_source_ids_limit(
                merged_full_source_ids,
                max_source_limit,
                limit_method,
                identifier=f"`{need_insert_id}`",
            )

            # 5. Update graph database and vector database with limited source_ids (conditional)
            limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

            if limited_source_id_str != existing_node.get("source_id", ""):
                updated = True
                updated_node_data = {
                    **existing_node,
                    "source_id": limited_source_id_str,
                }
                await knowledge_graph_inst.upsert_node(
                    need_insert_id, node_data=updated_node_data
                )

                # Update vector database
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                    entity_content = (
                        f"{need_insert_id}\n{existing_node.get('description', '')}"
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_name": need_insert_id,
                            "source_id": limited_source_id_str,
                            "entity_type": existing_node.get("entity_type", "UNKNOWN"),
                            "file_path": existing_node.get(
                                "file_path", "unknown_source"
                            ),
                        }
                    }
                    await safe_vdb_operation_with_exception(
                        operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                        operation_name="existing_entity_update",
                        entity_name=need_insert_id,
                        max_retries=3,
                        retry_delay=0.1,
                    )

            # 6. Log once at the end if any update occurred
            if updated:
                status_message = f"Chunks appended from relation: `{need_insert_id}`"
                logger.info(status_message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = status_message
                        pipeline_status["history_messages"].append(status_message)

    edge_created_at = int(time.time())
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=edge_created_at,
            truncate=truncation_info,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
        weight=weight,
    )

    # Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
    if src_id > tgt_id:
        src_id, tgt_id = tgt_id, src_id

    if relationships_vdb is not None:
        rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )
        rel_content = f"{keywords}\t{src_id}\n{tgt_id}\n{description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": rel_content,
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
            operation_name="relationship_upsert",
            entity_name=f"{src_id}-{tgt_id}",
            max_retries=3,
            retry_delay=0.2,
        )

    return edge_data


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> None:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
    """

    # Check for cancellation at the start of merge
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # ===== Phase 1: Process all entities concurrently =====
    log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_entity_name(entity_name, entities):
        async with semaphore:
            # Check for cancellation before processing entity
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during entity merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    logger.debug(f"Processing entity {entity_name}")
                    entity_data = await _merge_nodes_then_upsert(
                        entity_name,
                        entities,
                        knowledge_graph_inst,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        entity_chunks_storage,
                    )

                    return entity_data

                except Exception as e:
                    error_msg = f"Error processing entity `{entity_name}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                            pipeline_status is not None
                            and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"`{entity_name}`"
                    )
                    raise prefixed_exception from e

    # Create entity processing tasks
    entity_tasks = []
    for entity_name, entities in all_nodes.items():
        task = asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        entity_tasks.append(task)

    # Execute entity tasks with error handling
    processed_entities = []
    if entity_tasks:
        done, pending = await asyncio.wait(
            entity_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None
        processed_entities = []

        for task in done:
            try:
                result = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                processed_entities.append(result)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    processed_entities.append(result)

        if first_exception is not None:
            raise first_exception

    # ===== Phase 2: Process all relationships concurrently =====
    log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            # Check for cancellation before processing edges
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during relation merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])

            async with get_storage_keyed_lock(
                sorted_edge_key,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    added_entities = []  # Track entities added during edge processing

                    logger.debug(f"Processing relation {sorted_edge_key}")
                    edge_data = await _merge_edges_then_upsert(
                        edge_key[0],
                        edge_key[1],
                        edges,
                        knowledge_graph_inst,
                        relationships_vdb,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        added_entities,  # Pass list to collect added entities
                        relation_chunks_storage,
                        entity_chunks_storage,  # Add entity_chunks_storage parameter
                    )

                    if edge_data is None:
                        return None, []

                    return edge_data, added_entities

                except Exception as e:
                    error_msg = f"Error processing relation `{sorted_edge_key}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                            pipeline_status is not None
                            and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"{sorted_edge_key}"
                    )
                    raise prefixed_exception from e

    # Create relationship processing tasks
    edge_tasks = []
    for edge_key, edges in all_edges.items():
        task = asyncio.create_task(_locked_process_edges(edge_key, edges))
        edge_tasks.append(task)

    # Execute relationship tasks with error handling
    processed_edges = []
    all_added_entities = []

    if edge_tasks:
        done, pending = await asyncio.wait(
            edge_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None

        for task in done:
            try:
                edge_data, added_entities = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                if edge_data is not None:
                    processed_edges.append(edge_data)
                all_added_entities.extend(added_entities)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    edge_data, added_entities = result
                    if edge_data is not None:
                        processed_edges.append(edge_data)
                    all_added_entities.extend(added_entities)

        if first_exception is not None:
            raise first_exception

    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            final_entity_names = set()

            # Add original processed entities
            for entity_data in processed_entities:
                if entity_data and entity_data.get("entity_name"):
                    final_entity_names.add(entity_data["entity_name"])

            # Add entities that were added during relationship processing
            for added_entity in all_added_entities:
                if added_entity and added_entity.get("entity_name"):
                    final_entity_names.add(added_entity["entity_name"])

            # Collect all relation pairs
            final_relation_pairs = set()
            for edge_data in processed_edges:
                if edge_data:
                    src_id = edge_data.get("src_id")
                    tgt_id = edge_data.get("tgt_id")
                    if src_id and tgt_id:
                        relation_pair = tuple(sorted([src_id, tgt_id]))
                        final_relation_pairs.add(relation_pair)

            log_message = f"Phase 3: Updating final {len(final_entity_names)}({len(processed_entities)}+{len(all_added_entities)}) entities and  {len(final_relation_pairs)} relations from {doc_id}"
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            # Update storage
            if final_entity_names:
                await full_entities_storage.upsert(
                    {
                        doc_id: {
                            "entity_names": list(final_entity_names),
                            "count": len(final_entity_names),
                        }
                    }
                )

            if final_relation_pairs:
                await full_relations_storage.upsert(
                    {
                        doc_id: {
                            "relation_pairs": [
                                list(pair) for pair in final_relation_pairs
                            ],
                            "count": len(final_relation_pairs),
                        }
                    }
                )

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: {len(final_entity_names)} entities (original: {len(processed_entities)}, added: {len(all_added_entities)}), {len(final_relation_pairs)} relations"
            )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    entity_types = global_config["addon_params"].get(
        "entity_types", DEFAULT_ENTITY_TYPES
    )

    examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_system_prompt"
        ].format(**{**context_base, "input_text": content})
        entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": content}
        )
        entity_continue_extraction_user_prompt = PROMPTS[
            "entity_continue_extraction_user_prompt"
        ].format(**{**context_base, "input_text": content})

        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            timestamp,
            file_path,
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        if entity_extract_max_gleaning > 0:
            glean_result, timestamp = await use_llm_func_with_cache(
                entity_continue_extraction_user_prompt,
                use_llm_func,
                system_prompt=entity_extraction_system_prompt,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result,
                chunk_key,
                timestamp,
                file_path,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )

            # Merge results - compare description lengths to choose better version
            for entity_name, glean_entities in glean_nodes.items():
                if entity_name in maybe_nodes:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_nodes[entity_name][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_entities[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_nodes[entity_name] = list(glean_entities)
                    # Otherwise keep original version
                else:
                    # New entity from gleaning stage
                    maybe_nodes[entity_name] = list(glean_entities)

            for edge_key, glean_edges in glean_edges.items():
                if edge_key in maybe_edges:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_edges[edge_key][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_edges[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_edges[edge_key] = list(glean_edges)
                    # Otherwise keep original version
                else:
                    # New edge from gleaning stage
                    maybe_edges[edge_key] = list(glean_edges)

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                return await _process_single_content(chunk)
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None
    chunk_results = []

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                chunk_results.append(task.result())
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Add progress prefix to the exception message
        progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

        # Re-raise the original exception with a prefix
        prefixed_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise prefixed_exception from first_exception

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
    entity_chunks_db: BaseKVStorage = None,
    domain: DomainConfig | None = None,
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Based on different query_param settings, different fields will be populated:
        - only_need_context=True: content contains context string
        - only_need_prompt=True: content contains complete prompt
        - stream=True: response_iterator contains streaming response, raw_data contains complete data
        - default: content contains LLM response text, raw_data contains complete data

        Returns None when no relevant context could be constructed for the query.
    """
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv, domain
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build query context (unified interface)
    context_result = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
        entity_chunks_db,
        domain,
    )

    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    # Return different content based on query parameters
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_result.context, raw_data=context_result.raw_data
        )

    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else get_prompt("rag_response", domain)
    logger.info(f"[DEBUG] Domain: {domain.name if domain else 'None'}, Using custom prompt: {domain.rag_response is not None if domain else False}")
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context_result.context,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=context_result.raw_data)

    # Call LLM
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        hl_keywords_str,
        ll_keywords_str,
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )

    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )

    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "hl_keywords": hl_keywords_str,
                "ll_keywords": ll_keywords_str,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response,
            raw_data=context_result.raw_data,
            is_streaming=True,
        )


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    domain: DomainConfig | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results
        domain: Optional domain configuration for prompt overrides

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv, domain
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    domain: DomainConfig | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    
    Args:
        text: The query text to extract keywords from
        param: Query parameters
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results
        domain: Optional domain configuration for prompt overrides
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(
        param.mode,
        text,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        try:
            keywords_data = json_repair.loads(cached_response)
            return keywords_data.get("high_level_keywords", []), keywords_data.get(
                "low_level_keywords", []
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    # 3. Build the keyword-extraction prompt (use domain-specific prompt if available)
    keywords_prompt_template = get_prompt("keywords_extraction", domain)
    kw_prompt = keywords_prompt_template.format(
        query=text,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # DEBUG: Log conversation history
    if param.conversation_history:
        logger.info(f"[DEBUG] Conversation history has {len(param.conversation_history)} messages")
        for i, msg in enumerate(param.conversation_history[-4:]):  # Show last 4 messages
            logger.info(f"[DEBUG] History[{i}]: role={msg.get('role')}, content={msg.get('content', '')[:100]}...")
    else:
        logger.info("[DEBUG] Conversation history is EMPTY")

    # 4. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(
        kw_prompt,
        keyword_extraction=True,
        history_messages=param.conversation_history,
    )

    # 5. Parse out JSON from the LLM response
    logger.info(f"[DEBUG] Raw LLM result for keywords: {repr(result)}")
    result = remove_think_tags(result)
    logger.info(f"[DEBUG] After remove_think_tags: {repr(result)}")
    try:
        keywords_data = json_repair.loads(result)
        logger.info(f"[DEBUG] Parsed keywords_data: {keywords_data}")
        if not keywords_data:
            logger.error("No JSON-like structure found in the LLM respond.")
            return [], []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"LLM respond: {result}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 6. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    try:
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k
        cosine_threshold = chunks_vdb.cosine_better_than_threshold

        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding
        )
        if not results:
            logger.info(
                f"Naive query: 0 chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
            )
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                # Try multiple keys for chunk_id (Qdrant uses __id__, others might use id)
                chunk_id = result.get("__id__") or result.get("id") or result.get("chunk_id")
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                    "chunk_id": chunk_id,  # Add chunk_id for deduplication
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
        )
        # DEBUG: Log top 5 vector chunks
        if valid_chunks:
            top5_info = [((c.get("chunk_id") or "")[:30], (c.get("content") or "")[:60]) for c in valid_chunks[:5]]
            logger.info(f"DEBUG: Top 5 vector chunks: {top5_info}")
        return valid_chunks

    except Exception as e:
        import traceback
        logger.error(f"Error in _get_vector_context: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


async def _perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> dict[str, Any]:
    """
    Pure search logic that retrieves raw entities, relations, and vector chunks.
    No token truncation or formatting - just raw search results.
    """

    # Initialize result containers
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []
    vector_chunks = []
    chunk_tracking = {}

    # Handle different query modes

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute query embedding once for all vector operations
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    query_embedding = None
    if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
        # Use chunks_vdb embedding_func for consistency with vector search
        actual_embedding_func = chunks_vdb.embedding_func if chunks_vdb else text_chunks_db.embedding_func
        logger.info(f"DEBUG: embedding_func from chunks_vdb={chunks_vdb is not None}, func={actual_embedding_func is not None}")
        if actual_embedding_func:
            try:
                embedding_result = await actual_embedding_func([query])
                # Check if result is valid (handle numpy array case)
                result_len = len(embedding_result) if embedding_result is not None else 0
                logger.info(f"DEBUG: Raw embedding result type={type(embedding_result).__name__}, len={result_len}")
                if result_len > 0:
                    query_embedding = embedding_result[0]
                    # Convert numpy array to list if needed
                    if hasattr(query_embedding, 'tolist'):
                        query_embedding = query_embedding.tolist()
                    logger.info(f"DEBUG: Pre-computed embedding len={len(query_embedding)}")
                else:
                    logger.warning("DEBUG: Embedding result is None or empty")
                    query_embedding = None
            except Exception as e:
                logger.warning(f"Failed to pre-compute query embedding: {e}")
                query_embedding = None

    # Handle local and global modes
    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )

    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

    else:  # hybrid or mix mode
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                query_param,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
            )

        # Get vector chunks for mix mode
        if query_param.mode == "mix" and chunks_vdb:
            has_embedding = query_embedding is not None and (hasattr(query_embedding, '__len__') and len(query_embedding) > 0)
            logger.info(f"DEBUG: Mix mode calling _get_vector_context, query_embedding is {'set' if has_embedding else 'None'}")
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
                query_embedding,
            )
            logger.info(f"DEBUG: Mix mode got {len(vector_chunks)} vector chunks")
            # Track vector chunks with source metadata
            for i, chunk in enumerate(vector_chunks):
                # Qdrant stores chunk ID in __id__ field within payload
                chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,  # Vector chunks always have frequency 1
                        "order": i + 1,  # 1-based order in vector search results
                    }
                else:
                    logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Round-robin merge entities
    final_entities = []
    seen_entities = {}  # entity_name -> rank (to compare and keep higher-ranked version)
    max_len = max(len(local_entities), len(global_entities))
    
    # Debug: Check Điều 26 in local vs global before merge
    local_dieu26 = [(i, e.get("entity_name"), e.get("rank", 0), e.get("is_direct_link", False)) for i, e in enumerate(local_entities) if "điều 26 - luật doanh nghiệp 2020" in e.get("entity_name", "").lower()]
    global_dieu26 = [(i, e.get("entity_name"), e.get("rank", 0), e.get("is_direct_link", False)) for i, e in enumerate(global_entities) if "điều 26 - luật doanh nghiệp 2020" in e.get("entity_name", "").lower()]
    if local_dieu26 or global_dieu26:
        logger.info(f"DEBUG: Điều 26 before merge - LOCAL: {local_dieu26}, GLOBAL: {global_dieu26}")
    
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name:
                existing_rank = seen_entities.get(entity_name, -1)
                current_rank = entity.get("rank", 0)
                # Keep entity with higher rank (or add if not seen)
                if existing_rank < 0:
                    final_entities.append(entity)
                    seen_entities[entity_name] = current_rank
                elif current_rank > existing_rank:
                    # Replace with higher-ranked version
                    for idx, e in enumerate(final_entities):
                        if e.get("entity_name") == entity_name:
                            final_entities[idx] = entity
                            seen_entities[entity_name] = current_rank
                            break

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name:
                existing_rank = seen_entities.get(entity_name, -1)
                current_rank = entity.get("rank", 0)
                # Keep entity with higher rank (or add if not seen)
                if existing_rank < 0:
                    final_entities.append(entity)
                    seen_entities[entity_name] = current_rank
                elif current_rank > existing_rank:
                    # Replace with higher-ranked version
                    for idx, e in enumerate(final_entities):
                        if e.get("entity_name") == entity_name:
                            final_entities[idx] = entity
                            seen_entities[entity_name] = current_rank
                            break


    # Round-robin merge relations
    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, {len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, str],
    ll_keywords: str = "",  # Low-level keywords to boost matching entities
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.
    
    Entities whose names closely match ll_keywords get priority boost to survive truncation.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Debug: Check for duplicate Điều 26 entities
    dieu26_all = [(i, e.get("entity_name"), e.get("rank", 0), e.get("is_direct_link", False)) for i, e in enumerate(final_entities) if "điều 26" in e.get("entity_name", "").lower() and "luật doanh nghiệp" in e.get("entity_name", "").lower()]
    if len(dieu26_all) > 1:
        logger.info(f"DEBUG: Found {len(dieu26_all)} Điều 26 instances in final_entities (DUPLICATE): {dieu26_all}")
    elif dieu26_all:
        logger.info(f"DEBUG: Found 1 Điều 26 in final_entities: {dieu26_all}")

    # Boost rank for entities whose names match the query keywords
    # This ensures entities directly relevant to the query survive truncation
    
    def normalize_for_match(s: str) -> str:
        """Normalize string for matching by removing separators and extra spaces."""
        import re
        # Remove common separators: " - ", " – ", " — " and normalize spaces
        s = re.sub(r'\s*[-–—]\s*', ' ', s.lower())
        # Collapse multiple spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    if ll_keywords:
        keywords_list = [kw.strip().lower() for kw in ll_keywords.split(",") if kw.strip()]
        keywords_normalized = [normalize_for_match(kw) for kw in keywords_list]
        boosted_count = 0
        for entity in final_entities:
            entity_name = entity.get("entity_name", "")
            entity_name_normalized = normalize_for_match(entity_name)
            # Check if entity name closely matches any keyword (after normalization)
            for kw, kw_norm in zip(keywords_list, keywords_normalized):
                # Match if normalized keyword is contained in normalized entity name or vice versa
                if kw_norm in entity_name_normalized or entity_name_normalized in kw_norm:
                    # Boost rank significantly (higher than supplementary boost of 1000)
                    entity["_query_match_boost"] = True
                    entity["rank"] = entity.get("rank", 0) + 5000  # Highest priority
                    boosted_count += 1
                    break
        if boosted_count > 0:
            logger.info(f"Boosted {boosted_count} entities matching query keywords: {keywords_list[:3]}")

    # Sort entities by rank (descending) to prioritize high-rank entities during truncation
    # Priority order: query-matched > supplementary > regular (by rank)
    final_entities = sorted(
        final_entities,
        key=lambda x: (
            x.get("_query_match_boost", False),  # Query-matched entities first
            x.get("is_supplementary", False),  # Then supplementary entities
            x.get("rank", 0),  # Then by rank
        ),
        reverse=True,
    )
    
    # Log top entities for debugging
    supplementary_count = sum(1 for e in final_entities if e.get("is_supplementary"))
    top_entities_info = [
        (e.get("entity_name", "")[:40], e.get("rank", 0), e.get("is_supplementary", False))
        for e in final_entities[:10]
    ]
    logger.info(
        f"Sorted entities: {len(final_entities)} total, {supplementary_count} supplementary. "
        f"Top 10: {top_entities_info}"
    )
    
    # Debug: Check for specific entities (Khoản 10 - Điều 23)
    khoan_10_entities = [
        (i, e.get("entity_name"), e.get("rank", 0), e.get("is_supplementary", False))
        for i, e in enumerate(final_entities)
        if "khoản 10" in e.get("entity_name", "").lower() and "điều 23" in e.get("entity_name", "").lower()
    ]
    if khoan_10_entities:
        logger.info(f"DEBUG Khoản 10 - Điều 23 position after sort: {khoan_10_entities}")
    
    # Debug: Check for Điều 26 - Luật Doanh nghiệp 2020
    dieu_26_entities = [
        (i, e.get("entity_name"), e.get("rank", 0), e.get("is_supplementary", False), e.get("is_direct_link", False))
        for i, e in enumerate(final_entities)
        if "điều 26" in e.get("entity_name", "").lower() and "luật doanh nghiệp" in e.get("entity_name", "").lower()
    ]
    if dieu_26_entities:
        logger.info(f"DEBUG Điều 26 position after sort: {dieu_26_entities}")


    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data
        entity_id_to_original[entity_name] = entity

        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation

        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context
    filtered_entities = []
    filtered_entity_id_to_original = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)
        
        # Debug: Check if Khoản 10 is in filtered entities
        khoan10_entities = [e for e in filtered_entities if "Khoản 10" in e.get("entity_name", "")]
        if khoan10_entities:
            logger.info(f"DEBUG: Khoản 10 in filtered_entities: {[(e.get('entity_name'), e.get('source_id'), e.get('is_supplementary'), e.get('is_direct_link')) for e in khoan10_entities]}")
        
        # Debug: Check if Điều 26 is in filtered entities
        dieu26_entities = [e for e in filtered_entities if "điều 26" in e.get("entity_name", "").lower() and "luật doanh nghiệp" in e.get("entity_name", "").lower()]
        if dieu26_entities:
            logger.info(f"DEBUG: Điều 26 in filtered_entities: YES - {[(e.get('entity_name'), e.get('rank')) for e in dieu26_entities]}")
        else:
            logger.info(f"DEBUG: Điều 26 in filtered_entities: NO - filtered out during truncation")


    filtered_relations = []
    filtered_relation_id_to_original = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }


def _parse_amendment_annotations(content: str) -> list[str]:
    """
    Parse amendment annotations from chunk content.
    Returns list of amendment entity names found in annotations like:
    [Điều này được bổ sung bởi Khoản 10 Điều 1 Luật Doanh nghiệp sửa đổi 2025]
    [Điều này được sửa đổi bởi Khoản 5a Điều 1 ...]
    """
    amendment_entities = []
    
    # Pattern to match amendment annotations
    # Matches: [Điều này được bổ sung/sửa đổi bởi Khoản X Điều Y Luật Z ...]
    patterns = [
        r'\[(?:Điều này|Khoản này|Điểm này) được (?:bổ sung|sửa đổi|thay thế) bởi (Khoản \d+\w*)\s+(Điều \d+)\s+([^\]]+?)\s*(?:có hiệu lực[^\]]*?)?\]',
        r'\[(?:Điều này|Khoản này|Điểm này) được (?:bổ sung|sửa đổi|thay thế) bởi (Khoản \d+\w*)\s+([^\]]+?)\s*(?:có hiệu lực[^\]]*?)?\]',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) >= 3:
                # Format: Khoản X - Điều Y - Luật Z
                khoan = groups[0].strip()
                dieu = groups[1].strip()
                luat = groups[2].strip()
                entity_name = f"{khoan} - {dieu} - {luat}"
            elif len(groups) >= 2:
                # Format: Khoản X - Luật Y
                khoan = groups[0].strip()
                luat = groups[1].strip()
                entity_name = f"{khoan} - {luat}"
            else:
                continue
            
            amendment_entities.append(entity_name)
    
    return amendment_entities


def _detect_cross_references(content: str) -> list[dict]:
    """
    Detect cross-references in chunk content.
    Looks for patterns like:
    - "quy định tại khoản X Điều Y"
    - "theo điều X"
    - "khoản X Điều Y của Luật này"
    - "được hướng dẫn bởi Điều X Nghị định Y"
    
    Returns list of detected references with article/clause info.
    """
    import re
    
    references = []
    
    # Pattern 0: "được hướng dẫn bởi Điều X Nghị định/Thông tư Y" (guidance reference)
    pattern0 = re.compile(
        r'(?:được hướng dẫn bởi|hướng dẫn bởi|hướng dẫn tại|quy định chi tiết tại|quy định chi tiết bởi)\s+'
        r'điều\s+(\d+[a-z]?)\s+'
        r'(nghị định|thông tư)\s+'
        r'(\d+[-/]\d+[-/][^\s\]]+)',
        re.IGNORECASE | re.UNICODE
    )
    
    # Find matches for pattern 0 (guidance references to decrees/circulars)
    for match in pattern0.finditer(content):
        dieu = match.group(1)
        doc_type = match.group(2)  # "nghị định" or "thông tư"
        doc_number = match.group(3)  # e.g., "47/2021/NĐ-CP"
        if dieu:
            references.append({
                "type": "guidance",
                "dieu": dieu,
                "doc_type": doc_type.lower(),
                "doc_number": doc_number,
                "full_match": match.group(0),
            })
    
    # Pattern 1: "quy định tại khoản X Điều Y" (most specific - clause + article)
    pattern1 = re.compile(
        r'(?:quy định tại|theo|tại|căn cứ|nêu tại|được quy định tại)\s+'
        r'khoản\s+(\d+[a-z]?)\s+'
        r'(?:điều\s+(\d+)|và khoản\s+\d+[a-z]?\s+điều\s+(\d+))',
        re.IGNORECASE | re.UNICODE
    )
    
    # Pattern 2: "quy định tại Điều X Luật/Nghị định Y" (article with optional law/decree name)
    pattern2 = re.compile(
        r'(?:quy định tại|theo|tại|căn cứ|nêu tại|được quy định tại)\s+'
        r'điều\s+(\d+[a-z]?)'
        r'(?:\s+(luật|nghị định|thông tư)\s+([^\s,;\.]+(?:\s+[^\s,;\.]+)?))?',
        re.IGNORECASE | re.UNICODE
    )
    
    # Pattern 2b: "Điều X của Luật/Nghị định này" (internal cross-reference within same document)
    pattern2b = re.compile(
        r'(?:quy định tại|theo|tại|căn cứ|nêu tại|được quy định tại)\s+'
        r'(?:khoản\s+(\d+[a-z]?)\s+)?'  # Optional Khoản - GROUP 1
        r'điều\s+(\d+[a-z]?)\s+'  # Điều - GROUP 2
        r'(?:của\s+)?(luật|nghị định|thông tư)\s+này',  # "của Luật này" - GROUP 3
        re.IGNORECASE | re.UNICODE
    )
    
    # Pattern 3: "khoản X Điều Y của Luật này" (without quy định prefix)
    pattern3 = re.compile(
        r'khoản\s+(\d+[a-z]?)\s+điều\s+(\d+)\s+(?:của\s+)?(?:luật|nghị định|thông tư)',
        re.IGNORECASE | re.UNICODE
    )
    
    # Pattern 3b: "Điều X, Điều Y,... Luật/Nghị định Z" (multiple articles in sequence)
    # Matches: "Điều 200, Điều 201 Luật Doanh nghiệp" or "Điều 5, Điều 6 và Điều 7 Nghị định 23/2022"
    pattern3b = re.compile(
        r'điều\s+(\d+)(?:\s*,\s*điều\s+(\d+))+(?:\s*(?:và|,)\s*điều\s+(\d+))?\s+'
        r'(luật|nghị định|thông tư)\s+([^\s,;\.]+(?:\s+[^\s,;\.]+)?)',
        re.IGNORECASE | re.UNICODE
    )
    
    # Find all matches for pattern 1 (most specific)
    for match in pattern1.finditer(content):
        khoan = match.group(1)
        dieu = match.group(2) or match.group(3)
        if dieu:
            references.append({
                "type": "khoan_dieu",
                "khoan": khoan,
                "dieu": dieu,
                "full_match": match.group(0),
            })
    
    # Find all matches for pattern 2b FIRST (internal "của Luật/Nghị định này")
    # This must run BEFORE pattern2 to capture internal refs with proper type
    for match in pattern2b.finditer(content):
        khoan = match.group(1)  # Optional
        dieu = match.group(2)
        doc_type = match.group(3)  # "luật", "nghị định", "thông tư"
        if dieu:
            # Check if this wasn't already captured
            already_captured = any(
                ref.get("dieu") == dieu and ref.get("khoan") == khoan
                for ref in references
            )
            if not already_captured:
                ref_data = {
                    "type": "internal_ref",  # Mark as internal reference
                    "dieu": dieu,
                    "doc_type": doc_type.lower(),
                    "full_match": match.group(0),
                }
                if khoan:
                    ref_data["khoan"] = khoan
                    ref_data["type"] = "khoan_dieu_internal"
                references.append(ref_data)
                logger.debug(f"DEBUG: Pattern2b internal ref: Điều {dieu}, Khoản {khoan}, doc_type={doc_type}")
    
    # Find all matches for pattern 2 (Điều with optional law/decree name)
    for match in pattern2.finditer(content):
        dieu = match.group(1)
        doc_type = match.group(2)  # "luật", "nghị định", "thông tư" or None
        doc_name = match.group(3)  # e.g., "Doanh nghiệp", "168/2024/NĐ-CP" or None
        if dieu:
            # Check if this Điều wasn't already captured with Khoản
            already_captured = any(
                ref["dieu"] == dieu for ref in references
            )
            if not already_captured:
                ref_data = {
                    "type": "dieu",
                    "dieu": dieu,
                    "full_match": match.group(0),
                }
                # Add law/decree info if captured
                if doc_type:
                    ref_data["doc_type"] = doc_type.lower()
                    ref_data["doc_name"] = doc_name.strip() if doc_name else None
                references.append(ref_data)
    
    # Find all matches for pattern 3
    for match in pattern3.finditer(content):
        khoan = match.group(1)
        dieu = match.group(2)
        if dieu:
            # Check if this wasn't already captured
            already_captured = any(
                ref.get("dieu") == dieu and ref.get("khoan") == khoan
                for ref in references
            )
            if not already_captured:
                references.append({
                    "type": "khoan_dieu",
                    "khoan": khoan,
                    "dieu": dieu,
                    "full_match": match.group(0),
                })
    
    # Find all matches for pattern 3b (multiple Điều in sequence)
    for match in pattern3b.finditer(content):
        full_text = match.group(0)
        doc_type = match.group(4)
        doc_name = match.group(5)
        logger.debug(f"DEBUG: Pattern3b matched: '{full_text}'")
        
        # Extract all Điều numbers from the matched text
        dieu_numbers = re.findall(r'điều\s+(\d+)', full_text, re.IGNORECASE)
        logger.debug(f"DEBUG: Pattern3b extracted Điều numbers: {dieu_numbers}")
        for dieu in dieu_numbers:
            already_captured = any(ref.get("dieu") == dieu for ref in references)
            if not already_captured:
                ref_data = {
                    "type": "dieu",
                    "dieu": dieu,
                    "full_match": full_text,
                }
                if doc_type:
                    ref_data["doc_type"] = doc_type.lower()
                    ref_data["doc_name"] = doc_name.strip() if doc_name else None
                references.append(ref_data)
                logger.debug(f"DEBUG: Pattern3b added ref: Điều {dieu} {doc_type} {doc_name}")
    
    # Pattern 4: "[...được sửa đổi bởi Khoản X Điều Y Luật Z...]" or "[...được sửa đổi bởi Điểm X Khoản Y Điều Z...]"
    # This detects amendment annotations in brackets
    pattern4 = re.compile(
        r'\[(?:[^\]]*?)'  # Start of bracket, any content
        r'(?:được sửa đổi|sửa đổi|bổ sung|thay thế|bãi bỏ)\s+bởi\s+'
        r'(?:điểm\s+([a-z])\s+)?'  # Optional "Điểm a/b/c" - GROUP 1
        r'(?:khoản\s+(\d+[a-z]?)\s+)?'  # Optional "Khoản X" - GROUP 2
        r'điều\s+(\d+[a-z]?)\s+'  # Required "Điều Y" - GROUP 3
        r'(luật|nghị định|thông tư)\s+'  # Required doc type - GROUP 4
        r'([^\]]+?)'  # Doc name until end of bracket - GROUP 5
        r'(?:\s+có hiệu lực[^\]]*)?'  # Optional "có hiệu lực..."
        r'\]',
        re.IGNORECASE | re.UNICODE
    )
    
    for match in pattern4.finditer(content):
        diem = match.group(1)  # Optional "Điểm a/b/c"
        khoan = match.group(2)  # Optional "Khoản X"
        dieu = match.group(3)  # Required "Điều Y"
        doc_type = match.group(4).lower()  # "luật", "nghị định", "thông tư"
        doc_name = match.group(5).strip()  # e.g., "08/2023/NĐ-CP"
        
        if dieu:
            # Check if this wasn't already captured (must match dieu, doc_type AND doc_name)
            already_captured = any(
                ref.get("dieu") == dieu and ref.get("doc_type") == doc_type and ref.get("doc_name") == doc_name
                for ref in references
            )
            if not already_captured:
                ref_data = {
                    "type": "amendment_ref",
                    "dieu": dieu,
                    "doc_type": doc_type,
                    "doc_name": doc_name,
                    "full_match": match.group(0),
                }
                if diem:
                    ref_data["diem"] = diem
                if khoan:
                    ref_data["khoan"] = khoan
                references.append(ref_data)
                logger.info(f"DEBUG: Detected amendment ref: Điểm {diem} Khoản {khoan} Điều {dieu} {doc_type} {doc_name[:30]}")
    
    return references


async def _resolve_cross_reference_chunks(
    cross_refs: list[dict],
    text_chunks_db: BaseKVStorage,
    seen_chunk_ids: set,
    source_file_path: str = None,  # For internal refs, restrict to same file
) -> list[dict]:
    """
    Resolve cross-references to actual chunks.
    For each reference, find chunks that contain the referenced article/clause content.
    
    OPTIMIZED: Uses batch loading instead of individual get_by_id calls.
    """
    if not cross_refs or not text_chunks_db:
        return []
    
    resolved_chunks = []
    
    # OPTIMIZATION: Limit number of cross-refs to resolve per call
    MAX_REFS_TO_RESOLVE = 10
    cross_refs = cross_refs[:MAX_REFS_TO_RESOLVE]
    
    # OPTIMIZATION: Use cached chunk data if available, otherwise batch load
    # Check if _data is available (in-memory storage)
    chunk_content_cache = {}
    try:
        if hasattr(text_chunks_db, '_data') and text_chunks_db._data:
            # Direct access to in-memory data - no DB calls needed!
            chunk_content_cache = text_chunks_db._data
            all_keys = list(chunk_content_cache.keys())
        else:
            # Fall back to getting keys and batch loading
            all_keys = []
            logger.warning("Cross-ref resolution: No in-memory cache available")
            return []
    except Exception as e:
        logger.warning(f"Cross-ref resolution error: {e}")
        return []
    
    # For amendment_ref, we need to search MORE chunks because the amendment content
    # might be in a different document (e.g., Nghị định 65/2022 amending NĐ 153/2020)
    # Check if any cross_ref is amendment_ref type
    has_amendment_refs = any(ref.get("type") == "amendment_ref" for ref in cross_refs)
    
    # Use higher limit for amendment refs, lower limit for regular cross-refs
    if has_amendment_refs:
        MAX_CHUNKS_TO_SEARCH = 2000  # Search more for amendments - they're critical!
    else:
        MAX_CHUNKS_TO_SEARCH = 500  # Regular cross-refs can use smaller search
    search_keys = all_keys[:MAX_CHUNKS_TO_SEARCH]
    
    for ref in cross_refs:
        dieu = ref.get("dieu")
        khoan = ref.get("khoan")
        ref_type = ref.get("type", "")
        
        # For guidance references, we need to search for specific decree/circular
        doc_type = ref.get("doc_type", "")  # "nghị định", "thông tư", "luật"
        doc_number = ref.get("doc_number", "")  # e.g., "47/2021/NĐ-CP" (for guidance type)
        doc_name = ref.get("doc_name", "")  # e.g., "Doanh nghiệp" (for dieu type)
        
        if ref_type == "guidance":
            logger.debug(f"Resolving guidance ref: Điều {dieu} {doc_type} {doc_number}")
        elif ref_type == "amendment_ref":
            logger.info(f"Resolving amendment_ref: Khoản {khoan} Điều {dieu} {doc_type} {doc_name[:50] if doc_name else ''}")
        else:
            logger.debug(f"Resolving cross-ref: Điều {dieu}, Khoản {khoan}")
        
        if not dieu:
            continue
        
        # Build search patterns based on reference type
        search_patterns = []
        
        if ref_type == "guidance":
            # For guidance references, look for the specific decree/circular article
            # Format: "Nghị định 47/2021/NĐ-CP. Điều 9. ..."
            doc_type_normalized = "Nghị định" if "nghị định" in doc_type else "Thông tư"
            # Normalize doc_number: 47/2021/NĐ-CP or 47-2021-NĐ-CP
            doc_number_variants = [
                doc_number,
                doc_number.replace("/", "-"),
                doc_number.replace("-", "/"),
            ]
            for variant in doc_number_variants:
                search_patterns.append(f"{doc_type_normalized} {variant}")
                # Also try with Điều
                search_patterns.append(f"Điều {dieu}")
        elif ref_type == "khoan_dieu":
            # Looking for specific Khoản within Điều
            search_patterns.append(f"Điều {dieu}.")
        else:
            # Looking for entire Điều
            search_patterns.append(f"Điều {dieu}.")
        
        # Search through chunks - OPTIMIZED: use limited search_keys and direct cache access
        found_chunk = False
        for chunk_key in search_keys:  # Use limited search_keys instead of all_keys
            # For amendment_ref, we need to find the chunk even if it's in seen_chunk_ids
            # because we need to inject it with is_amendment_content=True for priority
            skip_if_seen = True
            if ref_type == "amendment_ref":
                skip_if_seen = False  # Don't skip for amendments - we need to inject with priority
            
            if skip_if_seen and chunk_key in seen_chunk_ids:
                continue
            
            # OPTIMIZATION: Direct cache access - no await needed!
            chunk_data = chunk_content_cache.get(chunk_key)
            if not chunk_data:
                continue
            
            content = chunk_data.get("content", "")
            
            # Check if this is the MAIN chunk for the referenced Điều
            # (not just a chunk that references it)
            is_main_chunk = False
            
            if ref_type == "guidance":
                # For guidance references, check if chunk is from the specific decree/circular
                # AND contains the referenced Điều
                doc_patterns_match = False
                dieu_pattern_match = False
                
                for variant in doc_number_variants if 'doc_number_variants' in dir() else [doc_number]:
                    if variant.lower() in content.lower():
                        doc_patterns_match = True
                        break
                
                # Check if it contains the specific Điều as main content
                dieu_pattern = f"Điều {dieu}."
                if dieu_pattern in content:
                    pattern_idx = content.find(dieu_pattern)
                    # Điều should appear early in chunk for decree content
                    if pattern_idx < 200:
                        dieu_pattern_match = True
                
                if doc_patterns_match and dieu_pattern_match:
                    is_main_chunk = True
            elif ref_type == "dieu" and doc_type and doc_name:
                # For cross-references with law/decree name (e.g., "Điều 19 Luật Doanh nghiệp")
                doc_type_normalized = ""
                if "luật" in doc_type:
                    doc_type_normalized = "Luật"
                elif "nghị định" in doc_type:
                    doc_type_normalized = "Nghị định"
                elif "thông tư" in doc_type:
                    doc_type_normalized = "Thông tư"
                
                # Check if chunk contains the law/decree name
                doc_name_match = False
                content_lower = content.lower()
                doc_name_lower = doc_name.lower().strip()
                
                if doc_type_normalized.lower() in content_lower:
                    if doc_name_lower in content_lower:
                        doc_name_match = True
                    elif f"{doc_type_normalized.lower()} {doc_name_lower}" in content_lower:
                        doc_name_match = True
                
                # Check for the specific Điều as main content
                dieu_pattern = f"Điều {dieu}."
                if doc_name_match and dieu_pattern in content:
                    pattern_idx = content.find(dieu_pattern)
                    if pattern_idx < 200:
                        is_main_chunk = True
            elif ref_type in ("internal_ref", "khoan_dieu_internal") and doc_type:
                # For internal cross-references
                chunk_file_path = chunk_data.get("file_path", "")
                content_lower = content.lower()
                
                file_match = True
                if source_file_path:
                    source_name = source_file_path.lower().replace("_", " ").replace("-", " ")
                    chunk_name = chunk_file_path.lower().replace("_", " ").replace("-", " ")
                    file_match = (source_name == chunk_name) or (source_file_path == chunk_file_path)
                
                dieu_pattern = f"Điều {dieu}."
                if file_match and dieu_pattern in content:
                    pattern_idx = content.find(dieu_pattern)
                    if pattern_idx < 200:
                        is_main_chunk = True
            elif ref_type == "amendment_ref" and doc_type and doc_name:
                # For amendment references - CRITICAL: find the actual amendment content
                # doc_name might be: "65/2022/NĐ-CP", "Doanh nghiệp sửa đổi 2025", etc.
                content_lower = content.lower()
                doc_name_lower = doc_name.lower().strip()
                chunk_file_path = chunk_data.get("file_path", "").lower()
                
                doc_name_match = False
                
                # Strategy 1: Check if doc_name contains a decree/law number (e.g., "65/2022/NĐ-CP")
                # Extract just the number part for matching
                doc_number_match = re.search(r'(\d+)[/-](\d{4})[/-]?([A-Za-zĐđ-]+)?', doc_name)
                if doc_number_match:
                    doc_num = doc_number_match.group(1)  # e.g., "65"
                    doc_year = doc_number_match.group(2)  # e.g., "2022"
                    doc_suffix = doc_number_match.group(3) or ""  # e.g., "NĐ-CP"
                    
                    # Check file path first (most reliable)
                    if doc_num in chunk_file_path and doc_year in chunk_file_path:
                        doc_name_match = True
                        logger.debug(f"Amendment match by file path: {chunk_file_path[:40]}")
                    
                    # Check content for various formats
                    # Format 1: "Nghị định 65/2022/NĐ-CP"
                    if not doc_name_match:
                        full_doc_id = f"{doc_num}/{doc_year}"
                        if full_doc_id in content_lower:
                            doc_name_match = True
                            logger.debug(f"Amendment match by doc_id: {full_doc_id}")
                    
                    # Format 2: "Nghị định số 65/2022"
                    if not doc_name_match:
                        if f"số {doc_num}/{doc_year}" in content_lower or f"số {doc_num}-{doc_year}" in content_lower:
                            doc_name_match = True
                
                # Strategy 2: For law amendments like "Doanh nghiệp sửa đổi 2025"
                if not doc_name_match and "sửa đổi" in doc_name_lower:
                    year_match = re.search(r'(\d{4})', doc_name)
                    if year_match:
                        year = year_match.group(1)
                        # Check for "sửa đổi YYYY" or "sửa đổi năm YYYY"
                        if f"sửa đổi {year}" in content_lower or f"sửa đổi năm {year}" in content_lower:
                            doc_name_match = True
                        # Also check file path
                        if year in chunk_file_path and "sửa đổi" in chunk_file_path:
                            doc_name_match = True
                
                # Strategy 3: Direct doc_name substring match
                if not doc_name_match and doc_name_lower in content_lower:
                    doc_name_match = True
                
                if not doc_name_match:
                    continue
                
                # Check for the specific Điều (usually Điều 1 for amendments)
                dieu_pattern = f"Điều {dieu}."
                dieu_found = dieu_pattern in content or f"Điều {dieu}\n" in content
                
                # Check for the specific Khoản if provided
                khoan_found = True
                if khoan:
                    khoan_patterns = [
                        f"Khoản {khoan}.",
                        f"Khoản {khoan} ",
                        f"Khoản {khoan}\n",
                        f"\n{khoan}.",
                        f"\n{khoan} ",  # "6. Sửa đổi" format
                    ]
                    khoan_found = any(p in content for p in khoan_patterns)
                
                if doc_name_match and dieu_found and khoan_found:
                    is_main_chunk = True
                    logger.info(f"FOUND amendment content: Khoản {khoan} Điều {dieu} in {chunk_file_path[:30]}")
            else:
                # Original logic for other reference types
                for pattern in search_patterns:
                    if pattern in content:
                        pattern_idx = content.find(pattern)
                        if pattern_idx < 200:
                            is_main_chunk = True
                            break
                        if pattern_idx > 0:
                            before_text = content[max(0, pattern_idx-50):pattern_idx]
                            if "quy định tại" not in before_text.lower():
                                is_main_chunk = True
                                break
            
            if is_main_chunk:
                if ref_type == "khoan_dieu" and khoan:
                    khoan_pattern = f"{khoan}."
                    if khoan_pattern not in content and f"Khoản {khoan}" not in content:
                        continue
                
                chunk_entry = {
                    "content": content,
                    "file_path": chunk_data.get("file_path", "unknown"),
                    "chunk_id": chunk_key,
                    "is_priority": True,
                    "is_cross_reference": True,
                    "source": "cross_reference_resolution",
                    "referenced_from": ref.get("full_match", ""),
                }
                
                if ref_type == "amendment_ref":
                    chunk_entry["is_amendment_content"] = True
                    chunk_entry["source"] = "amendment_ref_resolution"
                
                resolved_chunks.append(chunk_entry)
                seen_chunk_ids.add(chunk_key)
                logger.debug(f"Resolved cross-reference to chunk {chunk_key[:20]}")
                break  # Found main chunk for this reference
    
    return resolved_chunks


async def _build_amendment_chunk_map(
    all_chunks: list[dict],
    filtered_entities: list[dict],
    text_chunks_db: BaseKVStorage = None,
    entity_chunks_db: BaseKVStorage = None,
    all_source_chunks: list[dict] = None,
) -> dict[str, list[dict]]:
    """
    Build a map from main chunk_id to list of amendment chunks.
    For each chunk with amendment annotations, find the chunks of the referenced entities.
    """
    chunk_to_amendments = {}
    
    if not text_chunks_db:
        return chunk_to_amendments
    
    # Build entity name to chunks map - prefer from entity_chunks_db for complete chunk_ids
    entity_to_chunks = {}
    
    # First, try to get all chunk_ids from entity_chunks_db
    if entity_chunks_db:
        for entity in filtered_entities:
            entity_name = entity.get("entity_name", "")
            if entity_name:
                try:
                    entity_chunk_data = await entity_chunks_db.get_by_id(entity_name)
                    if entity_chunk_data and "chunk_ids" in entity_chunk_data:
                        entity_to_chunks[entity_name] = entity_chunk_data["chunk_ids"]
                except Exception:
                    pass
    
    # Fallback: use source_id from filtered_entities if entity_chunks_db not available
    if not entity_to_chunks:
        for entity in filtered_entities:
            entity_name = entity.get("entity_name", "")
            source_id = entity.get("source_id", "")
            if entity_name and source_id:
                if entity_name not in entity_to_chunks:
                    entity_to_chunks[entity_name] = []
                entity_to_chunks[entity_name].append(source_id)
    
    # Debug: Log entity_to_chunks for Khoản 10
    khoan10_entities = {k: v for k, v in entity_to_chunks.items() if "khoản 10" in k.lower()}
    if khoan10_entities:
        logger.info(f"DEBUG: entity_to_chunks for Khoản 10: {khoan10_entities}")
    
    # For each chunk, parse annotations and find amendment chunks
    for chunk in all_chunks:
        # Qdrant stores chunk ID in __id__ field within payload
        chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
        content = chunk.get("content", "")
        
        # Debug: Check if this is Điều 23 chunk
        if chunk_id and "f39190da" in chunk_id:
            logger.info(f"DEBUG: Processing Điều 23 chunk {chunk_id[:20]}, content[:100]={content[:100] if content else 'EMPTY'}")
        
        if not chunk_id or not content:
            continue
        
        # Parse amendment annotations
        amendment_entity_names = _parse_amendment_annotations(content)
        
        if amendment_entity_names:
            logger.info(f"DEBUG: Chunk {chunk_id[:20]} has amendment annotations: {amendment_entity_names}")
        
        for entity_name in amendment_entity_names:
            # Extract "Khoản X" from entity name for precise matching
            khoan_match = re.search(r'Khoản\s+(\d+[a-z]?)', entity_name, re.IGNORECASE)
            dieu_match = re.search(r'Điều\s+(\d+)', entity_name, re.IGNORECASE)
            khoan_num = khoan_match.group(1) if khoan_match else None
            dieu_num = dieu_match.group(1) if dieu_match else None
            
            logger.info(f"DEBUG: Looking for amendment khoan={khoan_num}, dieu={dieu_num}")
            
            # Try to find matching entity with EXACT Khoản AND Điều number
            matched_chunks = []
            
            for ent_name, chunk_ids in entity_to_chunks.items():
                # Entity must have same Khoản AND Điều number if specified
                if khoan_num and dieu_num:
                    ent_khoan_match = re.search(r'Khoản\s+(\d+[a-z]?)', ent_name, re.IGNORECASE)
                    ent_dieu_match = re.search(r'Điều\s+(\d+)', ent_name, re.IGNORECASE)
                    if ent_khoan_match and ent_dieu_match:
                        ent_khoan_num = ent_khoan_match.group(1)
                        ent_dieu_num = ent_dieu_match.group(1)
                        if ent_khoan_num.lower() == khoan_num.lower() and ent_dieu_num == dieu_num:
                            matched_chunks.extend(chunk_ids)
                            logger.info(f"DEBUG: EXACT Khoản+Điều match entity '{ent_name}' with chunks {chunk_ids[:3]}")
            
            # If no match in entity_to_chunks, try direct lookup from entity_chunks_db
            if not matched_chunks and entity_chunks_db and khoan_num:
                # Build possible entity names based on parsed annotation
                possible_names = [
                    entity_name,  # Original parsed name: "Khoản 10 - Điều 1 - Luật Doanh nghiệp sửa đổi 2025"
                ]
                
                # Extract law name and build alternative formats
                law_match = re.search(r'(Luật\s+[\w\s]+(?:sửa đổi\s+)?\d{4})', entity_name, re.IGNORECASE)
                if law_match and dieu_num:
                    law_name = law_match.group(1)
                    possible_names.extend([
                        f"Khoản {khoan_num} - Điều {dieu_num} - {law_name}",
                        f"Khoản {khoan_num} Điều {dieu_num} {law_name}",
                        f"Khoản {khoan_num} - Điều {dieu_num} {law_name}",
                        f"Khoản {khoan_num} Điều {dieu_num} - {law_name}",
                    ])
                
                for possible_name in possible_names:
                    try:
                        entity_chunk_data = await entity_chunks_db.get_by_id(possible_name)
                        if entity_chunk_data and "chunk_ids" in entity_chunk_data:
                            matched_chunks.extend(entity_chunk_data["chunk_ids"])
                            logger.info(f"DEBUG: Direct lookup found entity '{possible_name}' with chunks {entity_chunk_data['chunk_ids'][:3]}")
                            break  # Found, no need to try other names
                    except Exception:
                        pass
            
            # If still no match, try content-based search
            if not matched_chunks and khoan_num and dieu_num and all_source_chunks:
                logger.info(f"DEBUG: No entity match, searching by content for Khoản {khoan_num} Điều {dieu_num}")
                # Search through all source chunks for amendment content
                for src_chunk in all_source_chunks:
                    src_chunk_id = src_chunk.get("chunk_id", "")
                    src_content = src_chunk.get("content", "")
                    
                    # Skip self
                    if src_chunk_id == chunk_id:
                        continue
                    
                    # Pattern: "Khoản X. Bổ sung khoản X ... Điều Y"
                    pattern1 = rf'Khoản\s+{khoan_num}[.:]?\s+Bổ sung.*?Điều\s+{dieu_num}'
                    pattern2 = rf'Điều\s+1[.:]?.*?Khoản\s+{khoan_num}[.:]?\s+Bổ sung.*?Điều\s+{dieu_num}'
                    
                    if re.search(pattern1, src_content, re.IGNORECASE | re.DOTALL) or re.search(pattern2, src_content, re.IGNORECASE | re.DOTALL):
                        matched_chunks.append(src_chunk_id)
                        logger.info(f"DEBUG: Content match found in chunk {src_chunk_id[:20]}")
                        break  # Take first match
            
            if matched_chunks:
                # Get chunk contents - now filter to only use chunks containing actual amendment content
                for amendment_chunk_id in matched_chunks:
                    # Skip if amendment chunk is the same as main chunk
                    if amendment_chunk_id == chunk_id:
                        continue
                    
                    try:
                        chunk_data = await text_chunks_db.get_by_id(amendment_chunk_id)
                        if chunk_data:
                            amendment_content = chunk_data.get("content", "")
                            
                            # Verify this chunk actually contains the amendment content (Khoản X)
                            if khoan_num:
                                verify_pattern = rf'Khoản\s+{khoan_num}[.:\s]'
                                if not re.search(verify_pattern, amendment_content, re.IGNORECASE):
                                    logger.info(f"DEBUG: Skipping chunk {amendment_chunk_id[:20]} - doesn't contain Khoản {khoan_num}")
                                    continue
                            
                            if chunk_id not in chunk_to_amendments:
                                chunk_to_amendments[chunk_id] = []
                            
                            # Avoid duplicates
                            existing_ids = [c.get("chunk_id") for c in chunk_to_amendments[chunk_id]]
                            if amendment_chunk_id not in existing_ids:
                                chunk_to_amendments[chunk_id].append({
                                    "chunk_id": amendment_chunk_id,
                                    "content": amendment_content,
                                    "file_path": chunk_data.get("file_path", "unknown_source"),
                                    "is_amendment": True,
                                })
                                logger.info(f"DEBUG: Mapped amendment chunk {amendment_chunk_id[:20]} to main chunk {chunk_id[:20]}")
                    except Exception as e:
                        logger.debug(f"Failed to get amendment chunk {amendment_chunk_id}: {e}")
    
    return chunk_to_amendments


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage = None,
    text_chunks_db: BaseKVStorage = None,
    query_param: QueryParam = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding: list[float] = None,
    entity_chunks_db: BaseKVStorage = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # Get chunks from entities
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Get chunks from relations
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Build amendment chunk map - maps main chunk_id to amendment chunks
    all_source_chunks = vector_chunks + entity_chunks + relation_chunks
    amendment_chunk_map = await _build_amendment_chunk_map(
        all_source_chunks, filtered_entities, text_chunks_db, entity_chunks_db, all_source_chunks
    )
    if amendment_chunk_map:
        logger.info(f"Built amendment chunk map with {len(amendment_chunk_map)} main chunks having amendments")

    # Helper function to inject amendment chunks after a main chunk
    def inject_amendment_chunks(main_chunk_id: str, merged_chunks: list, seen_chunk_ids: set) -> int:
        """Inject amendment chunks right after the main chunk. Returns count of added chunks."""
        added = 0
        if main_chunk_id in amendment_chunk_map:
            for amend_chunk in amendment_chunk_map[main_chunk_id]:
                amend_id = amend_chunk.get("chunk_id")
                if amend_id and amend_id not in seen_chunk_ids:
                    seen_chunk_ids.add(amend_id)
                    merged_chunks.append({
                        "content": amend_chunk["content"],
                        "file_path": amend_chunk.get("file_path", "unknown_source"),
                        "chunk_id": amend_id,
                        "is_priority": True,
                        "is_amendment": True,
                        "source": "amendment_injection",
                    })
                    added += 1
                    logger.info(f"DEBUG: Injected amendment chunk {amend_id[:20]} after main chunk {main_chunk_id[:20]}")
        return added

    # FIRST: Add TOP vector chunks as high priority (for mix mode - naive retrieval)
    # These are the most relevant chunks from pure vector similarity search
    merged_chunks = []
    seen_chunk_ids = set()
    priority_count = 0
    vector_priority_count = 0
    amendment_injection_count = 0
    cross_ref_injection_count = 0
    
    # Build a quick lookup for main chunk content from all source chunks
    main_chunk_content_map = {}
    for chunk in all_source_chunks:
        # Qdrant stores chunk ID in __id__ field within payload
        chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
        if chunk_id:
            main_chunk_content_map[chunk_id] = chunk
    
    # CRITICAL: Add top vector chunks FIRST (they are highest relevance to query)
    # This ensures Điều 93 etc. survive truncation even if many amendments exist
    TOP_VECTOR_PRIORITY = 10
    logger.info(f"DEBUG: _merge_all_chunks received {len(vector_chunks)} vector_chunks, processing top {TOP_VECTOR_PRIORITY}")
    for chunk in vector_chunks[:TOP_VECTOR_PRIORITY]:
        # Qdrant stores chunk ID in __id__ field within payload
        chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(
                {
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "chunk_id": chunk_id,
                    "is_priority": True,
                    "source": "vector_top",
                    "vector_rank": vector_priority_count,  # Track original rank
                }
            )
            vector_priority_count += 1
            logger.info(f"DEBUG: Added vector_top chunk {chunk_id[:30]} at position {len(merged_chunks)-1}")
            
            # Inject amendment chunks right after this main chunk
            amendment_injection_count += inject_amendment_chunks(chunk_id, merged_chunks, seen_chunk_ids)
            
            # Also check for cross-references in this vector chunk
            cross_refs = _detect_cross_references(chunk["content"])
            logger.info(f"DEBUG: Vector chunk {chunk_id[:20]} has {len(cross_refs)} cross-refs detected")
            if cross_refs:
                logger.info(f"DEBUG: Cross-refs found: {[r.get('full_match', '')[:40] for r in cross_refs[:5]]}")
                # Pass source file_path for internal refs
                source_file_path = chunk.get("file_path")
                resolved_chunks = await _resolve_cross_reference_chunks(
                    cross_refs, text_chunks_db, seen_chunk_ids, source_file_path
                )
                logger.info(f"DEBUG: Resolved {len(resolved_chunks)} chunks for cross-refs")
                # Insert cross-ref chunks RIGHT AFTER current position to ensure they survive truncation
                insert_position = len(merged_chunks)  # Current position after adding this chunk
                for i, resolved_chunk in enumerate(resolved_chunks):
                    merged_chunks.insert(insert_position + i, resolved_chunk)
                    cross_ref_injection_count += 1
                    logger.info(f"DEBUG: Inserted cross-ref chunk {resolved_chunk.get('chunk_id', '')[:20]} at position {insert_position + i}")
    
    if vector_priority_count > 0:
        logger.info(f"Added {vector_priority_count} top vector chunks as HIGH priority (with {amendment_injection_count} amendments, {cross_ref_injection_count} cross-refs)")
    
    # NEW: Also add chunks from HIGH-RANK entities (rank > 9000) to ensure they survive truncation
    # This ensures form templates (Mẫu số 2) with boosted entities get their chunks included early
    HIGH_RANK_THRESHOLD = 9000
    high_rank_entity_chunks = [c for c in entity_chunks if c.get("entity_rank", 0) > HIGH_RANK_THRESHOLD and c.get("is_priority")]
    # Sort by entity_rank descending
    high_rank_entity_chunks.sort(key=lambda c: c.get("entity_rank", 0), reverse=True)
    high_rank_count = 0
    for chunk in high_rank_entity_chunks[:15]:  # Limit to top 15 high-rank chunks
        chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk_id,
                "is_priority": True,
                "source": "high_rank_entity",
                "entity_rank": chunk.get("entity_rank", 0),
            })
            high_rank_count += 1
            logger.info(f"DEBUG: Added high-rank entity chunk {chunk_id[:30]} with rank={chunk.get('entity_rank', 0)}")
    if high_rank_count > 0:
        logger.info(f"Added {high_rank_count} chunks from high-rank entities (rank > {HIGH_RANK_THRESHOLD})")

    
    # DISABLED: Query title matching - causes linear scan of ALL chunks which is very slow
    # This was scanning entire text_chunks_db._data.keys() which causes O(n) performance
    # If needed, this should be implemented using vector search instead
    query_match_count = 0
    # Skipping query title matching for performance
    
    # NEW: Pre-inject ALL main chunks AND their amendment chunks as priority
    # This ensures BOTH original content and amendment content are included
    pre_injected_count = 0
    for main_chunk_id, amend_chunks in amendment_chunk_map.items():
        # First: inject the MAIN chunk (original content that is being amended)
        if main_chunk_id not in seen_chunk_ids and main_chunk_id in main_chunk_content_map:
            main_chunk = main_chunk_content_map[main_chunk_id]
            seen_chunk_ids.add(main_chunk_id)
            merged_chunks.append({
                "content": main_chunk.get("content", ""),
                "file_path": main_chunk.get("file_path", "unknown_source"),
                "chunk_id": main_chunk_id,
                "is_priority": True,
                "is_main_amended": True,
                "source": "main_chunk_priority",
            })
            pre_injected_count += 1
            logger.info(f"DEBUG: Pre-injected MAIN chunk {main_chunk_id[:20]}")
        
        # Then: inject all amendment chunks
        for amend_chunk in amend_chunks:
            amend_id = amend_chunk.get("chunk_id")
            if amend_id and amend_id not in seen_chunk_ids:
                seen_chunk_ids.add(amend_id)
                merged_chunks.append({
                    "content": amend_chunk["content"],
                    "file_path": amend_chunk.get("file_path", "unknown_source"),
                    "chunk_id": amend_id,
                    "is_priority": True,
                    "is_amendment": True,
                    "source": "amendment_priority",
                })
                pre_injected_count += 1
                logger.info(f"DEBUG: Pre-injected amendment chunk {amend_id[:20]} for main chunk {main_chunk_id[:20]}")
    
    if pre_injected_count > 0:
        logger.info(f"Pre-injected {pre_injected_count} main+amendment chunks as top priority")
    
    # Cross-reference resolution with LIMITS to prevent performance explosion
    # Only process first N chunks and limit total cross-refs to avoid O(n^2) behavior
    import time as _time
    _cross_ref_start = _time.time()
    cross_ref_injection_count = 0
    MAX_CHUNKS_FOR_CROSSREF = 20  # Only check first N chunks for cross-refs
    MAX_TOTAL_CROSSREFS = 30  # Cap total cross-refs to inject
    
    for chunk in merged_chunks[:MAX_CHUNKS_FOR_CROSSREF]:  # LIMIT: Only first N chunks
        if cross_ref_injection_count >= MAX_TOTAL_CROSSREFS:
            logger.info(f"Reached cross-ref limit ({MAX_TOTAL_CROSSREFS}), stopping")
            break
        content = chunk.get("content", "")
        cross_refs = _detect_cross_references(content)
        if cross_refs:
            # Limit cross-refs per chunk too
            cross_refs = cross_refs[:5]  # Max 5 cross-refs per chunk
            logger.debug(f"Found {len(cross_refs)} cross-references in chunk {(chunk.get('chunk_id') or '')[:20]}")
            source_file_path = chunk.get("file_path")
            resolved_chunks = await _resolve_cross_reference_chunks(
                cross_refs, text_chunks_db, seen_chunk_ids, source_file_path
            )
            for resolved_chunk in resolved_chunks:
                if cross_ref_injection_count >= MAX_TOTAL_CROSSREFS:
                    break
                merged_chunks.append(resolved_chunk)
                cross_ref_injection_count += 1
    
    _cross_ref_elapsed = _time.time() - _cross_ref_start
    if cross_ref_injection_count > 0:
        logger.info(f"Cross-ref resolution: {cross_ref_injection_count} chunks in {_cross_ref_elapsed:.2f}s")
    
    # DISABLED: Nested cross-reference resolution - causes exponential processing time
    # Multi-hop cross-refs were scanning ALL cross-ref chunks and recursively resolving
    # This was O(n^2) or worse behavior that slowed queries significantly
    nested_cross_ref_count = 0
    # Skipping nested cross-reference resolution for performance
    
    # Amendment ref processing with LIMITS
    # E.g., "[Điểm này được sửa đổi bởi Khoản 6 Điều 1 Luật DN sửa đổi 2025]"
    _amend_start = _time.time()
    amendment_ref_injection_count = 0
    MAX_CHUNKS_FOR_AMENDMENT = 15  # Only check first N chunks
    MAX_AMENDMENT_REFS = 20  # Cap total amendments
    
    for chunk in merged_chunks[:MAX_CHUNKS_FOR_AMENDMENT]:
        if amendment_ref_injection_count >= MAX_AMENDMENT_REFS:
            break
        # Skip chunks that are already amendment content to avoid infinite loops
        if chunk.get("is_amendment_content"):
            continue
        
        content = chunk.get("content", "")
        parent_chunk_id = chunk.get("chunk_id", "")
        
        # Detect amendment_ref patterns in this chunk
        cross_refs = _detect_cross_references(content)
        amendment_refs = [r for r in cross_refs if r.get("type") == "amendment_ref"]
        
        if amendment_refs:
            # Limit amendment refs per chunk
            amendment_refs = amendment_refs[:3]  # Max 3 per chunk
            logger.debug(f"Found {len(amendment_refs)} amendment_ref in chunk {parent_chunk_id[:20] if parent_chunk_id else ''}")
            chunk_file_path = chunk.get("file_path", "")
            resolved_chunks = await _resolve_cross_reference_chunks(
                amendment_refs, text_chunks_db, seen_chunk_ids, chunk_file_path
            )
            for resolved_chunk in resolved_chunks:
                if amendment_ref_injection_count >= MAX_AMENDMENT_REFS:
                    break
                resolved_chunk["is_amendment_content"] = True
                resolved_chunk["amendment_from"] = parent_chunk_id
                # Just append, don't insert (insert is O(n) for each operation)
                merged_chunks.append(resolved_chunk)
                amendment_ref_injection_count += 1
    
    _amend_elapsed = _time.time() - _amend_start
    if amendment_ref_injection_count > 0:
        logger.info(f"Amendment ref resolution: {amendment_ref_injection_count} chunks in {_amend_elapsed:.2f}s")
    
    # THEN: Add priority chunks from entity_chunks (from supplementary entities)
    # Sort priority chunks by entity_rank (highest first) to ensure high-rank entities' chunks come first
    # Add secondary sort by: 1) has relation annotations [], 2) query relevance
    priority_entity_chunks = [c for c in entity_chunks if c.get("is_priority")]
    
    # Extract keywords from query for relevance scoring
    query_keywords = set(query.lower().split()) if query else set()
    
    def sort_key(chunk):
        entity_rank = chunk.get("entity_rank", 0)
        content = chunk.get("content", "").lower()
        
        # Priority 1: Check if chunk has relation annotations [] - these are more important
        # Chunks with [] contain references to other legal clauses
        has_relation_annotation = 1 if re.search(r'\[[^\]]+\]', content) else 0
        
        # Priority 1b: Check if chunk IS an amendment/supplement content
        # These chunks contain the actual new content added by amendments
        amendment_patterns = [
            r'bổ sung khoản',
            r'sửa đổi khoản', 
            r'thay thế khoản',
            r'bổ sung điểm',
            r'sửa đổi điểm',
            r'bổ sung điều',
            r'sửa đổi điều',
        ]
        is_amendment_content = 1 if any(re.search(p, content) for p in amendment_patterns) else 0
        
        # Combine: either has bracket annotation OR is amendment content
        has_priority_marker = max(has_relation_annotation, is_amendment_content)
        
        # Priority 2: Calculate keyword match score
        keyword_match = sum(1 for kw in query_keywords if kw in content and len(kw) > 2)
        
        # Return tuple: (entity_rank, has_priority_marker, keyword_match) - all descending
        return (entity_rank, has_priority_marker, keyword_match)
    
    priority_entity_chunks.sort(key=sort_key, reverse=True)
    
    # Debug: Check if Khoản 5a chunk is in priority_entity_chunks
    khoan5a_in_priority = [c for c in priority_entity_chunks if "ffacaa2e" in str(c.get("chunk_id", "") or c.get("id", ""))]
    if khoan5a_in_priority:
        chunk = khoan5a_in_priority[0]
        content = chunk.get("content", "")
        has_bracket = 1 if re.search(r'\[[^\]]+\]', content) else 0
        logger.info(f"DEBUG: Khoản 5a chunk ffacaa2e in priority_entity_chunks, entity_rank={chunk.get('entity_rank', 0)}, has_bracket={has_bracket}")
    else:
        logger.info(f"DEBUG: Khoản 5a chunk ffacaa2e NOT in priority_entity_chunks")
    
    # Debug: Show top 20 priority chunks with entity_rank
    for i, chunk in enumerate(priority_entity_chunks[:20]):
        # Qdrant stores chunk ID in __id__ field within payload
        chunk_id = chunk.get("__id__") or chunk.get("chunk_id", "") or chunk.get("id", "")
        entity_rank = chunk.get("entity_rank", 0)
        content = chunk.get("content", "")
        has_bracket = 1 if re.search(r'\[[^\]]+\]', content) else 0
        logger.info(f"DEBUG: Priority chunk #{i+1}: rank={entity_rank}, bracket={has_bracket}, id={chunk_id[:30]}...")
    
    for chunk in priority_entity_chunks:
        # Qdrant stores chunk ID in __id__ field within payload
        chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(
                {
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "chunk_id": chunk_id,
                    "is_priority": True,
                    "entity_rank": chunk.get("entity_rank", 0),
                }
            )
            priority_count += 1
            
            # Inject amendment chunks right after this priority chunk
            amendment_injection_count += inject_amendment_chunks(chunk_id, merged_chunks, seen_chunk_ids)
            
            # Debug: Check if this is Khoản 10 chunk
            if "f5c8b3c5" in chunk_id:
                logger.info(f"DEBUG: Added Khoản 10 chunk f5c8b3c5 to merged_chunks at position {len(merged_chunks)}, entity_rank={chunk.get('entity_rank', 0)}, content[:100]={chunk.get('content', '')[:100]}")
            # Debug: Check if this is Khoản 5a chunk
            if "ffacaa2e" in chunk_id:
                logger.info(f"DEBUG: Added Khoản 5a chunk ffacaa2e to merged_chunks at position {len(merged_chunks)}, entity_rank={chunk.get('entity_rank', 0)}, content[:100]={chunk.get('content', '')[:100]}")
            # Debug: Check if this is Phụ lục II D61.9 chunk
            if "94f1e306" in chunk_id:
                logger.info(f"DEBUG: Added Phụ lục II D61.9 chunk 94f1e306 to merged_chunks at position {len(merged_chunks)}")
    
    if priority_count > 0:
        logger.info(f"Added {priority_count} priority chunks from supplementary entities (sorted by entity_rank)")
    
    # THEN: Round-robin merge remaining chunks from different sources with deduplication
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            # Qdrant stores chunk ID in __id__ field within payload
            chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )
                # Inject amendment chunks after this chunk
                amendment_injection_count += inject_amendment_chunks(chunk_id, merged_chunks, seen_chunk_ids)

        # Add from entity chunks (Local mode) - skip priority ones (already added)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            if not chunk.get("is_priority"):  # Skip priority chunks (already added)
                # Qdrant stores chunk ID in __id__ field within payload
                chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_chunks.append(
                        {
                            "content": chunk["content"],
                            "file_path": chunk.get("file_path", "unknown_source"),
                            "chunk_id": chunk_id,
                        }
                    )
                    # Inject amendment chunks after this chunk
                    amendment_injection_count += inject_amendment_chunks(chunk_id, merged_chunks, seen_chunk_ids)

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            # Qdrant stores chunk ID in __id__ field within payload
            chunk_id = chunk.get("__id__") or chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )
                # Inject amendment chunks after this chunk
                amendment_injection_count += inject_amendment_chunks(chunk_id, merged_chunks, seen_chunk_ids)

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (priority={priority_count}, amendment_injections={amendment_injection_count}, deduplicated {origin_len - len(merged_chunks)})"
    )

    return merged_chunks


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunk_tracking: dict = None,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
    domain: DomainConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    tokenizer = global_config.get("tokenizer")
    logger.info(f"[DEBUG-CONTEXT] domain={domain.name if domain else 'None'}, has_rag_response={domain.rag_response is not None if domain else False}")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Get the system prompt template from domain config or global_config or PROMPTS
    sys_prompt_template = global_config.get(
        "system_prompt_template", get_prompt("rag_response", domain)
    )

    kg_context_template = PROMPTS["kg_query_context"]
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))

    # Calculate preliminary system prompt tokens
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

    # Calculate available tokens for text chunks
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )

    logger.info(
        f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )
    logger.info(f"DEBUG: Input merged_chunks count: {len(merged_chunks)}")

    # Apply token truncation to chunks using the dynamic limit
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
        source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )
    
    logger.info(f"DEBUG: Output truncated_chunks count: {len(truncated_chunks)}")
    
    # Check if cross-ref chunk d543571a (Điều 201) is in truncated
    dieu201_in_truncated = any('d543571a' in str(c.get('chunk_id', '')) for c in truncated_chunks)
    dieu201_in_merged = any('d543571a' in str(c.get('chunk_id', '')) for c in merged_chunks)
    if dieu201_in_merged:
        pos = next((i for i, c in enumerate(merged_chunks) if 'd543571a' in str(c.get('chunk_id', ''))), -1)
        logger.info(f"DEBUG: Điều 201 chunk d543571a in MERGED at position {pos}")
        if dieu201_in_truncated:
            logger.info(f"DEBUG: Điều 201 chunk d543571a IN truncated_chunks ✓")
        else:
            logger.info(f"DEBUG: Điều 201 chunk d543571a NOT in truncated_chunks ✗")
    
    # Check if cross-ref chunk e5b872f0 (Điều 200) is in truncated
    dieu200_in_truncated = any('e5b872f0' in str(c.get('chunk_id', '')) for c in truncated_chunks)
    dieu200_in_merged = any('e5b872f0' in str(c.get('chunk_id', '')) for c in merged_chunks)
    if dieu200_in_merged:
        pos200 = next((i for i, c in enumerate(merged_chunks) if 'e5b872f0' in str(c.get('chunk_id', ''))), -1)
        logger.info(f"DEBUG: Điều 200 chunk e5b872f0 in MERGED at position {pos200}")
        if dieu200_in_truncated:
            logger.info(f"DEBUG: Điều 200 chunk e5b872f0 IN truncated_chunks ✓")
        else:
            logger.info(f"DEBUG: Điều 200 chunk e5b872f0 NOT in truncated_chunks ✗")
    else:
        logger.info(f"DEBUG: Điều 200 chunk e5b872f0 NOT in merged_chunks ✗")
    
    # Check if Khoản 10 chunk is in truncated_chunks
    khoan10_chunks = [c for c in truncated_chunks if 'chủ sở hữu hưởng lợi' in c.get('content', '').lower() or 'beneficial owner' in c.get('content', '').lower()]
    if khoan10_chunks:
        logger.info(f"DEBUG: Khoản 10 chunks in truncated: {len(khoan10_chunks)}")
    else:
        # Check in merged_chunks
        khoan10_merged = [c for c in merged_chunks if 'chủ sở hữu hưởng lợi' in c.get('content', '').lower() or 'bổ sung khoản 10' in c.get('content', '').lower()]
        logger.info(f"DEBUG: Khoản 10 chunks in MERGED: {len(khoan10_merged)}, but NOT in truncated")
        if khoan10_merged:
            for c in khoan10_merged[:2]:
                logger.info(f"DEBUG: Khoản 10 merged chunk: {c.get('chunk_id', '?')[:20]}, priority={c.get('is_priority')}")
    
    # Check if Khoản 5a chunk is in truncated_chunks
    khoan5a_truncated = [c for c in truncated_chunks if 'ffacaa2e' in str(c.get('chunk_id', ''))]
    if khoan5a_truncated:
        logger.info(f"DEBUG: Khoản 5a chunk ffacaa2e IN truncated_chunks")
    else:
        khoan5a_merged = [c for c in merged_chunks if 'ffacaa2e' in str(c.get('chunk_id', ''))]
        if khoan5a_merged:
            pos = merged_chunks.index(khoan5a_merged[0]) if khoan5a_merged else -1
            logger.info(f"DEBUG: Khoản 5a chunk ffacaa2e in MERGED at position {pos}, but NOT in truncated")
        else:
            logger.info(f"DEBUG: Khoản 5a chunk ffacaa2e NOT in merged_chunks")

    # Check if Điều 93 chunk is in truncated_chunks
    dieu93_truncated = [c for c in truncated_chunks if '22f37374' in str(c.get('chunk_id', ''))]
    if dieu93_truncated:
        logger.info(f"DEBUG: Điều 93 chunk 22f37374 IN truncated_chunks")
    else:
        dieu93_merged = [c for c in merged_chunks if '22f37374' in str(c.get('chunk_id', ''))]
        if dieu93_merged:
            pos = merged_chunks.index(dieu93_merged[0]) if dieu93_merged else -1
            logger.info(f"DEBUG: Điều 93 chunk 22f37374 in MERGED at position {pos}/{len(merged_chunks)}, but NOT in truncated (truncated count={len(truncated_chunks)})")
        else:
            logger.info(f"DEBUG: Điều 93 chunk 22f37374 NOT in merged_chunks")

    # Check if Mẫu số 2 chunk (23fad5cb) is in truncated_chunks
    mauso2_truncated = [c for c in truncated_chunks if '23fad5cb' in str(c.get('chunk_id', ''))]
    if mauso2_truncated:
        content = mauso2_truncated[0].get('content', '')[:200]
        has_url = 'thuvienphapluat' in mauso2_truncated[0].get('content', '')
        logger.info(f"DEBUG: Mẫu số 2 chunk 23fad5cb IN truncated_chunks, has_url={has_url}, content_preview={content}")
    else:
        mauso2_merged = [c for c in merged_chunks if '23fad5cb' in str(c.get('chunk_id', ''))]
        if mauso2_merged:
            pos = merged_chunks.index(mauso2_merged[0]) if mauso2_merged else -1
            logger.info(f"DEBUG: Mẫu số 2 chunk 23fad5cb in MERGED at position {pos}/{len(merged_chunks)}, but NOT in truncated")
        else:
            logger.info(f"DEBUG: Mẫu số 2 chunk 23fad5cb NOT in merged_chunks")


    # Generate reference list from truncated chunks using the new common function
    reference_list, truncated_chunks = generate_reference_list_from_chunks(
        truncated_chunks
    )

    # Rebuild chunks_context with truncated chunks
    # The actual tokens may be slightly less than available_chunk_tokens due to deduplication logic
    chunks_context = []
    for i, chunk in enumerate(truncated_chunks):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not chunks_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
    )
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data.get('data', {}).get('entities', []))} entities, {len(final_data.get('data', {}).get('relationships', []))} relationships, {len(final_data.get('data', {}).get('chunks', []))} chunks"
    )
    return result, final_data


# Now let's update the old _build_query_context to use the new architecture
async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
    entity_chunks_db: BaseKVStorage = None,
    domain: DomainConfig | None = None,
) -> QueryContextResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Returns unified QueryContextResult containing both context and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    import time as _time
    _total_start = _time.time()

    # Stage 1: Pure search
    _stage1_start = _time.time()
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )
    logger.info(f"[PERF] Stage 1 (KG Search): {_time.time() - _stage1_start:.2f}s")

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["chunk_tracking"]:
                return None

    # Stage 2: Apply token truncation for LLM efficiency
    _stage2_start = _time.time()
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
        ll_keywords=ll_keywords,  # Pass keywords to boost matching entities
    )
    logger.info(f"[PERF] Stage 2 (Truncation): {_time.time() - _stage2_start:.2f}s")

    # Stage 3: Merge chunks using filtered entities/relations
    _stage3_start = _time.time()
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
        entity_chunks_db=entity_chunks_db,
    )
    logger.info(f"[PERF] Stage 3 (Merge Chunks): {_time.time() - _stage3_start:.2f}s")

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    _stage4_start = _time.time()
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
        domain=domain,
    )
    logger.info(f"[PERF] Stage 4 (Build Context): {_time.time() - _stage4_start:.2f}s")
    logger.info(f"[PERF] Total _build_query_context: {_time.time() - _total_start:.2f}s")

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    results = await entities_vdb.query(query, top_k=query_param.top_k)

    # DEBUG: Log top entity results with similarity scores
    if results:
        top_entities = [(r.get("entity_name", "?"), r.get("distance", r.get("score", "?"))) for r in results[:15]]
        logger.info(f"[DEBUG] Top 15 entity vector search results: {top_entities}")

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results] if results else []
    seen_entity_ids = set(node_ids)

    
    # Also search entities by description if query contains concept keywords
    # This helps find entities like "Điều 195" when querying for concepts like "sở hữu chéo"
    query_keywords = [kw.strip() for kw in query.split(",") if kw.strip()]
    if query_keywords:
        try:
            description_results = await knowledge_graph_inst.search_entities_by_description(
                keywords=query_keywords, limit=10
            )
            for ent in description_results:
                ent_id = ent.get("entity_id")
                if ent_id and ent_id not in seen_entity_ids:
                    node_ids.append(ent_id)
                    seen_entity_ids.add(ent_id)
                    # Add to results with a lower similarity score
                    results.append({
                        "entity_name": ent_id,
                        "created_at": None,
                        "_from_description_search": True  # Mark for debugging
                    })
                    logger.debug(f"Added entity from description search: {ent_id}")
        except Exception as e:
            logger.warning(f"Description search failed: {e}")

    if not len(results):
        return [], []

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    # Multi-hop traversal: expand to find connected entities if hop_depth > 1
    hop_depth = getattr(query_param, "hop_depth", 1)
    if hop_depth > 1 and use_relations:
        expanded_entities, expanded_relations = await _expand_entities_by_hop(
            initial_entities=node_datas,
            initial_relations=use_relations,
            knowledge_graph_inst=knowledge_graph_inst,
            query_param=query_param,
            current_depth=1,
            max_depth=hop_depth,
        )
        node_datas = expanded_entities
        use_relations = expanded_relations
        logger.info(
            f"After {hop_depth}-hop expansion: {len(node_datas)} entities, {len(use_relations)} relations"
        )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _expand_entities_by_hop(
    initial_entities: list[dict],
    initial_relations: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    query_param: QueryParam,
    current_depth: int,
    max_depth: int,
    query_entity_names: set[str] = None,  # Track entities from original query for higher priority
) -> tuple[list[dict], list[dict]]:
    """
    Expand entities by traversing the knowledge graph for multi-hop retrieval.

    This is useful for legal/regulatory domains where:
    - An amendment (Khoản 10 sửa đổi 2025) supplements an original article (Điều 23)
    - The query finds "Điều 23" but needs to also include the amendment

    Args:
        initial_entities: Currently collected entities
        initial_relations: Currently collected relations
        knowledge_graph_inst: Graph storage instance
        query_param: Query parameters
        current_depth: Current traversal depth (1-based)
        max_depth: Maximum traversal depth
        query_entity_names: Set of entity names from original query (for priority boost)

    Returns:
        Expanded list of entities and relations
    """
    if current_depth >= max_depth:
        return initial_entities, initial_relations

    # Initialize query_entity_names on first call (depth=1)
    # These are the entities directly found from the query, they get highest priority
    if query_entity_names is None:
        query_entity_names = {e["entity_name"] for e in initial_entities}
        logger.debug(f"Hop {current_depth + 1}: Tracking {len(query_entity_names)} query entities for priority boost")

    seen_entity_names = {e["entity_name"] for e in initial_entities}
    seen_relation_pairs = {r["src_tgt"] for r in initial_relations}

    # IMPORTANT: Boost initial entities from direct vector search  
    # Only boost TOP N entities (ordered by similarity), with decreasing boost based on position
    # This ensures high-similarity entities (like Điều 26) rank above generic entities
    TOP_N_TO_BOOST = 20  # Only boost top 20 entities from vector search
    BASE_BOOST = 10000   # Starting boost for top entity
    BOOST_DECREASE = 200  # Decrease per position
    
    all_entities = []
    boosted_count = 0
    for idx, entity in enumerate(initial_entities):
        boosted_entity = entity.copy()
        # Only boost top N entities from vector search (they're ordered by similarity)
        if idx < TOP_N_TO_BOOST and not boosted_entity.get("is_direct_link"):
            boosted_entity["is_direct_link"] = True
            boosted_entity["is_supplementary"] = True  # For sorting purposes
            # Sliding boost: top entity gets BASE_BOOST, decreasing for lower positions
            position_boost = max(BASE_BOOST - (idx * BOOST_DECREASE), 6000)  # Minimum 6000
            current_rank = boosted_entity.get("rank", 0)
            boosted_entity["rank"] = current_rank + position_boost
            boosted_count += 1
        all_entities.append(boosted_entity)
    
    if boosted_count > 0:
        boosted_sample = [(e.get("entity_name", "?")[:40], e.get("rank", 0)) for e in all_entities[:10]]
        logger.info(f"[HOP] Boosted {boosted_count}/{len(initial_entities)} direct search entities (top {TOP_N_TO_BOOST}). Sample: {boosted_sample}")



    all_relations = list(initial_relations)

    # Find entities connected via relations but not yet in our entity list
    # Also identify "supplementary" relations (bổ sung, sửa đổi, thay thế, etc.)
    new_entity_names = set()
    supplementary_entity_names = set()  # Entities connected via supplementary relations
    direct_query_linked_entities = set()  # Entities directly linked to query entities (highest priority)
    
    # Keywords indicating supplementary/amendment relations (lowercase for matching)
    # Include both singular and plural forms, and Vietnamese equivalents
    supplementary_keywords = [
        "bổ sung", "sửa đổi", "thay thế", "điều chỉnh", "cập nhật",
        "supplement", "supplements", "amend", "amends", "amended",
        "replace", "replaces", "modify", "modifies", "modified",
        "update", "updates", "revise", "revises", "revised",
        "add", "adds", "added", "change", "changes", "changed",
        "luật sửa đổi"
    ]
    
    # Keywords indicating parent-child legal structure (Điều → Khoản → Điểm)
    # These should also be prioritized when traversing legal documents
    parent_child_keywords = [
        "is part of", "part of", "belongs to", "thuộc về", "thuộc", 
        "nằm trong", "quy định tại", "specified in", "stipulates"
    ]
    
    # Track relations that link to Khoản 10 or Điều 23 for debugging
    khoan_10_relations = []
    dieu_23_relations = []
    
    for relation in initial_relations:
        src, tgt = relation["src_tgt"]
        description = relation.get("description", "").lower()
        
        # Debug: Check for Khoản 10 related relations
        if "khoản 10" in src.lower() or "khoản 10" in tgt.lower():
            khoan_10_relations.append((src, tgt, description[:100]))
        
        # Debug: Check for Điều 23 related relations
        if "điều 23" in src.lower() or "điều 23" in tgt.lower():
            dieu_23_relations.append((src, tgt, description[:100]))
        
        # Check if this is a supplementary relation
        is_supplementary = any(kw in description for kw in supplementary_keywords)
        
        # Also check for parent-child legal relations (Khoản is part of Điều)
        is_parent_child = any(kw in description for kw in parent_child_keywords)
        
        # Check if this relation links to a query entity (highest priority)
        src_is_query = src in query_entity_names
        tgt_is_query = tgt in query_entity_names
        
        if src not in seen_entity_names:
            new_entity_names.add(src)
            if is_supplementary or is_parent_child:
                supplementary_entity_names.add(src)
                # DON'T mark as direct_query_linked at HOP 1 (current_depth=1)
                # At HOP 1, there are too many parent-child relations from initial entities
                # Wait until HOP 2+ where relations come from expanded entities (like Điều 23)
                # This is handled in the new_relations processing section below
        if tgt not in seen_entity_names:
            new_entity_names.add(tgt)
            if is_supplementary or is_parent_child:
                supplementary_entity_names.add(tgt)
                # DON'T mark at HOP 1, same reason as above
    
    # Log if we found any Khoản 10 or Điều 23 related relations
    if khoan_10_relations:
        logger.info(f"Hop {current_depth + 1}: Found {len(khoan_10_relations)} relations involving Khoản 10: {khoan_10_relations[:3]}")
    if dieu_23_relations:
        logger.info(f"Hop {current_depth + 1}: Found {len(dieu_23_relations)} relations involving Điều 23: {dieu_23_relations[:3]}")
    
    # Log direct query linked entities (highest priority)
    if direct_query_linked_entities:
        logger.info(f"Hop {current_depth + 1}: Found {len(direct_query_linked_entities)} entities DIRECTLY linked to query entities")
    
    # Log total initial_relations count for debugging
    logger.debug(f"Hop {current_depth + 1}: Processing {len(initial_relations)} initial relations to find new entities")

    if not new_entity_names:
        return all_entities, all_relations

    # Fetch data for new entities
    new_entity_names_list = list(new_entity_names)
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(new_entity_names_list),
        knowledge_graph_inst.node_degrees_batch(new_entity_names_list),
    )

    new_entities = []
    supplementary_entities = []  # High-priority entities from supplementary relations
    
    for name in new_entity_names_list:
        node_data = nodes_dict.get(name)
        if node_data:
            # Boost rank for supplementary entities to prioritize them in truncation
            base_rank = degrees_dict.get(name, 0)
            is_supplementary = name in supplementary_entity_names
            is_direct_link = name in direct_query_linked_entities  # Check if directly linked to query entity
            
            # Apply higher boost for entities directly linked to query entities
            # Using 2000 to ensure they rank reasonably high but not dominate everything
            if is_direct_link:
                boost = 2000  # High priority - parent-child link to existing/query entity
            elif is_supplementary:
                boost = 1000  # Medium priority - supplementary relation
            else:
                boost = 0
            
            entity = {
                **node_data,
                "entity_name": name,
                "rank": base_rank + boost,
                "hop_level": current_depth + 1,
                "is_supplementary": is_supplementary or is_direct_link,  # Treat direct links as supplementary for sorting
                "is_direct_link": is_direct_link,  # Keep this flag for chunk prioritization
            }
            
            # Debug log for Khoản 10
            if "Khoản 10" in name:
                logger.debug(f"Creating entity '{name}': base_rank={base_rank}, is_supplementary={is_supplementary}, is_direct_link={is_direct_link}, boost={boost}, final_rank={entity['rank']}")
            
            if is_supplementary or is_direct_link:
                supplementary_entities.append(entity)
            else:
                new_entities.append(entity)
            seen_entity_names.add(name)

    # Insert supplementary entities right after initial entities (higher priority)
    # This ensures they survive truncation better
    if supplementary_entities:
        logger.info(
            f"Hop {current_depth + 1}: Found {len(supplementary_entities)} supplementary entities: "
            f"{[e['entity_name'] for e in supplementary_entities[:5]]}..."
        )
        # Insert supplementary entities at a position that gives them better survival chance
        insert_position = min(len(all_entities), query_param.top_k // 2)
        for i, entity in enumerate(supplementary_entities):
            all_entities.insert(insert_position + i, entity)
    
    # Add regular new entities at the end
    all_entities.extend(new_entities)

    # Find new relations from the newly discovered entities
    all_new_entities = supplementary_entities + new_entities
    if all_new_entities:
        new_node_datas = [{"entity_name": e["entity_name"]} for e in all_new_entities]
        
        # Debug: Log if Điều 23 is in new entities
        dieu_23_in_new = [e["entity_name"] for e in all_new_entities if "điều 23" in e["entity_name"].lower()]
        if dieu_23_in_new:
            logger.info(f"Hop {current_depth + 1}: Fetching relations for Điều 23 entities: {dieu_23_in_new}")
        
        new_relations_raw = await _find_most_related_edges_from_entities(
            new_node_datas,
            query_param,
            knowledge_graph_inst,
        )
        
        # Debug: Log Điều 23 relations found
        dieu_23_new_relations = [(r["src_tgt"], r.get("description", "")[:80]) for r in new_relations_raw 
                                  if any("điều 23" in str(s).lower() for s in r["src_tgt"])]
        if dieu_23_new_relations:
            logger.info(f"Hop {current_depth + 1}: New relations of Điều 23: {dieu_23_new_relations[:5]}")

        # Separate supplementary relations from regular relations
        # Also track entities that should be boosted due to supplementary relations
        supplementary_relations = []
        regular_relations = []
        entities_to_boost = set()  # Entity names that should be boosted
        
        for rel in new_relations_raw:
            src, tgt = rel["src_tgt"]
            description = rel.get("description", "").lower()
            
            # Debug: Check if this is the Khoản 10 - Điều 23 relation
            is_khoan_10_rel = ("khoản 10" in src.lower() and "điều 23" in src.lower()) or ("khoản 10" in tgt.lower() and "điều 23" in tgt.lower())
            if is_khoan_10_rel:
                in_seen = rel["src_tgt"] in seen_relation_pairs
                logger.info(f"DEBUG: Found Khoản 10 relation: {src} <-> {tgt}, already_seen={in_seen}, desc={description[:80]}")
            
            if rel["src_tgt"] not in seen_relation_pairs:
                is_supplementary = any(kw in description for kw in supplementary_keywords)
                is_parent_child = any(kw in description for kw in parent_child_keywords)
                
                # Check if connected to query entity OR existing entity (high priority)
                # query_entity_names: entities from original query (highest priority)
                # seen_entity_names: entities already collected (high priority for parent-child)
                src_is_query = src in query_entity_names
                tgt_is_query = tgt in query_entity_names
                src_is_existing = src in seen_entity_names
                tgt_is_existing = tgt in seen_entity_names
                
                if is_supplementary or is_parent_child:
                    # Boost weight for supplementary/parent-child relations
                    rel["weight"] = rel.get("weight", 1.0) + 100
                    supplementary_relations.append(rel)
                    # Track entities from this relation for boosting
                    entities_to_boost.add(src)
                    entities_to_boost.add(tgt)
                    
                    # Mark as direct_query_linked if it's a supplementary relation
                    # This ensures entities like "Khoản 10" (added by amendment) get high priority
                    # We no longer require "not in seen_entity_names" because we want to boost
                    # entities even if they were discovered earlier through other relations
                    if is_supplementary:
                        direct_query_linked_entities.add(src)
                        direct_query_linked_entities.add(tgt)
                        logger.debug(f"Adding '{src}', '{tgt}' to direct_query_linked (SUPPLEMENTARY relation)")
                    
                    # Debug: log when Khoản 10 - Điều 23 is added to boost list
                    if "khoản 10" in src.lower() and "điều 23" in src.lower():
                        # Extra debug for Khoản 10 relation
                        logger.info(f"DEBUG Khoản 10 relation details: src='{src}', tgt='{tgt}', src_in_seen={src in seen_entity_names}, tgt_in_seen={tgt in seen_entity_names}, is_parent={is_parent_child}, is_supp={is_supplementary}")
                        logger.info(f"DEBUG: Adding '{src}' to entities_to_boost (is_supp={is_supplementary}, is_parent={is_parent_child}, in_direct_query={src in direct_query_linked_entities})")
                    if "khoản 10" in tgt.lower() and "điều 23" in tgt.lower():
                        logger.info(f"DEBUG: Adding '{tgt}' to entities_to_boost (is_supp={is_supplementary}, is_parent={is_parent_child}, in_direct_query={tgt in direct_query_linked_entities})")
                    
                    # Debug: log when Khoản 5a - Điều 8 is processed
                    if "khoản 5a" in src.lower() or "khoản 5a" in tgt.lower():
                        logger.info(f"DEBUG Khoản 5a relation: src='{src}', tgt='{tgt}', is_supp={is_supplementary}, is_parent={is_parent_child}, src_in_direct={src in direct_query_linked_entities}, tgt_in_direct={tgt in direct_query_linked_entities}")
                else:
                    regular_relations.append(rel)
                seen_relation_pairs.add(rel["src_tgt"])
        
        # Debug: Check if Khoản 10 - Điều 23 is in entities_to_boost
        khoan_10_dieu_23_in_boost = [e for e in entities_to_boost if "khoản 10" in e.lower() and "điều 23" in e.lower()]
        if khoan_10_dieu_23_in_boost:
            logger.info(f"DEBUG: Khoản 10 - Điều 23 entities in boost list: {khoan_10_dieu_23_in_boost}")
        
        # Debug: Check if Khoản 10 - Điều 23 is in direct_query_linked_entities
        khoan_10_direct = [e for e in direct_query_linked_entities if "khoản 10" in e.lower() and "điều 23" in e.lower()]
        if khoan_10_direct:
            logger.info(f"DEBUG: Khoản 10 - Điều 23 in DIRECT QUERY LINKED list: {khoan_10_direct}")
        
        # Boost entities that are connected via supplementary relations
        # Move them to higher priority positions in the entity list
        # Also add NEW entities that were discovered from supplementary relations
        if entities_to_boost:
            # Find entities that are NOT yet in all_entities
            existing_entity_names = {e.get("entity_name", "") for e in all_entities}
            new_entity_names_from_relations = entities_to_boost - existing_entity_names
            
            # Debug: Check if Khoản 10 - Điều 23 is a new entity or already exists
            khoan_10_in_existing = [e for e in existing_entity_names if "khoản 10" in e.lower() and "điều 23" in e.lower()]
            khoan_10_in_new = [e for e in new_entity_names_from_relations if "khoản 10" in e.lower() and "điều 23" in e.lower()]
            if khoan_10_in_existing:
                logger.info(f"DEBUG: Khoản 10 - Điều 23 ALREADY EXISTS in all_entities: {khoan_10_in_existing}")
            if khoan_10_in_new:
                logger.info(f"DEBUG: Khoản 10 - Điều 23 will be ADDED as NEW entity: {khoan_10_in_new}")
            
            # Fetch data for new entities from supplementary relations
            if new_entity_names_from_relations:
                new_names_list = list(new_entity_names_from_relations)
                new_nodes_dict, new_degrees_dict = await asyncio.gather(
                    knowledge_graph_inst.get_nodes_batch(new_names_list),
                    knowledge_graph_inst.node_degrees_batch(new_names_list),
                )
                
                new_supplementary_entities = []
                for name in new_names_list:
                    node_data = new_nodes_dict.get(name)
                    if node_data:
                        is_direct_link = name in direct_query_linked_entities  # Check if directly linked to query entity
                        boost = 3000 if is_direct_link else 1000  # 3000 for direct links (parent-child of query entities)
                        entity = {
                            **node_data,
                            "entity_name": name,
                            "rank": new_degrees_dict.get(name, 0) + boost,
                            "hop_level": current_depth + 1,
                            "is_supplementary": True,
                            "is_direct_link": is_direct_link,  # Keep this flag for chunk prioritization
                        }
                        new_supplementary_entities.append(entity)
                        seen_entity_names.add(name)
                        # Debug: log when Khoản 10 - Điều 23 is added
                        if "khoản 10" in name.lower() and "điều 23" in name.lower():
                            logger.info(f"DEBUG: Adding entity '{name}' with rank={entity['rank']}, is_direct_link={is_direct_link}, boost={boost}")
                
                if new_supplementary_entities:
                    logger.info(
                        f"Hop {current_depth + 1}: Adding {len(new_supplementary_entities)} NEW entities from supplementary relations: "
                        f"{[e['entity_name'] for e in new_supplementary_entities[:5]]}..."
                    )
                    # Insert at high priority position
                    insert_pos = min(len(all_entities), query_param.top_k // 2)
                    for i, entity in enumerate(new_supplementary_entities):
                        all_entities.insert(insert_pos + i, entity)
            
            # Now boost existing entities
            boosted_entities = []
            remaining_entities = []
            for entity in all_entities:
                entity_name = entity.get("entity_name", "")
                if entity_name in entities_to_boost:
                    # Check if directly linked to query entities (SUPPLEMENTARY relations)
                    is_direct_link = entity_name in direct_query_linked_entities
                    new_boost = 3000 if is_direct_link else 1000
                    
                    # Already boosted before?
                    already_boosted = entity.get("is_supplementary", False)
                    current_is_direct = entity.get("is_direct_link", False)
                    
                    # Update if:
                    # 1. Not boosted yet, OR
                    # 2. Now is direct_link but wasn't before (upgrade boost from 1000 to 3000)
                    if not already_boosted:
                        # First time boosting
                        entity["is_supplementary"] = True
                        entity["is_direct_link"] = is_direct_link
                        entity["rank"] = entity.get("rank", 0) + new_boost
                        boosted_entities.append(entity)
                    elif is_direct_link and not current_is_direct:
                        # Upgrade: was 1000, now should be 3000
                        entity["is_direct_link"] = True
                        entity["rank"] = entity.get("rank", 0) + 2000  # Add extra 2000 to reach 3000 total
                        boosted_entities.append(entity)
                    else:
                        remaining_entities.append(entity)
                else:
                    remaining_entities.append(entity)
            
            if boosted_entities:
                logger.info(
                    f"Hop {current_depth + 1}: Boosting {len(boosted_entities)} existing entities from supplementary relations: "
                    f"{[e['entity_name'] for e in boosted_entities[:5]]}..."
                )
                # Insert boosted entities at high priority position
                insert_pos = min(len(remaining_entities), query_param.top_k // 2)
                all_entities = remaining_entities[:insert_pos] + boosted_entities + remaining_entities[insert_pos:]
        
        # Insert supplementary relations near the beginning for better truncation survival
        if supplementary_relations:
            logger.debug(
                f"Hop {current_depth + 1}: Found {len(supplementary_relations)} supplementary/parent-child relations"
            )
            insert_pos = min(len(all_relations), query_param.top_k // 2)
            for i, rel in enumerate(supplementary_relations):
                all_relations.insert(insert_pos + i, rel)
        
        # Add regular relations at the end
        all_relations.extend(regular_relations)

        # Recursively expand if we haven't reached max depth
        if current_depth + 1 < max_depth:
            all_entities, all_relations = await _expand_entities_by_hop(
                initial_entities=all_entities,
                initial_relations=all_relations,
                knowledge_graph_inst=knowledge_graph_inst,
                query_param=query_param,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                query_entity_names=query_entity_names,  # Pass through to maintain query entity tracking
            )

    logger.debug(
        f"Hop {current_depth + 1}: Added {len(all_new_entities)} entities, "
        f"total now {len(all_entities)} entities, {len(all_relations)} relations"
    )

    return all_entities, all_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Debug: Check Điều 26 in node_datas before processing
    dieu26_check = [e for e in node_datas if "Điều 26 - Luật Doanh nghiệp" in e.get("entity_name", "")]
    if dieu26_check:
        logger.info(f"DEBUG: Điều 26 in node_datas: {[(e.get('entity_name')[:50], e.get('rank'), e.get('is_direct_link'), e.get('is_supplementary')) for e in dieu26_check]}")

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []

    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )
    
    # Debug: Check Khoản 10 in entities_with_chunks
    khoan10_check = [e for e in entities_with_chunks if "Khoản 10 - Điều 23" in e.get("entity_name", "")]
    if khoan10_check:
        logger.info(f"DEBUG: Khoản 10-23 in entities_with_chunks: {[(e['entity_name'], e['chunks'][:2], e['entity_data'].get('is_supplementary'), e['entity_data'].get('is_direct_link')) for e in khoan10_check]}")

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    # But for supplementary entities, always keep their chunks
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        is_supplementary = entity_info.get("entity_data", {}).get("is_supplementary", False)
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it
            # Also keep for supplementary entities regardless of count
            if chunk_occurrence_count[chunk_id] == 1 or is_supplementary:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    # Debug: Check Khoản 10 after sorting
    khoan10_after_sort = [e for e in entities_with_chunks if "Khoản 10 - Điều 23" in e.get("entity_name", "")]
    if khoan10_after_sort:
        logger.info(f"DEBUG: Khoản 10-23 after sort: chunks={khoan10_after_sort[0].get('chunks', [])}, sorted_chunks={khoan10_after_sort[0].get('sorted_chunks', [])}, is_supplementary={khoan10_after_sort[0].get('entity_data', {}).get('is_supplementary')}")

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                # Collect priority chunks ONLY from entities with is_direct_link=True
                # These are entities directly linked to query entities (parent-child/supplementary)
                # This ensures only truly relevant chunks get priority treatment
                priority_chunk_ids = set()
                direct_link_entities_found = []
                for entity_info in entities_with_chunks:
                    entity_data = entity_info.get("entity_data", {})
                    entity_name = entity_info.get("entity_name", "")
                    # Only use is_direct_link for priority, not all is_supplementary
                    if entity_data.get("is_direct_link"):
                        direct_link_entities_found.append(entity_name)
                        for chunk_id in entity_info.get("sorted_chunks", []):
                            priority_chunk_ids.add(chunk_id)
                    
                    # Debug: Log Khoản 10 entity
                    if "khoản 10" in entity_name.lower() and "điều 23" in entity_name.lower():
                        logger.info(f"DEBUG: Khoản 10-23 entity_data: is_direct_link={entity_data.get('is_direct_link')}, is_supplementary={entity_data.get('is_supplementary')}, chunks={entity_info.get('sorted_chunks', [])}")
                    
                    # Debug: Log Khoản 5a entity
                    if "khoản 5a" in entity_name.lower():
                        logger.info(f"DEBUG: Khoản 5a entity_data: name={entity_name}, is_direct_link={entity_data.get('is_direct_link')}, is_supplementary={entity_data.get('is_supplementary')}, chunks={entity_info.get('sorted_chunks', [])}")
                
                logger.info(f"DEBUG: entities_with_chunks has {len(entities_with_chunks)} entities")
                logger.info(f"DEBUG: Found {len(direct_link_entities_found)} direct_link entities for priority: {direct_link_entities_found[:10]}")
                logger.info(f"DEBUG: priority_chunk_ids has {len(priority_chunk_ids)} chunks: {list(priority_chunk_ids)[:5]}")
                
                # Check if Khoản 10 chunks are in priority
                khoan10_priority = [c for c in priority_chunk_ids if "f5c8b3c5" in c]
                if khoan10_priority:
                    logger.info(f"DEBUG: Khoản 10-23 chunks IN priority_chunk_ids: {khoan10_priority}")
                else:
                    logger.info(f"DEBUG: Khoản 10-23 chunks NOT in priority_chunk_ids")
                
                # Check if Khoản 5a chunks are in priority
                khoan5a_priority = [c for c in priority_chunk_ids if "ffacaa2e" in c]
                if khoan5a_priority:
                    logger.info(f"DEBUG: Khoản 5a chunks IN priority_chunk_ids: {khoan5a_priority}")
                else:
                    logger.info(f"DEBUG: Khoản 5a chunks NOT in priority_chunk_ids")
                
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )
                
                # Ensure priority chunks from direct_link entities are always included
                if priority_chunk_ids:
                    selected_set = set(selected_chunk_ids)
                    missing_priority = priority_chunk_ids - selected_set
                    if missing_priority:
                        # Add missing priority chunks at the beginning
                        selected_chunk_ids = list(missing_priority) + selected_chunk_ids
                        logger.debug(f"Added {len(missing_priority)} priority chunks from supplementary entities")

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Build chunk_id to entity_rank mapping for priority sorting in merge
    chunk_to_entity_rank = {}
    for entity_info in entities_with_chunks:
        entity_data = entity_info.get("entity_data", {})
        entity_rank = entity_data.get("rank", 0)
        for chunk_id in entity_info.get("sorted_chunks", []):
            # Keep highest rank if chunk belongs to multiple entities
            if chunk_id not in chunk_to_entity_rank or entity_rank > chunk_to_entity_rank[chunk_id]:
                chunk_to_entity_rank[chunk_id] = entity_rank

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    # Mark priority chunks so they can be prioritized in merge
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            # Mark as priority if this chunk is from a direct_link entity
            chunk_data_copy["is_priority"] = chunk_id in priority_chunk_ids
            # Add entity_rank for priority sorting
            chunk_data_copy["entity_rank"] = chunk_to_entity_rank.get(chunk_id, 0)
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[False] = False,
) -> str | AsyncIterator[str]: ...


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> QueryResult | None:
    """
    Execute naive query and return unified QueryResult object.

    Args:
        query: Query string
        chunks_vdb: Document chunks vector database
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Returns None when no relevant chunks are retrieved.
    """

    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"])

    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        logger.info(
            "[naive_query] No relevant document chunks found; returning no-result."
        )
        return None

    # Calculate dynamic token limit for chunks
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate system prompt template tokens (excluding content_data)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a preliminary system prompt with empty content_data to calculate overhead
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=user_prompt,
        content_data="",  # Empty for overhead calculation
    )

    # Calculate available tokens for chunks
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from processed chunks using the new common function
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )

    logger.info(f"Final context: {len(processed_chunks_with_ref_ids)} chunks")

    # Build raw data structure for naive mode using processed chunks with reference IDs
    raw_data = convert_to_user_format(
        [],  # naive mode has no entities
        [],  # naive mode has no relationships
        processed_chunks_with_ref_ids,
        reference_list,
        "naive",
    )

    # Add complete metadata for naive mode
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": [],  # naive mode has no keyword extraction
        "low_level": [],  # naive mode has no keyword extraction
    }
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }

    # Build chunks_context from processed chunks with reference IDs
    chunks_context = []
    for i, chunk in enumerate(processed_chunks_with_ref_ids):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    naive_context_template = PROMPTS["naive_query_context"]
    context_content = naive_context_template.format(
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_content, raw_data=raw_data)

    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=user_prompt,
        content_data=context_content,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=raw_data)

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt) :]
                .replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return QueryResult(content=response, raw_data=raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response, raw_data=raw_data, is_streaming=True
        )
