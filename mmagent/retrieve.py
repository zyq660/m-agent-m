# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced agentic retrieval for multimodal long-term memory.

Design goals
------------
1. Keep the original public APIs backward compatible.
2. Preserve the original text-memory retrieval path as the safe fallback.
3. Add query-adaptive memory routing, temporal expansion, uncertainty-aware
   route switching, evidence/provenance enrichment, and utility-aware query
   selection without requiring changes to the rest of the project.
4. If the current VideoGraph exposes richer visual/audio evidence in node
   metadata, use it automatically. If it does not, no new dependency is needed.
5. Support optional on-demand evidence rehydration through an evidence_provider
   callback or compatible VideoGraph method, while remaining a no-op otherwise.
"""

import json
import logging
import math
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
    get_embedding_with_retry,
)
from .utils.general import validate_and_fix_python_list
from .prompts import *
from .memory_processing import parse_video_caption


processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config.get("max_retries", 3)
EMBEDDING_MODEL = processing_config.get("embedding_model", "text-embedding-3-large")

# Optional enhanced-retrieval settings. They deliberately have defaults so the
# existing processing_config.json does not need to be changed.
ACTIVE_RETRIEVAL_CONFIG = processing_config.get("active_memory_retrieval", {})
LOW_CONFIDENCE_THRESHOLD = float(
    ACTIVE_RETRIEVAL_CONFIG.get("low_confidence_threshold", 0.35)
)
LOW_CONFIDENCE_PATIENCE = int(
    ACTIVE_RETRIEVAL_CONFIG.get("low_confidence_patience", 2)
)
TEMPORAL_NEIGHBOR_WINDOW = int(
    ACTIVE_RETRIEVAL_CONFIG.get("temporal_neighbor_window", 1)
)
QUERY_RELEVANCE_WEIGHT = float(
    ACTIVE_RETRIEVAL_CONFIG.get("query_relevance_weight", 0.65)
)
QUERY_NOVELTY_WEIGHT = float(
    ACTIVE_RETRIEVAL_CONFIG.get("query_novelty_weight", 0.35)
)
ENABLE_EVIDENCE_REHYDRATION = bool(
    ACTIVE_RETRIEVAL_CONFIG.get("enable_evidence_rehydration", True)
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_id_from_key(key: Any) -> Optional[int]:
    if isinstance(key, int):
        return key
    match = re.search(r"CLIP_(\d+)", str(key))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    output = []
    for item in items:
        marker = str(item)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def _normalize_query_list(value: Any) -> List[str]:
    """Normalize LLM-produced search queries to a clean list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        value = [str(value)]

    queries = []
    for item in value:
        if item is None:
            continue
        item = str(item).strip()
        if item:
            queries.append(item)
    return _dedupe_keep_order(queries)


def _strip_reasoning_labels(text: str) -> str:
    """Remove known prompt labels safely; unlike str.strip this removes prefixes."""
    text = (text or "").strip()
    labels = (
        "### Reasoning:",
        "### Answer or Search:",
        "Reasoning:",
    )
    changed = True
    while changed:
        changed = False
        for label in labels:
            if text.startswith(label):
                text = text[len(label):].strip()
                changed = True
    return text


def _safe_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _normalize_similarity(score: float) -> float:
    """Map a cosine-like score to [0, 1] without assuming a strict backend range."""
    score = _safe_float(score)
    if -1.0 <= score <= 1.0:
        # Most retrieval scores in this project are cosine-like. Keep positive
        # scores intuitive while still handling negative values gracefully.
        return max(0.0, min(1.0, score)) if score >= 0 else (score + 1.0) / 2.0
    # For unbounded/custom scores use a smooth squash.
    return 1.0 / (1.0 + math.exp(-score))


# ---------------------------------------------------------------------------
# Query-adaptive memory routing
# ---------------------------------------------------------------------------

def infer_memory_route(question: str, retrieval_plan: Optional[str] = None) -> str:
    """
    Infer which memory view is most useful for a query.

    This does not require new graph fields. It is primarily used to:
    - prefer appropriate memory node types,
    - expand temporal neighbors for temporal/causal questions,
    - collect optional visual/audio evidence if such metadata exists,
    - decide when evidence rehydration is worth attempting.
    """
    text = f"{question or ''} {retrieval_plan or ''}".lower()

    visual_terms = (
        "color", "colour", "look", "looks", "wear", "wearing", "shape",
        "left hand", "right hand", "screen", "logo", "text on", "written",
        "read", "ocr", "visible", "appearance", "image", "frame", "object",
        "颜色", "穿", "长什么", "画面", "屏幕", "文字", "左手", "右手", "视觉",
    )
    audio_terms = (
        "say", "said", "speak", "spoken", "hear", "heard", "sound", "audio",
        "music", "voice", "speaker", "dialogue", "conversation", "tell",
        "说了", "听到", "声音", "音乐", "对话", "语音",
    )
    temporal_terms = (
        "before", "after", "then", "next", "previous", "first", "last",
        "earlier", "later", "when", "while", "during", "until", "sequence",
        "cause", "caused", "why", "because", "lead to", "result",
        "之前", "之后", "然后", "接着", "最先", "最后", "什么时候", "期间",
        "为什么", "导致", "因果", "顺序",
    )
    semantic_terms = (
        "usually", "generally", "often", "habit", "typically", "always",
        "normally", "relationship", "preference", "overall", "in general",
        "通常", "经常", "习惯", "一般", "总体", "长期", "偏好",
    )

    has_visual = any(term in text for term in visual_terms)
    has_audio = any(term in text for term in audio_terms)
    has_temporal = any(term in text for term in temporal_terms)
    has_semantic = any(term in text for term in semantic_terms)

    modal_count = int(has_visual) + int(has_audio)
    if modal_count >= 2:
        return "multimodal"
    if has_visual:
        return "visual"
    if has_audio:
        return "audio"
    if has_temporal:
        return "temporal"
    if has_semantic:
        return "semantic"
    return "text"


# ---------------------------------------------------------------------------
# Memory translation / entity normalization
# ---------------------------------------------------------------------------

def translate(video_graph, memories):
    new_memories = []
    for memory in memories:
        if memory is None:
            continue
        memory = str(memory)
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        entities = list(set(entities))
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            reverse_mappings = getattr(video_graph, "reverse_character_mappings", {})
            if entity_str in reverse_mappings:
                new_memory = new_memory.replace(entity_str, reverse_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories


def back_translate(video_graph, queries):
    translated_queries = []
    character_mappings = getattr(video_graph, "character_mappings", {})
    for query in queries:
        entities = parse_video_caption(video_graph, query)
        entities = list(set(entities))
        to_be_translated = [query]
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in character_mappings:
                mappings = character_mappings[entity_str]

                new_queries = []
                for mapping in mappings:
                    for partially_translated in to_be_translated:
                        new_query = partially_translated.replace(entity_str, mapping)
                        new_queries.append(new_query)

                # Avoid losing the original query if a malformed/empty mapping exists.
                if new_queries:
                    to_be_translated = new_queries

        translated_queries.extend(to_be_translated)
    return _dedupe_keep_order(translated_queries)


# ---------------------------------------------------------------------------
# Optional multimodal evidence extraction / rehydration
# ---------------------------------------------------------------------------

VISUAL_EVIDENCE_KEYS = (
    "visual_evidence",
    "visual_caption",
    "visual_description",
    "frame_caption",
    "ocr",
    "ocr_text",
    "image_caption",
)
AUDIO_EVIDENCE_KEYS = (
    "audio_evidence",
    "audio_caption",
    "transcript",
    "asr",
    "speech",
    "dialogue",
)


def _flatten_evidence_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        result = []
        for key, sub_value in value.items():
            for text in _flatten_evidence_value(sub_value):
                result.append(f"{key}: {text}")
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_evidence_value(item))
        return result
    return [str(value)]


def _extract_node_optional_evidence(node: Any, route: str) -> List[str]:
    metadata = getattr(node, "metadata", {}) or {}
    keys: Tuple[str, ...]
    if route == "visual":
        keys = VISUAL_EVIDENCE_KEYS
    elif route == "audio":
        keys = AUDIO_EVIDENCE_KEYS
    elif route == "multimodal":
        keys = VISUAL_EVIDENCE_KEYS + AUDIO_EVIDENCE_KEYS
    else:
        return []

    evidence = []
    for key in keys:
        if key not in metadata:
            continue
        prefix = "VISUAL" if key in VISUAL_EVIDENCE_KEYS else "AUDIO"
        for text in _flatten_evidence_value(metadata[key]):
            evidence.append(f"[{prefix}_EVIDENCE] {text}")
    return _dedupe_keep_order(evidence)


def _normalize_external_evidence(value: Any, clip_id: int, route: str) -> List[str]:
    items = _flatten_evidence_value(value)
    prefix = route.upper() if route else "MULTIMODAL"
    return [f"[REHYDRATED_{prefix}][CLIP_{clip_id}] {item}" for item in items if item]


def rehydrate_evidence(
    video_graph,
    clip_ids: Sequence[int],
    route: str,
    evidence_provider: Optional[Callable[[Any, int, str], Any]] = None,
) -> Dict[str, List[str]]:
    """
    Try to recover richer/raw evidence only when needed.

    Safe order:
      1) explicit evidence_provider(video_graph, clip_id, route), if supplied;
      2) VideoGraph.get_clip_evidence(clip_id, route=route), if available;
      3) VideoGraph.rehydrate_clip(clip_id, route=route), if available;
      4) no-op.

    Any provider failure is logged and falls back instead of breaking QA.
    """
    if not ENABLE_EVIDENCE_REHYDRATION:
        return {}
    if route not in {"visual", "audio", "multimodal", "temporal"}:
        return {}

    output: Dict[str, List[str]] = {}
    for clip_id in _dedupe_keep_order(clip_ids):
        if clip_id is None:
            continue
        evidence = None

        if evidence_provider is not None:
            try:
                evidence = evidence_provider(video_graph, clip_id, route)
            except Exception as exc:  # noqa: BLE001 - deliberate soft fallback
                logger.warning(
                    "Evidence provider failed for CLIP_%s (%s): %s",
                    clip_id,
                    route,
                    exc,
                )

        if evidence is None:
            for method_name in ("get_clip_evidence", "rehydrate_clip"):
                method = getattr(video_graph, method_name, None)
                if not callable(method):
                    continue
                try:
                    evidence = method(clip_id, route=route)
                except TypeError:
                    try:
                        evidence = method(clip_id)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("%s failed for CLIP_%s: %s", method_name, clip_id, exc)
                        evidence = None
                except Exception as exc:  # noqa: BLE001
                    logger.debug("%s failed for CLIP_%s: %s", method_name, clip_id, exc)
                    evidence = None
                if evidence is not None:
                    break

        normalized = _normalize_external_evidence(evidence, clip_id, route)
        if normalized:
            output[f"CLIP_{clip_id}"] = normalized

    return output


# ---------------------------------------------------------------------------
# Core VideoGraph retrieval
# ---------------------------------------------------------------------------

def get_related_nodes(video_graph, query):
    related_nodes = []
    entities = parse_video_caption(video_graph, query)
    character_mappings = getattr(video_graph, "character_mappings", {})
    reverse_mappings = getattr(video_graph, "reverse_character_mappings", {})

    for entity in entities:
        entity_type = entity[0]
        node_id = entity[1]
        entity_key = f"{entity_type}_{node_id}"
        if not (entity_key in character_mappings or entity_key in reverse_mappings):
            continue

        if entity_type == "character" and entity_key in character_mappings:
            for node in character_mappings[entity_key]:
                try:
                    related_nodes.append(int(str(node).split("_")[1]))
                except (ValueError, IndexError):
                    continue
        else:
            related_nodes.append(node_id)
    return list(set(related_nodes))


def _existing_clip_ids(video_graph) -> Optional[set]:
    by_clip = getattr(video_graph, "text_nodes_by_clip", None)
    if isinstance(by_clip, dict):
        return set(by_clip.keys())
    return None


def _expand_temporal_neighbors(
    video_graph,
    ranked_clip_ids: Sequence[int],
    clip_scores: Dict[int, float],
    window: int,
    before_clip: Optional[int],
) -> Dict[int, float]:
    if window <= 0 or not ranked_clip_ids:
        return clip_scores

    existing = _existing_clip_ids(video_graph)
    output = dict(clip_scores)
    for center in ranked_clip_ids:
        center_score = _safe_float(output.get(center, 0.0))
        for distance in range(1, window + 1):
            for neighbor in (center - distance, center + distance):
                if neighbor < 0:
                    continue
                if before_clip is not None and neighbor > before_clip:
                    continue
                if existing is not None and neighbor not in existing:
                    continue
                # Temporal neighbors are useful context, but should not outrank
                # the original semantic hit merely because they were expanded.
                decayed_score = center_score * (0.97 ** distance)
                output[neighbor] = max(_safe_float(output.get(neighbor, 0.0)), decayed_score)
    return output


def retrieve_from_videograph(
    video_graph,
    query,
    topk=5,
    mode="max",
    threshold=0,
    before_clip=None,
    route=None,
):
    """Retrieve ranked clips while preserving explicit CLIP_x references."""
    from .local_embedding import validate_graph

    validate_graph(video_graph)

    explicit_clip_ids = []
    for match in re.finditer(r"CLIP_(\d+)", query):
        try:
            clip_id = int(match.group(1))
        except ValueError:
            continue
        if before_clip is None or clip_id <= before_clip:
            explicit_clip_ids.append(clip_id)
    explicit_clip_ids = _dedupe_keep_order(explicit_clip_ids)

    queries = back_translate(video_graph, [query]) or [query]
    if len(queries) > 100:
        logger.warning(
            "Anomaly detected from query: %s, randomly sampling 100 translated queries",
            query,
        )
        queries = random.sample(queries, 100)

    related_nodes = get_related_nodes(video_graph, query)
    query_embeddings = parallel_get_embedding(
        EMBEDDING_MODEL,
        queries,
        input_type="query",
    )[0]

    full_clip_scores: Dict[int, List[float]] = {}
    clip_scores: Dict[int, float] = {}

    if mode not in {"sum", "max", "mean"}:
        raise ValueError(f"Unknown mode: {mode}")

    nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode="max")

    for node_id, node_score in nodes:
        clip_id = video_graph.nodes[node_id].metadata["timestamp"]
        if before_clip is not None and clip_id > before_clip:
            continue
        full_clip_scores.setdefault(clip_id, []).append(_safe_float(node_score))

    for clip_id, scores in full_clip_scores.items():
        if not scores:
            continue
        if mode == "sum":
            clip_score = sum(scores)
        elif mode == "max":
            clip_score = max(scores)
        else:  # mean
            clip_score = float(np.mean(scores))
        clip_scores[clip_id] = float(clip_score)

    route = route or infer_memory_route(query)

    # Temporal/causal questions benefit from neighboring clips, not just the
    # highest lexical/semantic hit.
    if route == "temporal" and TEMPORAL_NEIGHBOR_WINDOW > 0:
        ranked_base = [
            clip_id
            for clip_id, _ in sorted(
                clip_scores.items(), key=lambda item: item[1], reverse=True
            )[: max(topk, 1)]
        ]
        clip_scores = _expand_temporal_neighbors(
            video_graph,
            ranked_base,
            clip_scores,
            TEMPORAL_NEIGHBOR_WINDOW,
            before_clip,
        )

    # Explicit CLIP_x references are treated as hard anchors rather than being
    # accidentally overwritten by semantic ranking as in the original code.
    if explicit_clip_ids:
        current_max = max(clip_scores.values(), default=1.0)
        anchor_score = current_max + 1e-6
        for clip_id in explicit_clip_ids:
            clip_scores[clip_id] = max(_safe_float(clip_scores.get(clip_id)), anchor_score)

    sorted_clips = sorted(clip_scores.items(), key=lambda item: item[1], reverse=True)
    filtered = [
        clip_id
        for clip_id, score in sorted_clips
        if _safe_float(score) >= threshold
        and (before_clip is None or clip_id <= before_clip)
    ]

    # Keep explicit anchors first, then semantic/temporal hits, within top-k.
    top_clips = _dedupe_keep_order(explicit_clip_ids + filtered)[:topk]
    return top_clips, clip_scores, nodes


# ---------------------------------------------------------------------------
# Query generation and utility-aware query selection
# ---------------------------------------------------------------------------

def select_queries(action_content, responses, question=None):
    """
    Select a query by balancing relevance to the user question and novelty
    relative to previous SEARCH actions.

    This replaces the original "lowest average similarity only" policy, which
    could pick a very novel but irrelevant query.
    """
    queries = _normalize_query_list(action_content)
    if not queries:
        return None
    if len(queries) == 1:
        return queries[0]

    history_queries = [
        str(response.get("action_content", "")).strip()
        for response in (responses or [])
        if response.get("action_type") == "search"
        and str(response.get("action_content", "")).strip()
    ]

    try:
        embeddings = parallel_get_embedding(
            EMBEDDING_MODEL,
            queries,
            input_type="query",
        )[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to embed candidate queries; using the first query: %s", exc)
        return queries[0]

    question_embedding = None
    if question:
        try:
            question_embedding = get_embedding_with_retry(
                EMBEDDING_MODEL,
                question,
                input_type="query",
            )[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Question embedding unavailable for query selection: %s", exc)

    history_embeddings = []
    if history_queries:
        try:
            history_embeddings = parallel_get_embedding(
                EMBEDDING_MODEL,
                history_queries,
                input_type="query",
            )[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("History embeddings unavailable for query selection: %s", exc)
            history_embeddings = []

    question_route = infer_memory_route(question or "")
    scores = []
    for query, query_embedding in zip(queries, embeddings):
        relevance = (
            max(-1.0, min(1.0, _safe_cosine(query_embedding, question_embedding)))
            if question_embedding is not None
            else 1.0
        )
        relevance = (relevance + 1.0) / 2.0

        if history_embeddings:
            avg_similarity = float(
                np.mean([
                    _safe_cosine(query_embedding, hist_emb)
                    for hist_emb in history_embeddings
                ])
            )
            novelty = 1.0 - ((avg_similarity + 1.0) / 2.0)
        else:
            novelty = 1.0

        route_bonus = 0.05 if infer_memory_route(query) == question_route else 0.0
        utility = (
            QUERY_RELEVANCE_WEIGHT * relevance
            + QUERY_NOVELTY_WEIGHT * novelty
            + route_bonus
        )
        scores.append(utility)

    best_idx = int(np.argmax(scores))
    logger.debug(
        "Candidate query utilities: %s",
        [(queries[i], round(scores[i], 4)) for i in range(len(queries))],
    )
    return queries[best_idx]


def generate_action(
    question,
    knowledge,
    retrieval_plan=None,
    multiple_queries=False,
    responses=None,
    switch=False,
    model="gpt-4o-2024-11-20",
):
    responses = responses or []

    if not switch:
        prompt = (
            prompt_generate_action_with_plan_multiple_queries
            if multiple_queries
            else prompt_generate_action_with_plan
        )
    else:
        logger.info("Route switch triggered.")
        prompt = (
            prompt_generate_action_with_plan_multiple_queries_new_direction
            if multiple_queries
            else prompt_generate_action_with_plan_new_direction
        )

    input_data = [
        {
            "type": "text",
            "content": prompt.format(
                question=question,
                knowledge=knowledge,
                retrieval_plan=retrieval_plan,
            ),
        }
    ]
    messages = generate_messages(input_data)

    last_action = None
    for attempt in range(MAX_RETRIES):
        action = get_response_with_retry(model, messages)[0]
        last_action = action

        if "[ANSWER]" in action:
            reasoning, action_content = action.split("[ANSWER]", 1)
            action_content = action_content.strip()
            if action_content:
                return _strip_reasoning_labels(reasoning), "answer", action_content

        elif "[SEARCH]" in action:
            reasoning, raw_content = action.split("[SEARCH]", 1)
            raw_content = raw_content.strip()
            if multiple_queries:
                try:
                    parsed = validate_and_fix_python_list(raw_content)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse multi-query action: %s", exc)
                    parsed = [raw_content]
                action_content = select_queries(parsed, responses, question=question)
            else:
                action_content = raw_content

            if action_content:
                return _strip_reasoning_labels(reasoning), "search", action_content

        logger.warning(
            "Malformed/empty agent action on attempt %s/%s: %r",
            attempt + 1,
            MAX_RETRIES,
            action,
        )

    raise ValueError(f"Failed to generate a valid [SEARCH]/[ANSWER] action: {last_action}")


# ---------------------------------------------------------------------------
# Memory materialization and retrieval confidence
# ---------------------------------------------------------------------------

def search_routed(video_graph, query, action, seen_nodes, config, token_count,
                  before_clip=None, threshold=0.45):
    """Executable router policies; legacy search() stays reproducible.

    Returns memories, updated node history, clip scores, and routing details.
    The caller supplies the Control tokenizer for a shared, actual token cap.
    """
    from .routed_retrieval import search_strategy
    return search_strategy(video_graph, query, action, seen_nodes, config, token_count,
                           before_clip=before_clip, threshold=threshold)


def _ordered_nodes_for_route(video_graph, node_ids: Sequence[int], route: str) -> List[int]:
    """Prefer semantic or episodic nodes according to the inferred query route."""
    if route not in {"semantic", "temporal", "visual", "audio", "multimodal"}:
        return list(node_ids)

    def priority(node_id: int) -> int:
        node = video_graph.nodes[node_id]
        node_type = getattr(node, "type", "")
        if route == "semantic":
            return 0 if node_type == "semantic" else 1
        # Specific event/perceptual queries should prefer episodic evidence.
        return 0 if node_type == "episodic" else 1

    return sorted(node_ids, key=priority)


def _memory_contents_for_node(video_graph, node_id: int, route: str) -> List[str]:
    node = video_graph.nodes[node_id]
    metadata = getattr(node, "metadata", {}) or {}
    contents = metadata.get("contents", [])
    if isinstance(contents, str):
        contents = [contents]
    elif not isinstance(contents, (list, tuple)):
        contents = [str(contents)] if contents is not None else []

    translated = translate(video_graph, contents)
    optional_evidence = _extract_node_optional_evidence(node, route)
    return _dedupe_keep_order(translated + optional_evidence)


def estimate_retrieval_confidence(
    new_memories: Dict[str, List[str]],
    clip_scores: Dict[int, float],
) -> Dict[str, float]:
    """
    Lightweight evidence-sufficiency proxy.

    It is intentionally model-free so this file remains drop-in compatible.
    The score combines retrieval strength, evidence amount, and score margin.
    It is not a calibrated probability; it is used only for routing decisions.
    """
    retrieved_clip_ids = [
        clip_id
        for clip_id in (_clip_id_from_key(key) for key in new_memories.keys())
        if clip_id is not None
    ]
    selected_scores = [
        _normalize_similarity(clip_scores.get(clip_id, 0.0))
        for clip_id in retrieved_clip_ids
    ]
    selected_scores.sort(reverse=True)

    top_score = selected_scores[0] if selected_scores else 0.0
    second_score = selected_scores[1] if len(selected_scores) > 1 else 0.0
    margin = max(0.0, min(1.0, top_score - second_score))

    evidence_count = 0
    for memories in new_memories.values():
        if isinstance(memories, (list, tuple)):
            evidence_count += len(memories)
        elif memories:
            evidence_count += 1
    evidence_density = 1.0 - math.exp(-evidence_count / 3.0) if evidence_count else 0.0

    confidence = 0.60 * top_score + 0.25 * evidence_density + 0.15 * margin
    confidence = max(0.0, min(1.0, confidence))

    return {
        "confidence": confidence,
        "top_score": top_score,
        "margin": margin,
        "evidence_density": evidence_density,
        "evidence_count": float(evidence_count),
    }


def search(
    video_graph,
    query,
    current_clips,
    topk=5,
    mode="max",
    threshold=0,
    mem_wise=False,
    before_clip=None,
    episodic_only=False,
    route=None,
):
    route = route or infer_memory_route(query)
    top_clips, clip_scores, nodes = retrieve_from_videograph(
        video_graph,
        query,
        topk,
        mode,
        threshold,
        before_clip,
        route=route,
    )

    if mem_wise:
        new_memories = {}
        top_nodes_num = 0
        for top_node, _ in nodes:
            clip_id = video_graph.nodes[top_node].metadata["timestamp"]
            if before_clip is not None and clip_id > before_clip:
                continue
            node_type = getattr(video_graph.nodes[top_node], "type", "")
            if episodic_only and node_type == "semantic":
                continue

            new_memories.setdefault(clip_id, [])
            new_items = _memory_contents_for_node(video_graph, top_node, route)
            new_memories[clip_id].extend(new_items)
            top_nodes_num += len(new_items)
            if top_nodes_num >= topk:
                break

        new_memories = dict(sorted(new_memories.items(), key=lambda item: item[0]))
        new_memories = {
            f"CLIP_{key}": _dedupe_keep_order(value)
            for key, value in new_memories.items()
            if value
        }
        return new_memories, current_clips, clip_scores

    new_clips = [clip for clip in top_clips if clip not in current_clips]
    new_memories: Dict[Any, List[str]] = {}
    current_clips.extend(new_clips)

    text_nodes_by_clip = getattr(video_graph, "text_nodes_by_clip", {})
    for new_clip in new_clips:
        if new_clip not in text_nodes_by_clip:
            new_memories[new_clip] = [
                f"CLIP_{new_clip} not found in memory bank, please search for other information"
            ]
            continue

        related_nodes = _ordered_nodes_for_route(
            video_graph,
            text_nodes_by_clip[new_clip],
            route,
        )
        clip_memories = []
        for node_id in related_nodes:
            node_type = getattr(video_graph.nodes[node_id], "type", "")
            if episodic_only and node_type == "semantic":
                continue
            clip_memories.extend(_memory_contents_for_node(video_graph, node_id, route))

        new_memories[new_clip] = _dedupe_keep_order(clip_memories)

    new_memories = dict(sorted(new_memories.items(), key=lambda item: item[0]))
    new_memories = {f"CLIP_{key}": value for key, value in new_memories.items()}
    return new_memories, current_clips, clip_scores


# ---------------------------------------------------------------------------
# Main agent loop: uncertainty-guided active retrieval
# ---------------------------------------------------------------------------

def _merge_rehydrated_context(
    new_memories: Dict[str, List[str]],
    rehydrated: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    if not rehydrated:
        return new_memories
    merged = {key: list(value) for key, value in new_memories.items()}
    for key, items in rehydrated.items():
        merged.setdefault(key, [])
        merged[key].extend(items)
        merged[key] = _dedupe_keep_order(merged[key])
    return merged


def _force_final_answer(question, context, model):
    input_data = [
        {
            "type": "text",
            "content": prompt_answer_with_retrieval_final.format(
                question=question,
                information=context,
            ),
        }
    ]
    messages = generate_messages(input_data)
    resp = get_response_with_retry(model, messages)[0]
    if "[ANSWER]" in resp:
        reasoning, final_answer = resp.split("[ANSWER]", 1)
        return _strip_reasoning_labels(reasoning), final_answer.strip()
    # Be robust to a final model response that omits the marker.
    return "", resp.strip()


def answer_with_retrieval(
    video_graph,
    question,
    video_clip_base64=None,
    topk=5,
    auto_refresh=False,
    mode="max",
    multiple_queries=False,
    max_retrieval_steps=10,
    route_switch=True,
    threshold=0,
    model="gpt-4o-2024-11-20",
    before_clip=None,
    enable_active_retrieval=True,
    evidence_provider=None,
):
    """
    Backward-compatible main QA API with active multimodal memory retrieval.

    New optional arguments are appended at the end so existing callers remain
    valid:
      enable_active_retrieval: enables uncertainty-guided route switching.
      evidence_provider: optional callable(video_graph, clip_id, route) used for
                         on-demand raw/richer evidence retrieval.
    """
    if before_clip is not None:
        video_graph.truncate_memory_by_clip(before_clip)

    if auto_refresh:
        video_graph.refresh_equivalences()

    related_clips = []
    context = []
    final_answer = None
    memories = [[]]
    responses = []

    if video_clip_base64 is not None:
        input_data = [
            {
                "type": "video_base64/mp4",
                "content": video_clip_base64,
            },
            {
                "type": "text",
                "content": prompt_generate_plan.format(question=question),
            },
        ]
        messages = generate_messages(input_data)
        plan_model = processing_config.get("plan_model", "gemini-1.5-pro-002")
        retrieval_plan = get_response_with_retry(plan_model, messages)[0]
        logger.info("Retrieval plan: %s", retrieval_plan)
    else:
        retrieval_plan = None

    base_route = infer_memory_route(question, retrieval_plan)
    switch = False
    low_confidence_streak = 0

    for i in range(max_retrieval_steps):
        reasoning, action_type, action_content = generate_action(
            question,
            context,
            retrieval_plan,
            multiple_queries=multiple_queries,
            responses=responses,
            switch=switch,
            model=model,
        )
        reasoning = _strip_reasoning_labels(reasoning)

        if action_type == "answer":
            final_answer = action_content
            responses.append(
                {
                    "reasoning": reasoning,
                    "action_type": action_type,
                    "action_content": action_content,
                    "memory_route": base_route,
                }
            )
            logger.info("Answer: %s", final_answer)
            break

        if action_type != "search":
            raise ValueError(f"Unknown action type: {action_type}")

        # Preserve the original budget semantics: the final step is reserved for
        # a forced answer rather than performing an extra retrieval that cannot
        # be consumed by another reasoning turn.
        if i == max_retrieval_steps - 1:
            reasoning, final_answer = _force_final_answer(question, context, model)
            responses.append(
                {
                    "reasoning": reasoning,
                    "action_type": "answer",
                    "action_content": final_answer,
                    "completion_mode": "forced",
                    "memory_route": base_route,
                }
            )
            logger.info("Forced answer: %s", final_answer)
            break

        query_route = infer_memory_route(action_content, retrieval_plan)
        # If the generated query is generic, retain the original question route.
        if query_route == "text" and base_route != "text":
            query_route = base_route

        new_memories, related_clips, clip_scores = search(
            video_graph,
            action_content,
            related_clips,
            topk,
            mode,
            threshold=threshold,
            before_clip=before_clip,
            route=query_route,
        )

        retrieval_meta = estimate_retrieval_confidence(new_memories, clip_scores)
        confidence = retrieval_meta["confidence"]

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence_streak += 1
        else:
            low_confidence_streak = 0

        # On low-confidence visual/audio/temporal retrieval, try to rehydrate the
        # exact retrieved clips from raw/richer evidence if the project exposes
        # such an interface. Otherwise this is a safe no-op.
        rehydrated = {}
        if enable_active_retrieval and confidence < LOW_CONFIDENCE_THRESHOLD:
            retrieved_clip_ids = [
                clip_id
                for clip_id in (_clip_id_from_key(key) for key in new_memories.keys())
                if clip_id is not None
            ]
            rehydrated = rehydrate_evidence(
                video_graph,
                retrieved_clip_ids,
                query_route,
                evidence_provider=evidence_provider,
            )
            if rehydrated:
                new_memories = _merge_rehydrated_context(new_memories, rehydrated)
                # Evidence count changed; recompute the heuristic confidence.
                retrieval_meta = estimate_retrieval_confidence(new_memories, clip_scores)
                confidence = retrieval_meta["confidence"]
                if confidence >= LOW_CONFIDENCE_THRESHOLD:
                    low_confidence_streak = 0

        no_new_evidence = len(new_memories) == 0
        low_utility = low_confidence_streak >= max(1, LOW_CONFIDENCE_PATIENCE)

        if route_switch and (no_new_evidence or (enable_active_retrieval and low_utility)):
            switch = True
        else:
            switch = False

        context.append(
            {
                "reasoning": reasoning,
                "query": action_content,
                "memory_route": query_route,
                "retrieval_confidence": round(confidence, 4),
                "retrieval_meta": {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in retrieval_meta.items()
                },
                "rehydrated": bool(rehydrated),
                "retrieved memories": new_memories,
            }
        )

        new_response_item = {
            "reasoning": reasoning,
            "action_type": action_type,
            "action_content": action_content,
            "memory_route": query_route,
            "retrieval_confidence": round(confidence, 4),
            "route_switch_next": switch,
        }
        responses.append(new_response_item)

        new_memory_items = [
            {
                "clip_id": key,
                "memory": value,
            }
            for key, value in new_memories.items()
        ]
        memories.append(new_memory_items)

        if processing_config.get("logging") == "DETAIL":
            logger.debug("%s", "=" * 10 + f"Retrieval Step {i + 1}" + "=" * 10)
            logger.debug(new_response_item)
            logger.debug(new_memory_items)
            logger.debug("Retrieval meta: %s", retrieval_meta)

    return final_answer, (memories, responses)


# ---------------------------------------------------------------------------
# Evaluation / utility helpers
# ---------------------------------------------------------------------------

def verify_qa(question, gt, pred, model="gpt-4o-2024-11-20"):
    try:
        input_data = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=gt,
                    agent_answer=pred,
                ),
            }
        ]
        messages = generate_messages(input_data)
        response = get_response_with_retry(model, messages)
        result = response[0]
    except Exception as exc:  # noqa: BLE001
        logger.error("Error verifying qa: %s", question)
        logger.error("%s", exc)
        return None
    return result


def calculate_similarity(mem, query, related_nodes):
    from .local_embedding import validate_graph

    validate_graph(mem)
    if not related_nodes:
        return []
    related_nodes_embeddings = np.array(
        [mem.nodes[node_id].embeddings[0] for node_id in related_nodes]
    )
    query_embedding = np.array(
        get_embedding_with_retry(
            EMBEDDING_MODEL,
            query,
            input_type="query",
        )[0]
    ).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, related_nodes_embeddings)[0]
    return similarities.tolist()


def retrieve_all_episodic_memories(video_graph):
    episodic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "episodic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            episodic_memories.setdefault(clips_id, [])
            episodic_memories[clips_id].extend(
                video_graph.nodes[node_id].metadata["contents"]
            )
    return episodic_memories


def retrieve_all_semantic_memories(video_graph):
    semantic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "semantic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            semantic_memories.setdefault(clips_id, [])
            semantic_memories[clips_id].extend(
                video_graph.nodes[node_id].metadata["contents"]
            )
    return semantic_memories


if __name__ == "__main__":
    # Kept for parity with the original developer test. In a package with
    # relative imports, prefer running through the package entry point/module.
    from utils.general import load_video_graph
    import base64

    processing_config["logging"] = "DETAIL"
    processing_config["topk"] = 30

    def video_to_base64(video_path):
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            return base64.b64encode(video_bytes).decode("utf-8")

    video_graph_path = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems/CZ_1/Efk3K4epEzg_30_5_-1_10_20_0.3_0.6.pkl"
    video_graph = load_video_graph(video_graph_path)

    question = "Which collection has the highest starting price?"
    answer = answer_with_retrieval(
        video_graph,
        question,
        video_to_base64(
            "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_1/Efk3K4epEzg/39.mp4"
        ),
        topk=processing_config["topk"],
        multiple_queries=processing_config["multiple_queries"],
        max_retrieval_steps=processing_config["max_retrieval_steps"],
    )
