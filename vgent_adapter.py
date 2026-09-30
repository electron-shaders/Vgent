"""
vgent_adapter.py — adapter between lmms-eval and Vgent internals.

This file provides the programmatic bridge that lmms-eval needs, 
bypassing Vgent's CLI-only scripts (vgent_graph.py, vgent_rag.py) 
by interacting directly with the `Vgent` class and `utils`.
"""

import argparse
import os
import pickle
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Lazy-load Vgent internals
# ---------------------------------------------------------------------------

_embedding_model = None
_embedding_tokenizer = None
_vgent_instance = None

_vgent_init_lock = threading.Lock()
_embed_init_lock = threading.Lock()

def _lazy_init_embeddings():
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is not None:
        return
        
    with _embed_init_lock:
        if _embedding_model is None:
            from transformers import AutoModel, AutoTokenizer
            _embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
            _embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
            _embedding_model.eval()

@dataclass
class _VideoArtifacts:
    raw_video: Any
    fps: float
    video_inputs: list
    size_list: Any
    subtitles: Any
    video_graph: Any
    entity_graph: Any


@dataclass
class _VideoCacheEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0
    artifacts: _VideoArtifacts | None = None


# Entries live only while at least one query for a video is active. This gives
# overlapping questions a single video decode and graph build without retaining
# every dataset video in RAM for the duration of the evaluation.
_video_cache: dict[str, _VideoCacheEntry] = {}
_video_cache_lock = threading.Lock()


def _save_graph_atomic(graph_path, video_graph, entity_graph):
    graph_dir = os.path.dirname(graph_path)
    fd, temp_path = tempfile.mkstemp(prefix=".graph-", suffix=".pkl", dir=graph_dir)
    try:
        with os.fdopen(fd, "wb") as graph_file:
            pickle.dump(
                {"video_graph": video_graph, "entity_graph": entity_graph},
                graph_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temp_path, graph_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _load_or_build_artifacts(vgent, video_path, subtitle_path, graph_path):
    vgent.raise_if_cancelled()
    args = vgent.args
    raw_video, _, _, _, fps, video_inputs, size_list = vgent.load_video(video_path, args)
    vgent.raise_if_cancelled()
    if "llava_video" in args.model_name:
        video = vgent.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
        video_inputs = [video]
    if not isinstance(video_inputs, list):
        video_inputs = [video_inputs]

    subtitles = None
    if subtitle_path is not None and os.path.exists(subtitle_path):
        from utils.data import get_subtitles

        subtitles = get_subtitles(subtitle_path, None, None, None)

    try:
        with open(graph_path, "rb") as graph_file:
            saved_graph = pickle.load(graph_file)
        video_graph = saved_graph["video_graph"]
        entity_graph = saved_graph["entity_graph"]
    except Exception as exc:
        if os.path.exists(graph_path):
            warnings.warn(
                f"[vgent_adapter] Rebuilding unreadable graph '{graph_path}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        video_graph, entity_graph = vgent.construct_graph(video_inputs, subtitles)
        vgent.raise_if_cancelled()
        _save_graph_atomic(graph_path, video_graph, entity_graph)

    vgent.raise_if_cancelled()
    return _VideoArtifacts(
        raw_video=raw_video,
        fps=fps,
        video_inputs=video_inputs,
        size_list=size_list,
        subtitles=subtitles,
        video_graph=video_graph,
        entity_graph=entity_graph,
    )


def _acquire_video_artifacts(cache_key, vgent, video_path, subtitle_path, graph_path):
    vgent.raise_if_cancelled()
    with _video_cache_lock:
        entry = _video_cache.get(cache_key)
        if entry is None:
            entry = _VideoCacheEntry()
            _video_cache[cache_key] = entry
        entry.users += 1

    try:
        # A query waiting for another query's decode/build must also be able
        # to leave promptly when evaluation is cancelled.
        while not entry.lock.acquire(timeout=0.1):
            vgent.raise_if_cancelled()
        try:
            vgent.raise_if_cancelled()
            if entry.artifacts is None:
                entry.artifacts = _load_or_build_artifacts(
                    vgent,
                    video_path,
                    subtitle_path,
                    graph_path,
                )
            return entry, entry.artifacts
        finally:
            entry.lock.release()
    except BaseException:
        _release_video_artifacts(cache_key, entry)
        raise


def _release_video_artifacts(cache_key, entry):
    with _video_cache_lock:
        entry.users -= 1
        if entry.users == 0 and _video_cache.get(cache_key) is entry:
            _video_cache.pop(cache_key, None)
            entry.artifacts = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_vgent_instance(
    model_name: str,
    task: str,
    openai_client=None,
    openai_model_version: str = None,
    batch_size: int | None = None,
    openai_timeout: int = 600,
    graph_max_new_tokens: int | None = None,
    graph_max_new_tokens_limit: int | None = None,
):
    """
    Initialize the singleton Vgent instance on the main thread.

    Parameters
    ----------
    model_name : str
        Key into Vgent's MODEL_MAP (e.g. "lmms_eval_async_openai").
    task : str
        Task name passed to Vgent args.
    openai_client : openai.AsyncOpenAI, optional
        If provided, its server configuration is used to initialize Vgent's
        persistent OpenAI client.
    openai_model_version : str, optional
        Model name to use with the OpenAI client (e.g. "Qwen/Qwen3.5-4B").
    graph_max_new_tokens, graph_max_new_tokens_limit : int, optional
        Initial and maximum graph-chunk output budgets. When omitted, use
        VGENT_GRAPH_MAX_NEW_TOKENS / VGENT_GRAPH_MAX_NEW_TOKENS_LIMIT, or 2048 / 16384.
    """
    global _vgent_instance

    # Query-time calls omit batch_size and reuse the already configured
    # singleton. The initial standalone/non-API path remains serial by default.
    if batch_size is None and _vgent_instance is not None:
        return _vgent_instance
    batch_size = 1 if batch_size is None else int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    if _vgent_instance is not None:
        if int(getattr(_vgent_instance.args, "batch_size", 1)) != batch_size:
            raise RuntimeError(
                "Vgent is already initialized with a different batch_size; "
                "shut it down before reconfiguring concurrency."
            )
        return _vgent_instance

    if openai_client is not None:
        import models.lmms_eval_async_openai as _m

        _m.configure_openai_runtime(
            base_url=openai_client.base_url,
            api_key=openai_client.api_key,
            model=openai_model_version,
            concurrency=batch_size,
            timeout=openai_timeout,
        )

    with _vgent_init_lock:
        if _vgent_instance is not None:
            return _vgent_instance

        from utils.vgent import Vgent

        # Ensure model_name matches a valid key in Vgent's MODEL_MAP
        valid_keys = [
            "llava_video", "lmms_eval_async_openai",
            "qwenvl25_7b", "qwenvl25_3b", "qwenvl2_7b", "qwenvl2_2b",
            "internvl25_2b", "longvu",
        ]
        if not any(k in model_name for k in valid_keys):
            print(f"[vgent_adapter] Model '{model_name}' not in MODEL_MAP. Falling back to 'lmms_eval_async_openai'.")
            model_name = "lmms_eval_async_openai"

        if openai_client is not None:
            if "lmms_eval_async_openai" not in model_name:
                print(
                    f"[vgent_adapter] openai_client provided — overriding model "
                    f"'{model_name}' to 'lmms_eval_async_openai'."
                )
                model_name = "lmms_eval_async_openai"

        args = argparse.Namespace(
            model_name=model_name,
            chunk_size=64,
            task=task,
            uniform_frame=450,
            n_retrieval=20,
            n_refine=5,
            total_pixels=16384,
            fps=1.0,
            batch_size=batch_size,
            graph_max_new_tokens=graph_max_new_tokens,
            graph_max_new_tokens_limit=graph_max_new_tokens_limit,
        )
        _vgent_instance = Vgent(args)
        return _vgent_instance


def cancel_vgent_instance():
    """Non-blocking abort signal, safe before asyncio joins query workers.

    Keep the cancelled singleton installed until those workers have exited;
    otherwise an already queued query could initialize a fresh graph builder.
    """
    try:
        if _vgent_instance is not None:
            _vgent_instance.cancel()
    finally:
        # A native/local-model run may never have imported the API backend.
        backend = sys.modules.get("models.lmms_eval_async_openai")
        if backend is not None:
            backend.cancel_openai_runtime()


def shutdown_vgent_instance():
    """Release Vgent workers, cached videos, and the persistent API client."""
    global _vgent_instance
    with _vgent_init_lock:
        try:
            cancel_vgent_instance()
        finally:
            try:
                # Release API waiters BEFORE joining the graph pool.
                backend = sys.modules.get("models.lmms_eval_async_openai")
                if backend is not None:
                    backend.shutdown_openai_runtime()
            finally:
                try:
                    if _vgent_instance is not None:
                        _vgent_instance.close()
                finally:
                    _vgent_instance = None
                    with _video_cache_lock:
                        _video_cache.clear()

def run_vgent_query(video_id: str, query: str, video_path: str, output_dir: str, question: str, candidates: list[str], doc: dict, subtitle_path: str | None = None, model_name: str = "qwenvl25_7b", task: str = "custom", generation_kwargs: dict | None = None, return_raw_response: bool = False) -> str:
    """
    Run Vgent text-based retrieval for one query.
    Extracts the textual context of the top-k clips from the pre-built graph.
    Set return_raw_response for lmms-eval: only its task parser grades the final
    model output; Vgent's final-answer parsing becomes diagnostic only.
    """
    vgent = init_vgent_instance(model_name, task)
    vgent.raise_if_cancelled()
    args = vgent.args
    _lazy_init_embeddings()
    vgent.raise_if_cancelled()

    prompt = f"Question: {question}\n"
    prompt += "Options:\n"
    for op in candidates:
        prompt += f"{op}\n"

    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, "graph.pkl")
    cache_key = os.path.abspath(graph_path)
    cache_entry, artifacts = _acquire_video_artifacts(
        cache_key,
        vgent,
        video_path,
        subtitle_path,
        graph_path,
    )

    try:
        raw_video = artifacts.raw_video
        fps = artifacts.fps
        video_inputs = artifacts.video_inputs
        size_list = artifacts.size_list
        subtitles = artifacts.subtitles
        video_graph = artifacts.video_graph
        entity_graph = artifacts.entity_graph
        vgent.raise_if_cancelled()
        query_list, llm_info = vgent.extract_keywords(question, candidates, video_inputs)
        vgent.raise_if_cancelled()
        retrieved_node_list = vgent.retrieve_nodes(question, query_list, video_inputs, candidates, video_graph, entity_graph, subtitles, llm_info)
        vgent.raise_if_cancelled()
        refined_node_list, sql_check, check_result = vgent.refine_nodes(retrieved_node_list, question, llm_info, candidates, video_inputs, subtitles, size_list)
        vgent.raise_if_cancelled()
        pred = vgent.aggregate_nodes(refined_node_list, llm_info, video_inputs, raw_video, size_list, subtitles, prompt, doc, video_graph, sql_check, check_result, fps, generation_kwargs=generation_kwargs, return_raw_response=return_raw_response)
        vgent.raise_if_cancelled()
    finally:
        _release_video_artifacts(cache_key, cache_entry)

    return pred
