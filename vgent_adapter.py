"""
vgent_adapter.py — adapter between lmms-eval and Vgent internals.

This file provides the programmatic bridge that lmms-eval needs, 
bypassing Vgent's CLI-only scripts (vgent_graph.py, vgent_rag.py) 
by interacting directly with the `Vgent` class and `utils`.
"""

import argparse
import os
import pickle
import threading
import warnings

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

# Per-video build lock: prevents concurrent coroutines from building the same
# graph twice when the same video appears in multiple requests.
_jit_build_locks: dict[str, threading.Lock] = {}
_jit_build_locks_meta = threading.Lock()


def _get_jit_lock(key: str) -> threading.Lock:
    with _jit_build_locks_meta:
        if key not in _jit_build_locks:
            _jit_build_locks[key] = threading.Lock()
        return _jit_build_locks[key]

def _build_vgent_graph(video_id, video_inputs, subtitles, graph_path):
    with _get_jit_lock(video_id):
        if os.path.exists(graph_path):
            return None, None
        else:
            video_graph, entity_graph = _vgent_instance.construct_graph(video_inputs, subtitles)
            with open(graph_path, 'wb') as f:
                pickle.dump({"video_graph": video_graph, "entity_graph": entity_graph}, f)
    return video_graph, entity_graph

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_vgent_instance(model_name: str, task: str, openai_client=None, openai_model_version: str = None):
    """
    Initialize the singleton Vgent instance on the main thread.

    Parameters
    ----------
    model_name : str
        Key into Vgent's MODEL_MAP (e.g. "lmms_eval_async_openai").
    task : str
        Task name passed to Vgent args.
    openai_client : openai.AsyncOpenAI, optional
        If provided, injected into models.lmms_eval_async_openai so that
        graph-construction VLM calls are routed through the already-running
        vLLM OpenAI-compatible server.
    openai_model_version : str, optional
        Model name to use with the OpenAI client (e.g. "Qwen/Qwen3.5-4B").
    """
    global _vgent_instance

    if _vgent_instance is not None:
        return _vgent_instance

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
            import models.lmms_eval_async_openai as _m
            _m.openai_client = openai_client
            _m.model_version = openai_model_version
            # Store connection config so each asyncio.run() can create a fresh
            # client bound to its own event loop (avoids "Event loop is closed").
            _m._base_url = str(openai_client.base_url)
            _m._api_key = openai_client.api_key
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
        )
        _vgent_instance = Vgent(args)
        return _vgent_instance

def run_vgent_query(video_id: str, query: str, video_path: str, output_dir: str, question: str, candidates: list[str], doc: dict, subtitle_path: str | None = None, model_name: str = "qwenvl25_7b", task: str = "custom") -> str:
    """
    Run Vgent text-based retrieval for one query.
    Extracts the textual context of the top-k clips from the pre-built graph.
    """
    vgent = init_vgent_instance(model_name, task)
    args = vgent.args
    _lazy_init_embeddings()

    prompt = f"Question: {question}\n"
    prompt += "Options:\n"
    for op in candidates:
        prompt += f"{op}\n"

    raw_video, _, _, frame_idx, fps, video_inputs, size_list = vgent.load_video(video_path, args)
    if "llava_video" in args.model_name:
        video = vgent.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
        video_inputs = [video]
    if type(video_inputs) is not list:
        video_inputs = [video_inputs]

    if subtitle_path is not None:
        from utils.data import get_subtitles
        subtitles = get_subtitles(subtitle_path, None, None, None)

    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, "graph.pkl")
    if os.path.exists(graph_path):
        try:
            with open(graph_path, 'rb') as f:
                saved_graph = pickle.load(f)
                video_graph = saved_graph["video_graph"]
                entity_graph = saved_graph["entity_graph"]
        except Exception:
            video_graph, entity_graph = _build_vgent_graph(video_id, video_inputs, subtitles, graph_path)
            if video_graph is None or entity_graph is None:
                with open(graph_path, 'rb') as f:
                    saved_graph = pickle.load(f)
                    video_graph = saved_graph["video_graph"]
                    entity_graph = saved_graph["entity_graph"]
    else:
        video_graph, entity_graph = _build_vgent_graph(video_id, video_inputs, subtitles, graph_path)
        if video_graph is None or entity_graph is None:
            with open(graph_path, 'rb') as f:
                saved_graph = pickle.load(f)
                video_graph = saved_graph["video_graph"]
                entity_graph = saved_graph["entity_graph"]

    try:
        query_list, llm_info = vgent.extract_keywords(question, candidates, video_inputs)
        retrieved_node_list = vgent.retrieve_nodes(question, query_list, video_inputs, candidates, video_graph, entity_graph, subtitles, llm_info)
        refined_node_list, sql_check, check_result = vgent.refine_nodes(retrieved_node_list, question, llm_info, candidates, video_inputs, subtitles, size_list)
        pred = vgent.aggregate_nodes(refined_node_list, llm_info, video_inputs, raw_video, size_list, subtitles, prompt, doc, video_graph, sql_check, check_result, fps)
    except Exception as exc:
        warnings.warn(f"[vgent_adapter] Query failed: {exc}", RuntimeWarning, stacklevel=2)
        return ""

    return pred