"""
vgent_adapter.py — adapter between lmms-eval and Vgent internals.

This file provides the programmatic bridge that lmms-eval needs, 
bypassing Vgent's CLI-only scripts (vgent_graph.py, vgent_rag.py) 
by interacting directly with the `Vgent` class and `utils`.
"""

import os
import pickle
import torch
import argparse
import warnings
import threading

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

        # Ensure model_name matches a valid key in VGent's MODEL_MAP
        valid_keys = [
            "llava_video", "lmms_eval_async_openai",
            "qwenvl25_7b", "qwenvl25_3b", "qwenvl2_7b", "qwenvl2_2b",
            "internvl25_2b", "longvu",
        ]
        if not any(k in model_name for k in valid_keys):
            print(f"[vgent_adapter] Model '{model_name}' not in MODEL_MAP. Falling back to 'qwenvl25_7b'.")
            model_name = "qwenvl25_7b"

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
            n_retrieval=20,
            n_refine=5,
            total_pixels=16384,
            fps=1.0,
        )
        _vgent_instance = Vgent(args)
        return _vgent_instance

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_vgent_query(graph_dir: str, query: str, max_clips: int = 8) -> str:
    """
    Run VGent text-based retrieval for one query.
    Extracts the textual context of the top-k clips from the pre-built graph.
    """
    from utils.retrieval import compute_text_similarity, allocate_node
    
    _lazy_init_embeddings()

    # Vgent saves graphs as .pkl files.
    pkl_file = None
    if os.path.isfile(graph_dir):
        pkl_file = graph_dir
    elif os.path.isdir(graph_dir):
        for f in os.listdir(graph_dir):
            if f.endswith('.pkl'):
                pkl_file = os.path.join(graph_dir, f)
                break

    if not pkl_file or not os.path.isfile(pkl_file):
        return ""

    try:
        with open(pkl_file, 'rb') as f:
            saved_graph = pickle.load(f)
            video_graph = saved_graph["video_graph"]
            entity_graph = saved_graph["entity_graph"]

        args = argparse.Namespace(chunk_size=64, n_retrieval=max_clips, fps=1.0)
        query_list = [query]

        node_list = allocate_node(args, video_graph, entity_graph, query_list, _embedding_model, _embedding_tokenizer)
        if not node_list:
            return ""

        key_list = []
        for node_id in node_list:
            node_data = video_graph.nodes[node_id]
            desc = "; ".join(node_data.get('entities', [])) + "; " + \
                   "; ".join(node_data.get('actions', [])) + "; " + \
                   "; ".join(node_data.get('scenes', []))
            if node_data.get('subtitles'):
                desc += "; " + "; ".join(node_data.get('subtitles', []))
            key_list.append(desc)

        sims = compute_text_similarity(query_list, key_list, _embedding_model, _embedding_tokenizer, return_all=True)
        sorted_indices = torch.argsort(torch.mean(sims, dim=0), descending=True)
        top_nodes = [node_list[i] for i in sorted_indices][:max_clips]

        context_parts = []
        for node in top_nodes:
            data = video_graph.nodes[node]
            desc = f"[Clip {node}]"
            entities = data.get('entities', [])
            actions = data.get('actions', [])
            scenes = data.get('scenes', [])
            if entities: desc += f"\nEntities: {', '.join(entities)}"
            if actions: desc += f"\nActions: {', '.join(actions)}"
            if scenes: desc += f"\nScenes: {', '.join(scenes)}"
            if data.get('subtitles'): desc += f"\nSubtitles: {', '.join(data.get('subtitles', []))}"
            context_parts.append(desc)

        return "\n\n".join(context_parts)
    except Exception as exc:
        warnings.warn(f"[vgent_adapter] Query failed: {exc}", RuntimeWarning, stacklevel=2)
        return ""


def build_graph_for_video(video_path: str, output_dir: str, model_name: str = "qwenvl25_7b", task: str = "custom") -> str:
    """
    Build a VGent knowledge graph on-the-fly by importing the core Vgent class.
    """
    vgent = init_vgent_instance(model_name, task)
    args = vgent.args

    raw_video, _, _, frame_idx, fps, video_inputs, size_list = vgent.load_video(video_path, args)
    if "llava_video" in args.model_name:
        video = vgent.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
        video_inputs = [video]
    if type(video_inputs) is not list:
        video_inputs = [video_inputs]

    video_graph, entity_graph = vgent.construct_graph(video_inputs, subtitles=None)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "graph.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump({"video_graph": video_graph, "entity_graph": entity_graph}, f)

    return output_dir
