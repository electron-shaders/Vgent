"""
lmms_eval_async_openai.py — VGent model backend that delegates to a vLLM
OpenAI-compatible server via AsyncOpenAI client.

Drop-in replacement for models/qwenvl.py: exposes the same three callables
(load_video, load_model, mllm_response) consumed by utils/vgent.py.

Graph construction calls (mllm_response) are routed through the AsyncOpenAI
client that lmms-eval manages, so all requests share the same HTTP connection
pool and benefit from vLLM's continuous-batching scheduler natively.
"""

import asyncio

import numpy as np
import torch
from PIL import Image
from models.utils import fetch_video, resize_video
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import openai
import logging

_log = logging.getLogger(__name__)

# Module-level references set by vgent_adapter.init_vgent_instance()
openai_client = None   # openai.AsyncOpenAI instance (used only to read config)
model_version = None   # e.g. "Qwen/Qwen3.5-4B"
_base_url = None       # e.g. "http://localhost:8000/v1"
_api_key = None        # API key string


def load_video(video_path, args):
    """Load and resize video for VGent chunk processing."""
    raw_video, frame_idx, fps = fetch_video({"video": video_path, "fps": args.fps}, resize=False)
    video, fps = resize_video(
        raw_video,
        fps,
        total_pixels=args.total_pixels
        * max(1, int(round(np.ceil(len(raw_video) / args.chunk_size))))
        * 28
        * 28,
    )
    # construct_graph calls torch.split() on the video tensor
    video_tensor = torch.as_tensor(np.array(video))
    return [raw_video], None, None, frame_idx, fps, [video_tensor], None


def load_model(model_name=""):
    """No-op: the model is served remotely; only metadata is needed."""
    return None, None, None, None


def _frames_to_openai_content(video):
    """Convert a video chunk tensor to a list of base64 image_url content items."""
    import os
    from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

    if torch.is_tensor(video):
        video_np = video.cpu().numpy()
    else:
        video_np = np.array(video)

    # (T, C, H, W) → (T, H, W, C)
    if video_np.ndim == 4 and video_np.shape[1] == 3:
        video_np = np.transpose(video_np, (0, 2, 3, 1))
    if video_np.max() <= 1.0:
        video_np = (video_np * 255.0)
    video_np = video_np.astype(np.uint8)

    image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
    mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
    quality = (
        int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85"))
        if image_format in {"JPEG", "JPG", "WEBP"}
        else None
    )

    content = []
    for frame in video_np:
        image = Image.fromarray(frame)
        b64 = encode_image_to_base64(image, image_format=image_format, quality=quality)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
    return content


async def _call_api_with_retry(messages, max_new_tokens):
    """
    Call the OpenAI-compatible API with exponential backoff.
    Only the network call is retried; content preparation is excluded.

    A fresh AsyncOpenAI client is created for each asyncio.run() invocation
    to avoid "Event loop is closed" errors that occur when reusing a client
    whose internal httpx connection pool was bound to a different event loop.
    """
    @retry(
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )
    async def _do_call():
        # Create a fresh client bound to the current event loop so httpx
        # cleanup never touches a closed loop from a prior asyncio.run().
        async with openai.AsyncOpenAI(
            base_url=_base_url,
            api_key=_api_key,
        ) as client:
            return await client.chat.completions.create(
                model=model_version,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )

    response = await _do_call()
    return response.choices[0].message.content or ""


async def _async_mllm_response(text, video, max_new_tokens):
    """Prepare content (once), then call the API with retry."""
    if openai_client is None:
        raise ValueError(
            "[lmms_eval_async_openai] openai_client is not set. "
            "Call vgent_adapter.init_vgent_instance() with an openai_client before use."
        )

    # ── Content preparation (not retried) ──────────────────────────────────
    content = []
    if video is not None:
        content.extend(_frames_to_openai_content(video))
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]

    # ── Retried API call ───────────────────────────────────────────────────
    return await _call_api_with_retry(messages, max_new_tokens)


def mllm_response(
    video_llm,
    tokenizer,
    processor,
    text,
    image_inputs,
    video,
    max_new_tokens=512,
    size_list=None,
    fps=None,
):
    """
    Synchronous wrapper — VGent's graph builder is synchronous, so we run the
    coroutine with asyncio.run().  Because each call is independent and vLLM
    serves them concurrently via HTTP, this achieves the same throughput as
    the async path while keeping VGent's synchronous call-sites unchanged.
    """
    try:
        return asyncio.run(_async_mllm_response(text, video, max_new_tokens))
    except Exception:
        import traceback
        traceback.print_exc()
        return ""
