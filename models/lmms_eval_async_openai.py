"""
lmms_eval_async_openai.py — Vgent model backend that delegates to a vLLM
OpenAI-compatible server via AsyncOpenAI client.

Drop-in replacement for models/qwenvl.py: exposes the same three callables
(load_video, load_model, mllm_response) consumed by utils/vgent.py.

Graph construction calls (mllm_response) are routed through a persistent
AsyncOpenAI client configured by lmms-eval, so all requests share the same HTTP
connection pool and benefit from vLLM's continuous-batching scheduler natively.
"""

import asyncio
import logging
import threading

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

_log = logging.getLogger(__name__)

# The adapter configures one runtime for the evaluator process.  All Vgent
# callers submit work to this loop, so the client, connection pool, and
# semaphore are shared.
_runtime = None
_runtime_lock = threading.Lock()


class _PersistentOpenAIRuntime:
    """Own a persistent AsyncOpenAI client on a dedicated event-loop thread."""

    def __init__(self, base_url, api_key, model, concurrency, timeout):
        self.base_url = str(base_url)
        self.api_key = api_key
        self.model = model
        self.concurrency = int(concurrency)
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error = None
        self._client = None
        self._semaphore = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vgent-openai-runtime",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("Failed to initialize the Vgent OpenAI runtime") from self._startup_error

    @property
    def config(self):
        return (self.base_url, self.api_key, self.model, self.concurrency, self.timeout)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._client = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self._semaphore = asyncio.Semaphore(self.concurrency)
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._ready.set()

        if self._startup_error is not None:
            self._loop.close()
            return

        self._loop.run_forever()
        self._loop.run_until_complete(self._client.close())
        self._loop.close()

    async def _call_api_with_retry(self, messages, max_new_tokens):
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
            # Acquire for one HTTP attempt only.  Exceptions and cancellations
            # release the permit before tenacity performs its retry backoff.
            async with self._semaphore:
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )

        response = await _do_call()
        return response.choices[0].message.content or ""

    def request(self, messages, max_new_tokens):
        if self._closed:
            raise RuntimeError("The Vgent OpenAI runtime is closed")
        future = asyncio.run_coroutine_threadsafe(
            self._call_api_with_retry(messages, max_new_tokens),
            self._loop,
        )
        return future.result()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


def configure_openai_runtime(base_url, api_key, model, concurrency, timeout=600):
    """Create the process-wide runtime, replacing it only if config changed."""
    if int(concurrency) < 1:
        raise ValueError(f"concurrency must be a positive integer, got {concurrency!r}")

    global _runtime
    config = (str(base_url), api_key, model, int(concurrency), timeout)
    with _runtime_lock:
        if _runtime is not None and _runtime.config == config:
            return
        if _runtime is not None:
            _runtime.close()
        _runtime = _PersistentOpenAIRuntime(*config)


def shutdown_openai_runtime():
    """Close the persistent client and its event loop. Safe to call twice."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.close()
            _runtime = None


def load_video(video_path, args):
    """Load and resize video for Vgent chunk processing."""
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


def _prepare_messages(text, video):
    """Encode request content once, outside the retried network call."""
    content = []
    if video is not None:
        content.extend(_frames_to_openai_content(video))
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


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
    Synchronous wrapper for Vgent's graph/retrieval code. The request itself is
    submitted to the one persistent async runtime shared by all worker threads.
    """
    try:
        with _runtime_lock:
            runtime = _runtime
        if runtime is None:
            raise ValueError(
                "[lmms_eval_async_openai] OpenAI runtime is not configured. "
                "Call vgent_adapter.init_vgent_instance() first."
            )
        messages = _prepare_messages(text, video)
        return runtime.request(messages, max_new_tokens)
    except Exception:
        import traceback
        traceback.print_exc()
        return ""
