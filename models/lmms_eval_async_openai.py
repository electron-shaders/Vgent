"""
lmms_eval_async_openai.py — Vgent model backend that delegates to a vLLM
OpenAI-compatible server via AsyncOpenAI client.

Drop-in replacement for models/qwenvl.py: exposes the same three callables
(load_video, load_model, mllm_response) consumed by utils/vgent.py.

Graph construction calls (mllm_response) are routed through a persistent
AsyncOpenAI client configured by lmms-eval, so all requests share the same HTTP
connection pool and benefit from vLLM's continuous-batching scheduler natively.

Raw final answers requested by lmms-eval retry truncation with a doubled budget
up to VGENT_TRUNCATION_MAX_TOKENS (default: 16384), then return the last answer.
Graph/helper calls retain strict validation and separate graph-budget retries.
"""

import asyncio
import logging
import threading
from concurrent.futures import CancelledError

import numpy as np
import openai
import torch
from models.utils import fetch_video, resize_video
from PIL import Image
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from utils.generation import VgentResponseTruncatedError, resolve_final_token_limit

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
        self.final_token_limit = resolve_final_token_limit()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._close_started = False
        self._state_lock = threading.RLock()
        self._pending = set()
        self._startup_error = None
        self._shutdown_error = None
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

        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.run_until_complete(self._shutdown())
            except BaseException as exc:
                self._shutdown_error = exc
            finally:
                self._loop.close()

    async def _shutdown(self):
        # Drive cancellation to completion before closing the HTTP pool/loop.
        # Stopping the loop alone strands request().result() in worker threads.
        pending = asyncio.all_tasks(self._loop) - {asyncio.current_task()}
        for task in pending:
            task.cancel()
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await self._client.close()

    def raise_if_cancelled(self):
        if self._closed:
            raise CancelledError("The Vgent OpenAI runtime is closed")

    async def _call_api_with_retry(self, messages, max_new_tokens, generation_kwargs=None, return_raw_response=False):
        # Only the final answer may use reasoning. Structured helper calls need
        # their token allowance for JSON, not a hidden reasoning response.
        generation = dict(generation_kwargs or {})
        api_kwargs = {"temperature": generation.get("temperature", 0.0)}
        if "top_p" in generation:
            api_kwargs["top_p"] = generation["top_p"]
        if "qwen3" in self.model.lower():
            enable_thinking = generation_kwargs is not None
            api_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            if enable_thinking and "thinking_token_budget" in generation:
                api_kwargs["extra_body"]["thinking_token_budget"] = generation["thinking_token_budget"]
        if return_raw_response:
            max_new_tokens = int(max_new_tokens)
            if max_new_tokens < 1:
                raise ValueError("max_new_tokens must be a positive integer")
            if max_new_tokens > self.final_token_limit:
                _log.warning("Clamping Vgent final max_new_tokens=%s to VGENT_TRUNCATION_MAX_TOKENS=%s", max_new_tokens, self.final_token_limit)
            max_new_tokens = min(max_new_tokens, self.final_token_limit)

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
        async def _do_call(budget):
            # Acquire for one HTTP attempt only.  Exceptions and cancellations
            # release the permit before tenacity performs its retry backoff.
            async with self._semaphore:
                self.raise_if_cancelled()
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=budget,
                    **api_kwargs,
                )

        content = ""
        retrying_truncation = False
        while True:
            try:
                response = await _do_call(max_new_tokens)
            except openai.BadRequestError as exc:
                context_error = any(
                    marker in str(exc).lower()
                    for marker in ("maximum context length", "max_model_len", "context window")
                )
                if not retrying_truncation or not context_error:
                    raise
                _log.warning("Vgent truncation retry exceeds model context; returning last response: %s", exc)
                return content

            choice = response.choices[0]
            content = choice.message.content
            content = content if isinstance(content, str) else ""
            if choice.finish_reason == "length":
                if not return_raw_response:
                    # Structured graph/helper calls retain their own validators
                    # and budgets; never accept partial JSON as a valid graph.
                    raise VgentResponseTruncatedError(max_new_tokens)
                next_budget = min(max_new_tokens * 2, self.final_token_limit)
                _log.warning(
                    "Vgent final response truncated (finish_reason=length, max_new_tokens=%s, completion_tokens=%s); %s",
                    max_new_tokens, getattr(getattr(response, "usage", None), "completion_tokens", None),
                    f"retrying with max_new_tokens={next_budget}"
                    if next_budget > max_new_tokens else "token cap reached; returning raw response to evaluator",
                )
                if next_budget <= max_new_tokens:
                    return content
                max_new_tokens = next_budget
                retrying_truncation = True
                continue

            if return_raw_response:
                # Empty/whitespace answers are also graded by lmms-eval.
                return content
            if not content.strip():
                raise RuntimeError(f"Vgent returned no answer content (finish_reason={choice.finish_reason!r})")
            return content

    def request(self, messages, max_new_tokens, generation_kwargs=None, return_raw_response=False):
        # Admission and tracking must be atomic with respect to cancel/close.
        with self._state_lock:
            self.raise_if_cancelled()
            future = asyncio.run_coroutine_threadsafe(
                self._call_api_with_retry(messages, max_new_tokens, generation_kwargs, return_raw_response),
                self._loop,
            )
            self._pending.add(future)
            future.add_done_callback(self._forget_request)
        return future.result()

    def _forget_request(self, future):
        with self._state_lock:
            self._pending.discard(future)

    def cancel(self):
        """Reject new work and release callers, without joining any threads."""
        with self._state_lock:
            self._closed = True
            pending = tuple(self._pending)
        for future in pending:
            future.cancel()

    def close(self):
        self.cancel()
        with self._state_lock:
            if not self._close_started:
                self._close_started = True
                self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        if self._shutdown_error is not None:
            raise RuntimeError("Failed to close the Vgent OpenAI client") from self._shutdown_error


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


def cancel_openai_runtime():
    """Signal cancellation before asyncio waits for its executor workers."""
    with _runtime_lock:
        if _runtime is not None:
            _runtime.cancel()


def shutdown_openai_runtime():
    """Close the persistent client and its event loop. Safe to call twice."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            try:
                _runtime.close()
            finally:
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


def _frames_to_openai_content(video, check_cancelled=None):
    """Convert a video chunk tensor to a list of base64 image_url content items."""
    import os
    from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

    if check_cancelled is not None:
        check_cancelled()
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
        if check_cancelled is not None:
            check_cancelled()
        image = Image.fromarray(frame)
        b64 = encode_image_to_base64(image, image_format=image_format, quality=quality)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
    return content


def _prepare_messages(text, video, check_cancelled=None):
    """Encode request content once, outside the retried network call."""
    content = []
    if video is not None:
        content.extend(_frames_to_openai_content(video, check_cancelled))
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
    generation_kwargs=None,
    return_raw_response=False,
):
    """
    Synchronous wrapper for Vgent's graph/retrieval code. The request itself is
    submitted to the one persistent async runtime shared by all worker threads.
    """
    with _runtime_lock:
        runtime = _runtime
    if runtime is None:
        raise ValueError(
            "[lmms_eval_async_openai] OpenAI runtime is not configured. "
            "Call vgent_adapter.init_vgent_instance() first."
        )
    runtime.raise_if_cancelled()
    messages = _prepare_messages(text, video, runtime.raise_if_cancelled)
    return runtime.request(messages, max_new_tokens, generation_kwargs, return_raw_response)
