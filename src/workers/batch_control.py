from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Sequence, TypeVar

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in non-GPU test envs
    torch = None  # type: ignore[assignment]


T = TypeVar("T")


@dataclass
class AdaptiveBatchResult(Generic[T]):
    """Return object for adaptive batch executions."""

    result: T
    batch_size: int
    retries: int


_RETRYABLE_HINTS = (
    "out of memory",
    "cuda error",
    "cublas",
    "cudnn",
    "insufficient memory",
    "max position embeddings",
    "maximum context length",
    "context length",
    "sequence length",
    "requested tokens",
)


def _iter_exception_messages(exc: BaseException) -> Iterator[str]:
    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        msg = str(cursor).strip()
        if msg:
            yield msg.lower()
        cursor = cursor.__cause__ or cursor.__context__


def is_retryable_context_error(exc: BaseException) -> bool:
    """Return True for errors likely solved by reducing batch/chunk size."""
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return any(hint in msg for msg in _iter_exception_messages(exc) for hint in _RETRYABLE_HINTS)


def maybe_clear_cuda_cache() -> None:
    """Best-effort CUDA cache clear after retryable failures."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def adaptive_batch_execute(
    run_fn: Callable[[int], T],
    *,
    initial_batch_size: int,
    min_batch_size: int = 1,
    max_retries: int = 8,
    enabled: bool = True,
    phase_name: str = "phase",
    on_retry: Callable[[str], None] | None = None,
) -> AdaptiveBatchResult[T]:
    """Run `run_fn(batch_size)` with optional retry+halving fallback."""
    if initial_batch_size < 1:
        raise ValueError("initial_batch_size must be >= 1")
    if min_batch_size < 1:
        raise ValueError("min_batch_size must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    batch_size = initial_batch_size
    min_size = min(min_batch_size, initial_batch_size)
    retries = 0

    while True:
        try:
            return AdaptiveBatchResult(
                result=run_fn(batch_size),
                batch_size=batch_size,
                retries=retries,
            )
        except Exception as exc:
            can_retry = (
                enabled
                and retries < max_retries
                and batch_size > min_size
                and is_retryable_context_error(exc)
            )
            if not can_retry:
                raise

            next_batch = max(min_size, batch_size // 2)
            if next_batch == batch_size:
                next_batch = max(min_size, batch_size - 1)
            if next_batch >= batch_size:
                raise

            if on_retry is not None:
                on_retry(
                    f"{phase_name}: reducing batch/chunk size {batch_size} -> {next_batch} "
                    f"after retryable error: {exc}"
                )

            retries += 1
            batch_size = next_batch
            maybe_clear_cuda_cache()


def chunk_items(items: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    """Yield slices for chunked processing (chunk_size<=0 means one chunk)."""
    total = len(items)
    if total == 0:
        return
    if chunk_size <= 0 or chunk_size >= total:
        yield items
        return
    for start in range(0, total, chunk_size):
        yield items[start : start + chunk_size]
