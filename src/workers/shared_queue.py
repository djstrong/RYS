#!/usr/bin/env python
"""Shared queue helpers for multi-worker benchmark runs."""

from __future__ import annotations

import fcntl
import json
import pickle
import time
from pathlib import Path
from typing import Optional


class SharedWorkQueue:
    """Shared work queue for multi-GPU workers."""

    def __init__(self, queue_file: str, results_file: str, lock_timeout: float = 30.0):
        self.queue_file = Path(queue_file)
        self.results_file = Path(results_file)
        self.lock_timeout = lock_timeout

    def _acquire_lock(self, file_handle, exclusive: bool = True):
        """Acquire file lock with timeout."""
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        start_time = time.time()

        while True:
            try:
                fcntl.flock(file_handle, lock_type | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.time() - start_time > self.lock_timeout:
                    raise TimeoutError(f"Could not acquire lock within {self.lock_timeout}s")
                time.sleep(0.1)

    def _release_lock(self, file_handle):
        """Release file lock."""
        fcntl.flock(file_handle, fcntl.LOCK_UN)

    def get_next_config(self) -> Optional[dict]:
        """
        Atomically get the next config from the queue.

        Returns:
            Queue entry dict (typically {'idx', 'layers', ...} or legacy {'idx', 'key'}),
            or None if queue is empty.
        """
        if not self.queue_file.exists():
            return None

        with open(self.queue_file, "r+") as f:
            self._acquire_lock(f, exclusive=True)
            try:
                f.seek(0)
                content = f.read()
                if not content.strip():
                    return None

                queue = json.loads(content)
                if not queue:
                    return None

                entry = queue.pop(0)

                f.seek(0)
                f.truncate()
                json.dump(queue, f)

                return entry
            finally:
                self._release_lock(f)

    def get_queue_status(self) -> tuple[int, int]:
        """Get (remaining, completed) counts."""
        remaining = 0
        completed = 0

        if self.queue_file.exists():
            with open(self.queue_file, "r") as f:
                self._acquire_lock(f, exclusive=False)
                try:
                    content = f.read()
                    if content.strip():
                        queue = json.loads(content)
                        remaining = len(queue)
                finally:
                    self._release_lock(f)

        if self.results_file.exists():
            with open(self.results_file, "rb") as f:
                self._acquire_lock(f, exclusive=False)
                try:
                    results = pickle.load(f)
                    completed = len(results)
                finally:
                    self._release_lock(f)

        return remaining, completed

    def get_remaining_count(self) -> int:
        """Get only the number of remaining queue entries."""
        if not self.queue_file.exists():
            return 0
        with open(self.queue_file, "r") as f:
            self._acquire_lock(f, exclusive=False)
            try:
                content = f.read()
                if not content.strip():
                    return 0
                queue = json.loads(content)
                return len(queue)
            finally:
                self._release_lock(f)

    def _ensure_results_file(self) -> None:
        if self.results_file.exists():
            return
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "wb") as f:
            self._acquire_lock(f, exclusive=True)
            try:
                pickle.dump({}, f)
            finally:
                self._release_lock(f)

    def save_result(self, config_key: tuple[int, ...], score):
        """Atomically save a result."""
        self.save_results_bulk({config_key: score})

    def save_results_bulk(self, updates: dict[tuple[int, ...], object]) -> None:
        """Atomically save multiple results in one lock/read/write cycle."""
        if not updates:
            return
        self._ensure_results_file()
        with open(self.results_file, "r+b") as f:
            self._acquire_lock(f, exclusive=True)
            try:
                f.seek(0)
                results = pickle.load(f)
                results.update(updates)
                f.seek(0)
                f.truncate()
                pickle.dump(results, f)
            finally:
                self._release_lock(f)


def format_eta(seconds: float) -> str:
    """Format seconds as a short human-readable ETA string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:.0f}h {minutes:.0f}m"
