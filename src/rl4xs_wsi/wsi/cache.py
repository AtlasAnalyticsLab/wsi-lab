"""Bounded process-local caches for non-picklable slide resources."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from contextlib import suppress
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class ProcessLocalLRU(Generic[K, V]):
    """Thread-safe LRU that discards inherited entries after a process fork."""

    def __init__(
        self,
        max_items: int,
        *,
        on_evict: Callable[[V], None] | None = None,
    ) -> None:
        if max_items < 0:
            raise ValueError("max_items cannot be negative")
        self.max_items = max_items
        self._on_evict = on_evict
        self._items: OrderedDict[K, V] = OrderedDict()
        self._pid = os.getpid()
        self._lock = threading.RLock()

    def _check_process(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        # Handles inherited through fork must never be reused. Closing the
        # child's duplicate descriptors does not close the parent's handles.
        self._clear_unlocked()
        self._pid = pid

    def get(self, key: K) -> V | None:
        with self._lock:
            self._check_process()
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def put(self, key: K, value: V) -> V:
        with self._lock:
            self._check_process()
            previous = self._items.pop(key, None)
            if previous is not None and previous is not value:
                self._evict(previous)
            if self.max_items == 0:
                self._evict(value)
                return value
            self._items[key] = value
            while len(self._items) > self.max_items:
                _, evicted = self._items.popitem(last=False)
                self._evict(evicted)
            return value

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached
            value = factory()
            if self.max_items == 0:
                return value
            return self.put(key, value)

    def pop(self, key: K) -> V | None:
        with self._lock:
            self._check_process()
            return self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._check_process()
            self._clear_unlocked()

    def _clear_unlocked(self) -> None:
        values = list(self._items.values())
        self._items.clear()
        for value in values:
            self._evict(value)

    def _evict(self, value: V) -> None:
        if self._on_evict is not None:
            # Cache cleanup should make best effort during worker teardown.
            with suppress(Exception):
                self._on_evict(value)

    def __len__(self) -> int:
        with self._lock:
            self._check_process()
            return len(self._items)


class ViewportCache:
    """Copy-safe LRU for rendered uint8 RGB viewports."""

    def __init__(self, max_items: int = 256) -> None:
        self._cache: ProcessLocalLRU[Hashable, NDArray[np.uint8]] = ProcessLocalLRU(max_items)

    def get(self, key: Hashable) -> NDArray[np.uint8] | None:
        value = self._cache.get(key)
        return None if value is None else value.copy()

    def put(self, key: Hashable, value: NDArray[np.uint8]) -> None:
        array = np.asarray(value, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("cached viewport must have shape [H,W,3]")
        stored = np.ascontiguousarray(array).copy()
        stored.flags.writeable = False
        self._cache.put(key, stored)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class PersistentViewportCache:
    """Lazy lossless PNG cache; regions are materialized only when requested."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @staticmethod
    def key(
        slide_path: str | Path,
        bounds: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> str:
        path = Path(slide_path).expanduser().resolve()
        stat = path.stat()
        payload = json.dumps(
            {
                "schema": 1,
                "slide": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "bounds": list(map(int, bounds)),
                "output_size": list(map(int, output_size)),
                "encoding": "rgb-png-lossless",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path_for_key(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.png"

    def get(
        self,
        slide_path: str | Path,
        bounds: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> NDArray[np.uint8] | None:
        key = self.key(slide_path, bounds, output_size)
        path = self.path_for_key(key)
        if not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != output_size:
                    return None
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError):
            return None
        return array.copy()

    def put(
        self,
        slide_path: str | Path,
        bounds: tuple[int, int, int, int],
        output_size: tuple[int, int],
        value: NDArray[np.uint8],
    ) -> Path:
        array = np.asarray(value, dtype=np.uint8)
        if array.shape != (output_size[1], output_size[0], 3):
            raise ValueError(
                f"persistent viewport shape {array.shape} does not match output size {output_size}"
            )
        key = self.key(slide_path, bounds, output_size)
        target = self.path_for_key(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".png", dir=target.parent
        )
        os.close(descriptor)
        try:
            Image.fromarray(array, mode="RGB").save(temporary, format="PNG", compress_level=6)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        return target


__all__ = ["PersistentViewportCache", "ProcessLocalLRU", "ViewportCache"]
