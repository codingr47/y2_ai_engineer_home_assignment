"""Bounded process-local LRU with atomic operations and isolated values."""

from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("cache capacity must be a nonnegative integer")
        self.capacity = capacity
        self._values: OrderedDict[str, T] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            if key not in self._values:
                return None
            self._values.move_to_end(key)
            return deepcopy(self._values[key])

    def put(self, key: str, value: T) -> bool:
        """Store a copy; return whether an entry was evicted."""
        with self._lock:
            if not self.capacity:
                return False
            self._values[key] = deepcopy(value)
            self._values.move_to_end(key)
            if len(self._values) > self.capacity:
                self._values.popitem(last=False)
                return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
