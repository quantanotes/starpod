from contextlib import contextmanager
import threading

class RWLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._readers = 0

    def acquire_read(self):
        with self._count_lock:
            self._readers += 1
            if self._readers == 1:
                self._lock.acquire()

    def acquire_release(self):
        assert self._readers > 0
        with self._count_lock:
            self._readers -= 1
            if self._readers == 0:
                self._lock.release()

    def acquire_write(self):
        self._lock.acquire()

    def release_write(self):
        self._lock.release()

    @contextmanager
    def read_lock(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.acquire_release()

    @contextmanager
    def write_lock(self):
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
