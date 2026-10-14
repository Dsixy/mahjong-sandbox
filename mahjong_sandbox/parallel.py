"""Persistent process shards; shared file-backed NumPy pages, small control IPC."""
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import traceback
from .batch import BatchConfig, SyncBatchEnv, array_specs, snapshot, validate_actions
from .operators import numpy


def _layout(n):
    np = numpy()
    fields, offset = {}, 0
    for k, (shape, dtype) in array_specs(n).items():
        offset = (offset + 63) // 64 * 64
        fields[k] = (shape, dtype, offset)
        offset += int(np.prod(shape)) * np.dtype(dtype).itemsize
    return fields, offset


def _map(path, n):
    np = numpy()
    layout, size = _layout(n)
    memory = np.memmap(path, mode='r+', dtype='uint8', shape=(size,))
    arrays = {k: np.ndarray(shape, dtype=dtype, buffer=memory, offset=offset)
              for k, (shape, dtype, offset) in layout.items()}
    return memory, arrays


def _worker(connection, path, total, start, end, config):
    memory, arrays = _map(path, total)
    env = None
    try:
        env = SyncBatchEnv(end - start, config, lane_offset=start,
                           buffers={k: v[start:end] for k, v in arrays.items()})
        connection.send(('ok', None))
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            if operation == 'step':
                env._step(*payload)
            elif operation == 'reset':
                env.buffers['rewards'].fill(0)
                env._reset(payload)
            else:
                raise ValueError('unknown worker command')
            connection.send(('ok', None))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if env:
            env.close()
        arrays.clear()
        memory._mmap.close()
        connection.close()


class ProcessBatchEnv:
    """Many boards per process, never a process per board.

    Parent reads shared buffers only after all workers acknowledge. A timeout
    leaves the same request pending: call step_wait again. Any worker failure
    makes the environment unusable; close releases processes and shared files.
    """
    def __init__(self, n=1000, workers=4, config=None, *, timeout=120):
        if type(n) is not int or n <= 0 or type(workers) is not int or not 1 <= workers <= n:
            raise ValueError('positive n and 1..n workers required')
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError('positive timeout required')
        self.n, self.config, self.timeout = n, config or BatchConfig(), timeout
        self.closed, self.broken, self._pending = False, False, None
        self.processes, self.connections, self.ranges = [], [], []
        self._temporary = tempfile.TemporaryDirectory(prefix='mahjong-batch-')
        self.path = str(Path(self._temporary.name) / 'buffers.bin')
        _, size = _layout(n)
        with open(self.path, 'wb') as f:
            f.truncate(size)
        self.memory, self.buffers = _map(self.path, n)
        self.shared_bytes = size
        try:
            ctx = mp.get_context('spawn')
            for j in range(workers):
                start, end = n * j // workers, n * (j + 1) // workers
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, self.path, n, start, end, self.config), daemon=True)
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
                self.ranges.append((start, end))
            self._pending = set(range(workers))
            self._collect(timeout)
            self._last = snapshot(self.buffers)
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self.closed or self.broken:
            raise RuntimeError('environment is closed or a worker failed')

    def _collect(self, timeout):
        deadline = time.monotonic() + timeout
        for j in sorted(self._pending):
            connection = self.connections[j]
            while not connection.poll(min(.1, max(0, deadline - time.monotonic()))):
                if not self.processes[j].is_alive():
                    self.broken = True
                    raise RuntimeError(f'worker {j} exited with code {self.processes[j].exitcode}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('batch still running; retry step_wait on this environment')
            try:
                status, detail = connection.recv()
            except (EOFError, OSError) as error:
                self.broken = True
                raise RuntimeError(f'worker {j} disconnected') from error
            self._pending.remove(j)
            if status != 'ok':
                self.broken = True
                raise RuntimeError(f'worker {j} failed:\n{detail}')
        self._pending = None

    def observation(self):
        self._check()
        if self._pending is not None:
            raise RuntimeError('request pending; wait before reading buffers')
        return self._last

    def step_async(self, indices, tokens):
        self._check()
        if self._pending is not None:
            raise RuntimeError('request already pending')
        a, t = validate_actions(indices, tokens, self.buffers)
        try:
            for connection, (start, end) in zip(self.connections, self.ranges):
                connection.send(('step', (a[start:end], t[start:end])))
        except (OSError, EOFError) as error:
            self.broken = True
            raise RuntimeError('worker disconnected while submitting batch') from error
        self._pending = set(range(len(self.processes)))

    def step_wait(self, timeout=None):
        self._check()
        if self._pending is None:
            raise RuntimeError('no pending step')
        self._collect(self.timeout if timeout is None else timeout)
        self._last = snapshot(self.buffers)
        return self._last

    def step(self, indices, tokens):
        self.step_async(indices, tokens)
        return self.step_wait()

    def reset(self, indices=None):
        self._check()
        if self._pending is not None:
            raise RuntimeError('request pending')
        selected = SyncBatchEnv._indices(self, indices)
        for connection, (start, end) in zip(self.connections, self.ranges):
            connection.send(('reset', [i - start for i in selected if start <= i < end]))
        self._pending = set(range(len(self.processes)))
        return self.step_wait()

    def reset_done(self):
        return self.reset(numpy().flatnonzero(self._last['terminated'] | self._last['truncated']))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(.5)
            if process.is_alive():
                process.terminate()
                process.join(5)
        for connection in self.connections:
            connection.close()
        self.buffers.clear()
        if hasattr(self, 'memory'):
            self.memory._mmap.close()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
