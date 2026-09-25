"""Checkpoint storage: pluggable object stores and an incremental checkpoint log.

A checkpoint (:meth:`DurableAgent.run_until_step`) is plain ``bytes``. This
module moves those bytes to durable storage so a run can resume on another
machine without the caller shipping them around.

**Stores** (:class:`CheckpointStore`) are minimal key/value object stores:

- :class:`LocalDirectoryStore` writes files under a directory (atomic renames).
- :class:`S3Store` uses ``boto3`` (``pip install boto3``).
- :class:`GCSStore` uses ``google-cloud-storage``.

Each store supports an atomic create-if-absent write, which is what makes
concurrent writers safe.

**The log** (:class:`CheckpointLog`) stores a stream of checkpoints for one run
on top of any store:

- Checkpoints are content-addressed: the id is the SHA-256 of the full
  checkpoint bytes, and every load is verified against it.
- A checkpoint committed with a ``parent`` is stored as a *delta* (only the
  values and KV entries that changed, see
  :func:`continuum._native.checkpoint_delta`), with a full checkpoint every
  ``max_chain`` links so reconstruction stays bounded.
- Each checkpoint has a small JSON manifest record linking it to its parent
  and delta base, so any step can be rebuilt, and :meth:`CheckpointLog.fork`
  records fork lineage (source checkpoint + edited node).
- Objects and records are write-once (create-if-absent) and never modified,
  so any number of workers can resume from and commit to one log without
  locks and without overwriting each other.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continuum._native import DurableAgent as _NativeDurableAgent
from continuum._native import apply_checkpoint_delta, checkpoint_delta

__all__ = [
    "CheckpointCorruptionError",
    "CheckpointLog",
    "CheckpointRecord",
    "CheckpointStore",
    "GCSStore",
    "LocalDirectoryStore",
    "S3Store",
]


def _check_key(key: str) -> str:
    parts = key.split("/")
    if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid checkpoint key {key!r}: use non-empty '/'-separated names")
    return key


class CheckpointStore(ABC):
    """Minimal object store for checkpoint bytes.

    Keys are ``/``-separated relative names (no ``..``, no leading ``/``).
    """

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Write ``data`` under ``key``, replacing any existing object atomically."""

    @abstractmethod
    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Create ``key`` only if it does not exist. Returns False if it did."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Read ``key``. Raises ``KeyError`` if it does not exist."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether ``key`` exists."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """All keys starting with ``prefix``, sorted."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``; missing keys are ignored."""


class LocalDirectoryStore(CheckpointStore):
    """Files under ``root``. Writes go to a temp file and are renamed into
    place, so readers never see a partial checkpoint."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*_check_key(key).split("/"))

    def _write_temp(self, path: Path, data: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            os.unlink(tmp)
            raise
        return tmp

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.replace(self._write_temp(path, data), path)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        path = self._path(key)
        tmp = self._write_temp(path, data)
        try:
            os.link(tmp, path)  # atomic; fails if the target exists
            return True
        except FileExistsError:
            return False
        finally:
            os.unlink(tmp)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        out = []
        for p in self.root.rglob("*"):
            if p.is_file() and not p.name.startswith(".tmp-"):
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return str(getattr(exc, "code", ""))


class S3Store(CheckpointStore):
    """Amazon S3 (or any S3-compatible service) via ``boto3``.

    Args:
        bucket: Bucket name.
        prefix: Key prefix inside the bucket, e.g. ``"continuum/ckpt"``.
        client: A ``boto3`` S3 client; created with ``boto3.client("s3")``
            when omitted.

    ``put_if_absent`` uses S3 conditional writes (``If-None-Match: *``).
    """

    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        if client is None:  # pragma: no cover - needs boto3 + credentials
            import boto3

            client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client

    def _key(self, key: str) -> str:
        _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=self._key(key), Body=data, IfNoneMatch="*"
            )
            return True
        except Exception as exc:
            if _error_code(exc) in ("PreconditionFailed", "ConditionalRequestConflict", "412"):
                return False
            raise

    def get(self, key: str) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:
            if _error_code(exc) in ("NoSuchKey", "404", "NotFound"):
                raise KeyError(key) from None
            raise
        return bytes(obj["Body"].read())

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as exc:
            if _error_code(exc) in ("NoSuchKey", "404", "NotFound"):
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        full = f"{self.prefix}/{prefix}" if self.prefix else prefix
        strip = len(self.prefix) + 1 if self.prefix else 0
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            keys.extend(item["Key"][strip:] for item in page.get("Contents", []))
        return sorted(keys)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


class GCSStore(CheckpointStore):
    """Google Cloud Storage via ``google-cloud-storage``.

    Args:
        bucket: Bucket name.
        prefix: Object-name prefix inside the bucket.
        client: A ``google.cloud.storage.Client``; created when omitted.

    ``put_if_absent`` uses a generation precondition (``if_generation_match=0``).
    """

    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        if client is None:  # pragma: no cover - needs google-cloud-storage + credentials
            from google.cloud import storage

            client = storage.Client()
        self.client = client
        self.bucket = client.bucket(bucket)
        self.prefix = prefix.strip("/")

    def _name(self, key: str) -> str:
        _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self.bucket.blob(self._name(key)).upload_from_string(data)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        try:
            self.bucket.blob(self._name(key)).upload_from_string(data, if_generation_match=0)
            return True
        except Exception as exc:
            if _error_code(exc) == "412":
                return False
            raise

    def get(self, key: str) -> bytes:
        try:
            return bytes(self.bucket.blob(self._name(key)).download_as_bytes())
        except Exception as exc:
            if _error_code(exc) == "404":
                raise KeyError(key) from None
            raise

    def exists(self, key: str) -> bool:
        return bool(self.bucket.blob(self._name(key)).exists())

    def list(self, prefix: str = "") -> list[str]:
        full = f"{self.prefix}/{prefix}" if self.prefix else prefix
        strip = len(self.prefix) + 1 if self.prefix else 0
        return sorted(b.name[strip:] for b in self.client.list_blobs(self.bucket, prefix=full))

    def delete(self, key: str) -> None:
        blob = self.bucket.blob(self._name(key))
        try:
            blob.delete()
        except Exception as exc:
            if _error_code(exc) != "404":
                raise


class CheckpointCorruptionError(RuntimeError):
    """A stored checkpoint did not rebuild to the bytes its id promises."""


@dataclass(frozen=True)
class CheckpointRecord:
    """Manifest record for one checkpoint in a :class:`CheckpointLog`."""

    id: str
    parent: str | None
    """Checkpoint this one was derived from (previous step, or fork source)."""
    kind: str
    """``"full"`` or ``"delta"``."""
    base: str | None
    """For a delta: the checkpoint it applies to (equals ``parent``)."""
    depth: int
    """Deltas between this checkpoint and the nearest full one."""
    object: str
    """Store key of the payload."""
    size: int
    """Stored payload bytes."""
    full_size: int
    """Bytes of the reconstructed full checkpoint."""
    executed_nodes: int
    step: int | None
    label: str | None
    fork: dict[str, Any] | None
    """``{"node_id": ...}`` when this checkpoint forks ``parent`` by editing a node."""
    created: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> CheckpointRecord:
        return cls(**json.loads(raw))


class CheckpointLog:
    """Incremental, content-addressed checkpoint stream for one run.

    Args:
        store: Where objects and manifest records live.
        run_id: Namespace for this run inside the store.
        max_chain: Longest run of deltas before a full checkpoint is stored.
    """

    def __init__(self, store: CheckpointStore, run_id: str, max_chain: int = 16) -> None:
        self.store = store
        self.run_id = _check_key(run_id)
        self.max_chain = max(0, max_chain)
        # Recently rebuilt checkpoints, so committing step N+1 after step N
        # does not re-walk the delta chain.
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_limit = 8

    def _remember(self, cid: str, data: bytes) -> None:
        self._cache[cid] = data
        self._cache.move_to_end(cid)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    # -- keys --------------------------------------------------------------

    def _record_key(self, cid: str) -> str:
        return f"{self.run_id}/manifest/{cid}.json"

    # -- write -------------------------------------------------------------

    def commit(
        self,
        checkpoint: bytes,
        parent: str | None = None,
        *,
        step: int | None = None,
        label: str | None = None,
        fork: dict[str, Any] | None = None,
    ) -> str:
        """Store ``checkpoint`` and return its id.

        With a ``parent``, the payload is a delta against the parent unless
        the chain is already ``max_chain`` long or the delta would not be
        smaller. Committing the same checkpoint twice (from any worker) is a
        no-op that returns the same id.
        """
        cid = hashlib.sha256(checkpoint).hexdigest()
        if self.store.exists(self._record_key(cid)):
            return cid

        kind, base, depth, payload = "full", None, 0, checkpoint
        if parent is not None:
            parent_rec = self.record(parent)
            if parent_rec.depth < self.max_chain:
                delta = checkpoint_delta(self.load(parent), checkpoint)
                if len(delta) < len(checkpoint):
                    kind, base, depth, payload = "delta", parent, parent_rec.depth + 1, delta

        # Each writer's payload gets its own object key, so a record always
        # points at the payload its writer produced, whoever wins the record.
        obj = f"{self.run_id}/objects/{cid}-{kind}-{base or 'root'}-{uuid.uuid4().hex[:8]}"
        self.store.put_if_absent(obj, payload)
        info = _NativeDurableAgent.inspect(checkpoint)
        rec = CheckpointRecord(
            id=cid,
            parent=parent,
            kind=kind,
            base=base,
            depth=depth,
            object=obj,
            size=len(payload),
            full_size=len(checkpoint),
            executed_nodes=int(info["executed_nodes"]),
            step=step,
            label=label,
            fork=fork,
            created=time.time(),
        )
        if not self.store.put_if_absent(self._record_key(cid), rec.to_json().encode()):
            self.store.delete(obj)  # another worker committed this checkpoint first
        self._remember(cid, checkpoint)
        return cid

    def fork(
        self, checkpoint_id: str, node_id: int, new_value: Any, *, label: str | None = None
    ) -> str:
        """Commit a fork of ``checkpoint_id`` with ``node_id`` set to ``new_value``.

        The new record's ``parent`` is the source checkpoint and its ``fork``
        field names the edited node, so lineage survives in the manifest.
        """
        forked = _NativeDurableAgent.fork(self.load(checkpoint_id), node_id, new_value)
        return self.commit(
            forked, parent=checkpoint_id, label=label, fork={"node_id": int(node_id)}
        )

    # -- read --------------------------------------------------------------

    def record(self, checkpoint_id: str) -> CheckpointRecord:
        try:
            return CheckpointRecord.from_json(self.store.get(self._record_key(checkpoint_id)))
        except KeyError:
            raise KeyError(f"no checkpoint {checkpoint_id!r} in run {self.run_id!r}") from None

    def load(self, checkpoint_id: str) -> bytes:
        """Rebuild the full checkpoint bytes, verified against the id."""
        if checkpoint_id in self._cache:
            self._cache.move_to_end(checkpoint_id)
            return self._cache[checkpoint_id]
        chain: list[CheckpointRecord] = []
        rec = self.record(checkpoint_id)
        while True:
            chain.append(rec)
            if rec.kind == "full" or rec.base is None:
                break
            if rec.base in self._cache:
                break
            rec = self.record(rec.base)
        top = chain[-1]
        if top.kind == "full":
            data = self.store.get(top.object)
            self._verify(top.id, data)
            chain.pop()
        else:
            assert top.base is not None
            data = self._cache[top.base]
        for rec in reversed(chain):
            data = apply_checkpoint_delta(data, self.store.get(rec.object))
            self._verify(rec.id, data)
        self._remember(checkpoint_id, data)
        return data

    @staticmethod
    def _verify(checkpoint_id: str, data: bytes) -> None:
        if hashlib.sha256(data).hexdigest() != checkpoint_id:
            raise CheckpointCorruptionError(f"checkpoint {checkpoint_id} failed its content hash")

    def manifest(self) -> list[CheckpointRecord]:
        """Every checkpoint record in the run, oldest first."""
        prefix = f"{self.run_id}/manifest/"
        records = [CheckpointRecord.from_json(self.store.get(k)) for k in self.store.list(prefix)]
        return sorted(records, key=lambda r: (r.created, r.id))

    def lineage(self, checkpoint_id: str) -> list[CheckpointRecord]:
        """Records from the root of ``checkpoint_id``'s history down to it."""
        out = []
        cur: str | None = checkpoint_id
        while cur is not None:
            rec = self.record(cur)
            out.append(rec)
            cur = rec.parent
        return list(reversed(out))

    def children(self, checkpoint_id: str) -> list[CheckpointRecord]:
        return [r for r in self.manifest() if r.parent == checkpoint_id]

    def heads(self) -> list[CheckpointRecord]:
        """Checkpoints nothing has been committed on top of (branch tips)."""
        records = self.manifest()
        parents = {r.parent for r in records}
        return [r for r in records if r.id not in parents]

    def __iter__(self) -> Iterator[CheckpointRecord]:
        return iter(self.manifest())
