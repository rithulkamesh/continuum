"""LangGraph ``BaseCheckpointSaver`` on a Continuum checkpoint store."""

from __future__ import annotations

import base64
import json
import os
import random
import tempfile
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from continuum._native import Session
from continuum.checkpoints import CheckpointStore


def _enc(part: str) -> str:
    """Store-safe path component for an arbitrary string (thread id, channel, ...)."""
    return base64.urlsafe_b64encode(part.encode()).decode().rstrip("=") or "_"


def _dec(part: str) -> str:
    if part == "_":
        return ""
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)).decode()


def _typed_to_json(typed: tuple[str, bytes]) -> list[str]:
    return [typed[0], base64.b64encode(typed[1]).decode()]


def _typed_from_json(raw: list[str]) -> tuple[str, bytes]:
    return raw[0], base64.b64decode(raw[1])


class ContinuumCheckpointSaver(BaseCheckpointSaver[str]):
    """LangGraph checkpointer that persists to a Continuum ``CheckpointStore``.

    Args:
        store: Any :class:`~continuum.checkpoints.CheckpointStore`
            (``LocalDirectoryStore``, ``S3Store``, ``GCSStore``).
        session: Optional Continuum ``Session`` whose prefix-KV index is
            snapshotted with every checkpoint. When a thread is loaded into a
            session whose cache is still empty (a fresh process), the snapshot
            is restored, so the resumed run starts warm.
        prefix: Key prefix inside the store.

    Storage layout (under ``prefix``): each channel value is written once per
    version (``.../blobs/<channel>/<version>``), so a checkpoint only adds the
    channels that changed; checkpoint records name their parent, which keeps
    lineage for forks made with ``graph.update_state(past_config, ...)``.
    Checkpoint, blob, and write objects are create-if-absent, so several
    workers can share one store.
    """

    def __init__(
        self,
        store: CheckpointStore,
        *,
        session: Session | None = None,
        prefix: str = "langgraph",
        serde: Any = None,
    ) -> None:
        super().__init__(serde=serde)
        self.store = store
        self.session = session
        self.prefix = prefix.strip("/")

    # -- keys ------------------------------------------------------------------

    def _base(self, thread_id: str, ns: str) -> str:
        return f"{self.prefix}/{_enc(thread_id)}/{_enc(ns)}"

    def _checkpoint_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._base(thread_id, ns)}/checkpoints/{_enc(checkpoint_id)}"

    def _blob_key(self, thread_id: str, ns: str, channel: str, version: Any) -> str:
        return f"{self._base(thread_id, ns)}/blobs/{_enc(channel)}/{_enc(str(version))}"

    def _writes_prefix(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._base(thread_id, ns)}/writes/{_enc(checkpoint_id)}/"

    # -- KV snapshots ------------------------------------------------------------

    def _snapshot_kv(self) -> bytes | None:
        if self.session is None or self.session.cache_size() == 0:
            return None
        fd, path = tempfile.mkstemp(suffix=".cpkv")
        os.close(fd)
        try:
            if not self.session.save_cache_metadata(path):
                return None
            with open(path, "rb") as f:
                return f.read()
        finally:
            os.unlink(path)

    def _restore_kv(self, data: bytes) -> bool:
        assert self.session is not None
        fd, path = tempfile.mkstemp(suffix=".cpkv")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return bool(self.session.load_cache_metadata(path))
        finally:
            os.unlink(path)

    # -- read ------------------------------------------------------------------------

    def _tuple(self, thread_id: str, ns: str, checkpoint_id: str, record: dict[str, Any]) -> CheckpointTuple:
        checkpoint: Checkpoint = self.serde.loads_typed(_typed_from_json(record["checkpoint"]))
        values: dict[str, Any] = {}
        for channel, version in checkpoint["channel_versions"].items():
            try:
                typed = _typed_from_json(json.loads(self.store.get(self._blob_key(thread_id, ns, channel, version))))
            except KeyError:
                continue
            if typed[0] != "empty":
                values[channel] = self.serde.loads_typed(typed)
        writes = []
        for key in self.store.list(self._writes_prefix(thread_id, ns, checkpoint_id)):
            w = json.loads(self.store.get(key))
            writes.append((w["task_id"], w["channel"], self.serde.loads_typed(_typed_from_json(w["value"]))))
        parent = record.get("parent")
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}},
            checkpoint={**checkpoint, "channel_values": values},
            metadata=self.serde.loads_typed(_typed_from_json(record["metadata"])),
            parent_config=(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": parent}}
                if parent
                else None
            ),
            pending_writes=writes,
        )

    def _checkpoint_ids(self, thread_id: str, ns: str) -> list[str]:
        prefix = f"{self._base(thread_id, ns)}/checkpoints/"
        return sorted((_dec(k[len(prefix):]) for k in self.store.list(prefix)), reverse=True)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            ids = self._checkpoint_ids(thread_id, ns)
            if not ids:
                return None
            checkpoint_id = ids[0]
        try:
            record = json.loads(self.store.get(self._checkpoint_key(thread_id, ns, checkpoint_id)))
        except KeyError:
            return None
        if self.session is not None and self.session.cache_size() == 0 and record.get("kv"):
            try:
                self._restore_kv(self.store.get(record["kv"]))
            except KeyError:
                pass
        return self._tuple(thread_id, ns, checkpoint_id, record)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is not None:
            threads = [config["configurable"]["thread_id"]]
        else:
            threads = sorted({_dec(k.split("/")[1]) for k in self.store.list(self.prefix + "/")})
        want_ns = config["configurable"].get("checkpoint_ns") if config else None
        want_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        for thread_id in threads:
            base = f"{self.prefix}/{_enc(thread_id)}/"
            namespaces = sorted({_dec(k[len(base):].split("/")[0]) for k in self.store.list(base)})
            for ns in namespaces:
                if want_ns is not None and ns != want_ns:
                    continue
                for checkpoint_id in self._checkpoint_ids(thread_id, ns):
                    if want_id and checkpoint_id != want_id:
                        continue
                    if before_id and checkpoint_id >= before_id:
                        continue
                    record = json.loads(self.store.get(self._checkpoint_key(thread_id, ns, checkpoint_id)))
                    tup = self._tuple(thread_id, ns, checkpoint_id, record)
                    if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                        continue
                    if limit is not None:
                        if limit <= 0:
                            return
                        limit -= 1
                    yield tup

    # -- write -------------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        for channel, version in new_versions.items():
            typed = self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
            self.store.put_if_absent(
                self._blob_key(thread_id, ns, channel, version), json.dumps(_typed_to_json(typed)).encode()
            )
        kv_key = None
        kv = self._snapshot_kv()
        if kv is not None:
            kv_key = f"{self._base(thread_id, ns)}/kv/{_enc(checkpoint['id'])}"
            self.store.put_if_absent(kv_key, kv)
        record = {
            "checkpoint": _typed_to_json(self.serde.dumps_typed(c)),
            "metadata": _typed_to_json(self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))),
            "parent": config["configurable"].get("checkpoint_id"),
            "kv": kv_key,
        }
        self.store.put_if_absent(
            self._checkpoint_key(thread_id, ns, checkpoint["id"]), json.dumps(record).encode()
        )
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint["id"]}}

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        prefix = self._writes_prefix(thread_id, ns, checkpoint_id)
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            key = f"{prefix}{_enc(task_id)}-{write_idx:+06d}"
            payload = json.dumps(
                {
                    "task_id": task_id,
                    "channel": channel,
                    "value": _typed_to_json(self.serde.dumps_typed(value)),
                    "task_path": task_path,
                }
            ).encode()
            if write_idx >= 0:
                self.store.put_if_absent(key, payload)  # regular writes are write-once
            else:
                self.store.put(key, payload)  # special channels (errors, interrupts) overwrite

    def delete_thread(self, thread_id: str) -> None:
        for key in self.store.list(f"{self.prefix}/{_enc(thread_id)}/"):
            self.store.delete(key)

    # -- async (delegates; stores are synchronous) ----------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        # The random suffix keeps versions unique across forked branches, so two
        # branches never write different values under one write-once blob key.
        return f"{current_v + 1:032}.{random.random():016}"
