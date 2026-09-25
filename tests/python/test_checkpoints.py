"""Checkpoint stores (#15) and incremental, multi-worker checkpoint logs (#16)."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from continuum import DurableAgent
from continuum._native import apply_checkpoint_delta, checkpoint_delta, is_checkpoint_delta
from continuum.checkpoints import (
    CheckpointCorruptionError,
    CheckpointLog,
    CheckpointStore,
    GCSStore,
    LocalDirectoryStore,
    S3Store,
)

PROMPTS = ["research", "plan", "book flights", "book hotel", "write itinerary"]


# --- fake cloud clients --------------------------------------------------------


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, IfNoneMatch: str | None = None) -> None:  # noqa: N803
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise _ClientError("PreconditionFailed")
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _ClientError("NoSuchKey")
        data = self.objects[(Bucket, Key)]

        class Body:
            def read(self) -> bytes:
                return data

        return {"Body": Body()}

    def head_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _ClientError("404")

    def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        self.objects.pop((Bucket, Key), None)

    def get_paginator(self, name: str) -> Any:
        assert name == "list_objects_v2"
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
                keys = [k for (b, k) in objects if b == Bucket and k.startswith(Prefix)]
                return [{"Contents": [{"Key": k} for k in keys[:1]]}, {"Contents": [{"Key": k} for k in keys[1:]]}, {}]

        return Paginator()


class _GcsError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class FakeGCS:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def bucket(self, name: str) -> Any:
        objects = self.objects

        class Blob:
            def __init__(self, blob_name: str) -> None:
                self.name = blob_name

            def upload_from_string(self, data: bytes, if_generation_match: int | None = None) -> None:
                if if_generation_match == 0 and self.name in objects:
                    raise _GcsError(412)
                objects[self.name] = bytes(data)

            def download_as_bytes(self) -> bytes:
                if self.name not in objects:
                    raise _GcsError(404)
                return objects[self.name]

            def exists(self) -> bool:
                return self.name in objects

            def delete(self) -> None:
                if self.name not in objects:
                    raise _GcsError(404)
                del objects[self.name]

        class Bucket:
            def blob(self, blob_name: str) -> Blob:
                return Blob(blob_name)

        return Bucket()

    def list_blobs(self, bucket: Any, prefix: str) -> list[Any]:
        class Named:
            def __init__(self, n: str) -> None:
                self.name = n

        return [Named(n) for n in self.objects if n.startswith(prefix)]


@pytest.fixture(params=["local", "s3", "gcs"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> CheckpointStore:
    if request.param == "local":
        return LocalDirectoryStore(tmp_path / "ckpt")
    if request.param == "s3":
        return S3Store("bucket", prefix="runs/", client=FakeS3())
    return GCSStore("bucket", prefix="runs", client=FakeGCS())


# --- stores ---------------------------------------------------------------------


def test_store_contract(store: CheckpointStore) -> None:
    assert not store.exists("a/one")
    store.put("a/one", b"1")
    store.put("a/one", b"11")  # overwrite
    assert store.get("a/one") == b"11"
    assert store.put_if_absent("a/two", b"2")
    assert not store.put_if_absent("a/two", b"x")
    assert store.get("a/two") == b"2"
    store.put("b/three", b"3")
    assert store.list("a/") == ["a/one", "a/two"]
    assert store.list() == ["a/one", "a/two", "b/three"]
    store.delete("a/one")
    store.delete("a/one")  # idempotent
    assert not store.exists("a/one")
    with pytest.raises(KeyError):
        store.get("a/one")
    for bad in ("", "/abs", "a/../b", "a//b"):
        with pytest.raises(ValueError):
            store.put(bad, b"x")


def test_cloud_stores_propagate_unexpected_errors() -> None:
    class Broken(FakeS3):
        def put_object(self, *a: Any, **k: Any) -> None:
            raise _ClientError("AccessDenied")

        def get_object(self, *a: Any, **k: Any) -> dict[str, Any]:
            raise _ClientError("AccessDenied")

        def head_object(self, *a: Any, **k: Any) -> None:
            raise _ClientError("AccessDenied")

    s3 = S3Store("b", client=Broken())
    for call in (lambda: s3.put_if_absent("k", b""), lambda: s3.get("k"), lambda: s3.exists("k")):
        with pytest.raises(_ClientError):
            call()

    gcs = GCSStore("b", client=FakeGCS())
    gcs.bucket = type("B", (), {"blob": lambda self, n: _raising_blob()})()
    for call in (lambda: gcs.put_if_absent("k", b""), lambda: gcs.get("k"), lambda: gcs.delete("k")):
        with pytest.raises(_GcsError):
            call()


def _raising_blob() -> Any:
    class Blob:
        def upload_from_string(self, *a: Any, **k: Any) -> None:
            raise _GcsError(403)

        def download_as_bytes(self) -> bytes:
            raise _GcsError(403)

        def delete(self) -> None:
            raise _GcsError(403)

    return Blob()


# --- DurableAgent + store -----------------------------------------------------------


def test_agent_checkpoints_to_store_and_resumes(tmp_path: Path) -> None:
    store = LocalDirectoryStore(tmp_path)
    agent = DurableAgent()
    agent.begin(PROMPTS)
    blob = agent.run_until_step(1, store=store, key="job/step-2")
    assert store.get("job/step-2") == blob

    revived = DurableAgent()
    outputs = revived.resume_from(store=store, key="job/step-2")
    assert outputs == DurableAgent().resume_from(blob)
    assert revived.cache_size() > 0

    with pytest.raises(ValueError):
        agent.run_until_step(1, store=store)
    with pytest.raises(ValueError):
        DurableAgent().resume_from()
    with pytest.raises(ValueError):
        DurableAgent().resume_from(blob, store=store, key="job/step-2")


# --- deltas ---------------------------------------------------------------------------


def _step_blobs(n: int = 5) -> list[bytes]:
    agent = DurableAgent()
    agent.begin(PROMPTS)
    return [agent.run_until_step(i) for i in range(n)]


def test_native_delta_round_trip() -> None:
    blobs = _step_blobs()
    for base, nxt in zip(blobs, blobs[1:]):
        delta = checkpoint_delta(base, nxt)
        assert is_checkpoint_delta(delta) and not is_checkpoint_delta(nxt)
        assert len(delta) < len(nxt) // 2
        assert apply_checkpoint_delta(base, delta) == nxt
    # A delta is exact in either direction, including removed values.
    assert apply_checkpoint_delta(blobs[3], checkpoint_delta(blobs[3], blobs[0])) == blobs[0]
    with pytest.raises(RuntimeError):
        apply_checkpoint_delta(blobs[0], blobs[1])  # not a delta


def test_log_stores_deltas_and_rebuilds_every_step(store: CheckpointStore) -> None:
    blobs = _step_blobs()
    log = CheckpointLog(store, "run-1", max_chain=2)
    ids: list[str] = []
    for i, blob in enumerate(blobs):
        ids.append(log.commit(blob, ids[-1] if ids else None, step=i))

    kinds = [log.record(c).kind for c in ids]
    assert kinds == ["full", "delta", "delta", "full", "delta"]  # chain capped at 2
    assert [log.record(c).depth for c in ids] == [0, 1, 2, 0, 1]
    assert sum(log.record(c).size for c in ids) < sum(len(b) for b in blobs)

    fresh = CheckpointLog(store, "run-1")  # no cache: rebuild from storage
    for cid, blob in zip(ids, blobs):
        assert fresh.load(cid) == blob
    assert [r.id for r in fresh.lineage(ids[-1])] == ids
    assert [r.id for r in fresh.heads()] == [ids[-1]]
    assert fresh.commit(blobs[2], ids[1]) == ids[2]  # idempotent
    with pytest.raises(KeyError):
        fresh.record("nope")


def test_log_detects_corruption(tmp_path: Path) -> None:
    store = LocalDirectoryStore(tmp_path)
    blobs = _step_blobs(2)
    log = CheckpointLog(store, "run")
    a = log.commit(blobs[0])
    b = log.commit(blobs[1], a)
    store.put(log.record(a).object, blobs[1])  # swap the base payload
    with pytest.raises(CheckpointCorruptionError):
        CheckpointLog(store, "run").load(b)


def test_fork_lineage_recorded(tmp_path: Path) -> None:
    agent = DurableAgent()
    agent.begin(PROMPTS)
    log = CheckpointLog(LocalDirectoryStore(tmp_path), "run")
    root = log.commit(agent.run_until_step(1), step=1)
    edited_node = agent.prompt_node_ids[3]
    fork = log.fork(root, edited_node, "book a train instead", label="train")

    rec = log.record(fork)
    assert rec.parent == root
    assert rec.fork == {"node_id": edited_node}
    assert rec.label == "train"
    assert [c.id for c in log.children(root)] == [fork]
    real = DurableAgent().resume_from(log.load(root))
    alt = DurableAgent().resume_from(log.load(fork))
    assert real != alt


# --- multi-worker ----------------------------------------------------------------------


def _worker(root: str, start_id: str, worker: int) -> tuple[str, list[str]]:
    """Resume from a shared checkpoint, fork it, and commit two more steps."""
    log = CheckpointLog(LocalDirectoryStore(root), "shared")
    agent = DurableAgent()
    agent.begin(PROMPTS)
    assert DurableAgent.inspect(log.load(start_id))["executed_nodes"] > 0
    forked = log.fork(start_id, agent.prompt_node_ids[2], f"worker {worker} plan")
    outputs = DurableAgent().resume_from(log.load(forked))
    # Also commit the unforked continuation: identical across workers.
    same = log.commit(log.load(start_id), None)
    return forked, [repr(outputs), same]


def test_two_workers_share_one_log(tmp_path: Path) -> None:
    root = tmp_path / "shared-store"
    agent = DurableAgent()
    agent.begin(PROMPTS)
    log = CheckpointLog(LocalDirectoryStore(root), "shared")
    start = log.commit(agent.run_until_step(1), step=1)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        results = list(pool.map(_worker, [str(root)] * 4, [start] * 4, [0, 1, 0, 1]))

    forks = {r[0] for r in results}
    assert len(forks) == 2  # workers 0 and 1 produced distinct forks; repeats deduplicated
    assert {r[1][1] for r in results} == {start}

    fresh = CheckpointLog(LocalDirectoryStore(root), "shared")
    records = fresh.manifest()
    assert {r.id for r in records} == {start, *forks}
    for r in records:
        fresh.load(r.id)  # every record rebuilds and passes its content hash
    objects = LocalDirectoryStore(root).list("shared/objects/")
    assert len(objects) == len(records)  # losers of a commit race cleaned up
