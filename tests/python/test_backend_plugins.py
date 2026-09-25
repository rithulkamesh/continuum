"""Runtime backend plugins (#5) and the conformance kit (#6) from Python."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from continuum import backend_check
from continuum._native import BackendRegistry, Session, check_backend

ROOT = Path(__file__).resolve().parents[2]
ECHO_SRC = ROOT / "examples" / "plugins" / "echo_backend" / "echo_backend.c"


@pytest.fixture(scope="module")
def echo_plugin(tmp_path_factory: pytest.TempPathFactory) -> str:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler to build the example plugin")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = tmp_path_factory.mktemp("plugin") / f"libcontinuum_echo_backend{suffix}"
    flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(
        [cc, *flags, "-std=c99", "-O2", "-I", str(ROOT / "include"), str(ECHO_SRC), "-o", str(out)],
        check=True,
    )
    return str(out)


@pytest.mark.parametrize("name", ["fake", "libtorch", "vllm"])
def test_builtin_backends_conform(name: str) -> None:
    report = check_backend(name)
    assert report["passed"], report["summary"]


def test_plugin_conforms_and_round_trips_state(echo_plugin: str) -> None:
    report = check_backend(echo_plugin)
    assert report["passed"], report["summary"]
    status = {c["name"]: c["status"] for c in report["checks"]}
    assert status["state.roundtrip"] == "pass"
    assert status["tensor.*"] == "skip"


def test_plugin_serves_session_generate(echo_plugin: str) -> None:
    reg = BackendRegistry()
    reg.register_fake_llm()
    reg.load_plugin("echo", echo_plugin, priority=100)
    assert set(reg.names()) == {"fake", "echo"}
    session = Session("plugin", reg)
    assert session.generate(["hello"], "echo/model") == "echo: hello"


def test_plugins_from_env(echo_plugin: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_BACKEND_PLUGINS", f"echo@5={echo_plugin}")
    reg = BackendRegistry()
    assert reg.load_plugins_from_env() == 1
    assert reg.has("echo")
    assert check_backend("echo")["passed"]  # resolved through the env var


def test_bad_targets_raise() -> None:
    with pytest.raises(RuntimeError, match="unknown backend"):
        check_backend("definitely-not-a-backend")
    with pytest.raises(RuntimeError, match="cannot load backend plugin"):
        BackendRegistry().load_plugin("x", "/nonexistent/libx.so")


def test_cli(echo_plugin: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert backend_check.main([echo_plugin]) == 0
    assert "CONFORMANT" in capsys.readouterr().out
    assert backend_check.main(["fake", "--json", "--nondeterministic"]) == 0
    out = capsys.readouterr().out
    assert '"passed": true' in out
    assert "disabled by options" in out
