# Echo backend plugin

A complete out-of-tree Continuum backend in ~150 lines of C, built against only
the public header `include/continuum/backend/backend_abi.h`.

```bash
cmake -S examples/plugins/echo_backend -B build-echo
cmake --build build-echo            # -> build-echo/libcontinuum_echo_backend.so
```

Load it from Python:

```python
from continuum._native import BackendRegistry, Session, check_backend

print(check_backend("build-echo/libcontinuum_echo_backend.so"))   # conformance report

reg = BackendRegistry()
reg.load_plugin("echo", "build-echo/libcontinuum_echo_backend.so", priority=100)
Session("demo", reg).generate(["hello"], "echo/model")               # 'echo: hello'
```

or point the runtime at it without code changes:

```bash
export CONTINUUM_BACKEND_PLUGINS="echo@100=$PWD/build-echo/libcontinuum_echo_backend.so"
python -m continuum.backend_check echo          # or any plugin path
```

See "Backend plugins" in `docs/design/abi.md` for the contract.
