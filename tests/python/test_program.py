import continuum as ct


def test_program_traces_once():
    calls = {"n": 0}

    @ct.program
    def f(x):
        calls["n"] += 1
        return x

    f(1)
    f(2)
    assert calls["n"] == 1
