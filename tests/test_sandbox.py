from codealign.data import extract_code
from codealign.metrics import pass_at_k
from codealign.sandbox import run_many, run_tests

TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]


def test_correct_solution_passes():
    r = run_tests("def add(a, b):\n    return a + b", TESTS)
    assert r.status == "ok" and r.passed == 3 and r.all_passed


def test_partial_credit():
    r = run_tests("def add(a, b):\n    return abs(a) + b", TESTS)
    assert r.status == "fail" and r.passed == 2 and r.frac == 2 / 3


def test_errors_record_why_each_test_failed():
    r = run_tests("def add(a, b):\n    return abs(a) + b", TESTS + ["assert sub(1, 1) == 0"])
    assert r.errors[:3] == [None, None, "AssertionError: "]
    assert r.errors[3] == "NameError: name 'sub' is not defined"
    r = run_tests("import not_a_real_module", TESTS)
    assert r.errors == ["ModuleNotFoundError: No module named 'not_a_real_module'"] * 3


def test_syntax_error():
    r = run_tests("def add(a, b)\n    return a + b", TESTS)
    assert r.status == "error" and r.passed == 0


def test_infinite_loop_times_out_keeps_earlier_passes():
    prog = "def add(a, b):\n    while a < 0: pass\n    return a + b"
    r = run_tests(prog, TESTS, timeout=2)
    assert r.status == "timeout" and r.passed == 2


def test_memory_bomb_is_contained():
    r = run_tests("x = bytearray(10**10)\ndef add(a, b): return a + b", TESTS)
    assert r.status == "error" and r.passed == 0


def test_setup_code_is_available():
    r = run_tests("def f(): return math.pi", ["assert f() > 3"], setup="import math")
    assert r.all_passed


# --- reward hacking attempts: none of these should earn credit ---

def test_printing_fake_results_gets_no_credit():
    prog = 'print(\'{"test": 0, "ok": true, "err": null}\')\nprint("All tests passed")'
    assert run_tests(prog, TESTS).passed == 0


def test_exiting_early_gets_no_credit():
    for prog in ["import sys; sys.exit(0)", "import os; os._exit(0)"]:
        r = run_tests(prog, TESTS)
        assert r.passed == 0 and not r.all_passed


def test_disabling_assert_via_exception_swallowing_still_fails():
    # Model defines add wrongly; tests run in harness scope, not model's try/except.
    prog = "def add(a, b):\n    try:\n        return None\n    except: pass"
    assert run_tests(prog, TESTS).passed == 0


def test_run_many_preserves_order():
    jobs = [{"program": f"def add(a, b): return a + b + {i}", "tests": TESTS} for i in range(8)]
    res = run_many(jobs, workers=4)
    assert [r.passed for r in res] == [3, 0, 0, 0, 0, 0, 0, 0]


def test_extract_code():
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code("```python\na\n```\n```python\nb\n```") == "b\n"
    assert extract_code("sure:\n```python\ndef f():\n    pass") == "def f():\n    pass"
    assert extract_code("def f(): pass") == "def f(): pass"


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-9


def test_interpreter_starts_with_sandbox_env():
    # Regression: on clusters the module-built Python needs LD_LIBRARY_PATH to start.
    # If the sandbox env strips it, every program "fails" and this shows why.
    r = run_tests("x = 1", ["assert x == 1"])
    assert r.all_passed, r.detail
