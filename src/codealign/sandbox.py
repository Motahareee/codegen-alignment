"""Run untrusted model-generated code against unit tests in a subprocess.

Each program runs in a fresh `python -I` process inside a temp dir, with
CPU/memory limits and a wall-clock timeout. Tests are asserted one at a
time so we can give partial credit (useful as an RL reward).

Reward-hacking guard: the harness reports results on stdout prefixed with a
random nonce the model's code never sees, so a program that simply prints
"all tests passed" gets no credit.

This is a *learning-grade* sandbox, not a security boundary. On a shared
cluster, run it inside a container / job allocation, never on a login node.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# The harness receives the nonce and the payload on stdin, then execs the
# model's code in its own globals dict. Tests run one by one; each result is
# printed immediately so a later infinite loop can't erase earlier passes.
_HARNESS = r"""
import json, resource, sys
def _main():
    data = json.loads(sys.stdin.read())
    nonce = data["nonce"]
    mem, cpu = data["memory_mb"] * 1024 * 1024, data["cpu_seconds"]
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    out = sys.__stdout__
    g = {"__name__": "__main__"}
    try:
        exec(compile(data["setup"] + "\n" + data["program"], "<solution>", "exec"), g)
    except BaseException as e:
        out.write(nonce + json.dumps({"compile_error": type(e).__name__ + ": " + str(e)[:200]}) + "\n")
        out.flush()
        return
    for i, test in enumerate(data["tests"]):
        try:
            exec(test, g)
            ok, err = True, None
        except BaseException as e:
            ok, err = False, type(e).__name__
        out.write(nonce + json.dumps({"test": i, "ok": ok, "err": err}) + "\n")
        out.flush()
_main()
"""


# The interpreter itself needs these to start: module-built Pythons on HPC
# clusters find libpython via LD_LIBRARY_PATH. Everything else is dropped.
_PASSTHROUGH_ENV = ("PATH", "LD_LIBRARY_PATH")


@dataclass
class ExecResult:
    passed: int
    total: int
    status: str  # "ok" | "fail" | "error" | "timeout"
    detail: str = ""

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def frac(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_tests(
    program: str,
    tests: list[str],
    setup: str = "",
    timeout: float = 10.0,
    memory_mb: int = 1024,
) -> ExecResult:
    nonce = secrets.token_hex(16)
    payload = json.dumps({
        "nonce": nonce, "program": program, "tests": tests, "setup": setup,
        "memory_mb": memory_mb, "cpu_seconds": int(timeout) + 1,
    })
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _HARNESS],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp,
            env={**{k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}, "PYTHONHASHSEED": "0"},
            start_new_session=True,  # own process group, so we can kill any children too
            text=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()

    results = {}
    for line in stdout.splitlines():
        if not line.startswith(nonce):
            continue
        msg = json.loads(line[len(nonce):])
        if "compile_error" in msg:
            return ExecResult(0, len(tests), "error", msg["compile_error"])
        results[msg["test"]] = msg["ok"]

    passed = sum(results.values())
    if timed_out:
        return ExecResult(passed, len(tests), "timeout")
    if len(results) < len(tests):  # crashed mid-run (e.g. memory limit, os._exit)
        # stderr tail says why, e.g. a MemoryError, or the interpreter failing to start
        return ExecResult(passed, len(tests), "error", "harness terminated early: " + stderr.strip()[-300:])
    return ExecResult(passed, len(tests), "ok" if passed == len(tests) else "fail")


def run_many(jobs: list[dict], workers: int | None = None, **kwargs) -> list[ExecResult]:
    """Run many (program, tests, setup) jobs concurrently. Order is preserved."""
    workers = workers or os.cpu_count() or 4
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda j: run_tests(**j, **kwargs), jobs))
