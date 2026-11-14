"""Tests for PYTHONINTMAXSTRDIGITS=0 in the Python REPL.

Python 3.11+ limits int<->str conversions to 4300 digits by default (CVE-2020-10735).
The sandbox disables this limit via PYTHONINTMAXSTRDIGITS=0 because execution timeouts
and output caps already bound resource usage.

The tests check the setting and its effect at the first value the default limit refuses.
How CPython converts with the limit off (repr, f-strings, hex) is CPython's behavior, not
tested here.

References:
- CVE-2020-10735: https://python-security.readthedocs.io/vuln/large-int-str-dos.html
- CPython docs: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONINTMAXSTRDIGITS
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


class TestIntStrDigits:
    """The REPL interpreter starts with the int<->str digit limit off."""

    async def test_limit_is_disabled(self, scheduler: Scheduler) -> None:
        """PYTHONINTMAXSTRDIGITS=0 is set (0 means unlimited)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "import sys; print(sys.get_int_max_str_digits())"
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "0"

    async def test_4301_digits(self, scheduler: Scheduler) -> None:
        """4301 digits, str to int and back: the first value the default limit refuses."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "x = int('9' * 4301); print(len(str(x)))")

        assert result.exit_code == 0
        assert result.stdout.strip() == "4301"

    async def test_fibonacci_large(self, scheduler: Scheduler) -> None:
        """Fibonacci(100000) — the originally reported user scenario."""
        code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

result = str(fib(100000))
print(len(result))
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)

        assert result.exit_code == 0
        assert result.stdout.strip() == "20899"  # fib(100000) has 20899 digits

    async def test_user_env_var_cannot_override_limit(self, scheduler: Scheduler) -> None:
        """User env_vars are injected as Python code (os.environ), too late to affect
        PYTHONINTMAXSTRDIGITS which is read at interpreter startup."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            "import sys; print(sys.get_int_max_str_digits())",
            env_vars={"PYTHONINTMAXSTRDIGITS": "1000"},
        )

        assert result.exit_code == 0
        # Still 0 — env_vars are set via os.environ at runtime, not at process spawn
        assert result.stdout.strip() == "0"
