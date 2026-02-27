"""Tests for PYTHONINTMAXSTRDIGITS=0 in the Python REPL.

Python 3.11+ limits int<->str conversions to 4300 digits by default (CVE-2020-10735).
The sandbox disables this limit via PYTHONINTMAXSTRDIGITS=0 because execution timeouts
and output caps already bound resource usage.

References:
- CVE-2020-10735: https://python-security.readthedocs.io/vuln/large-int-str-dos.html
- CPython docs: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONINTMAXSTRDIGITS
"""

import pytest

from exec_sandbox.exceptions import OutputLimitError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Normal Cases: large int conversions that require the limit to be lifted
# =============================================================================
class TestIntStrDigitsNormal:
    """Verify that int<->str conversions beyond 4300 digits work."""

    async def test_limit_is_disabled(self, scheduler: Scheduler) -> None:
        """PYTHONINTMAXSTRDIGITS=0 is set (0 means unlimited)."""
        result = await scheduler.run(
            code="import sys; print(sys.get_int_max_str_digits())",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "0"

    async def test_int_to_str_5000_digits(self, scheduler: Scheduler) -> None:
        """Convert a 5000-digit integer to string (above default 4300 limit)."""
        result = await scheduler.run(
            code="print(len(str(10**4999)))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "5000"

    async def test_str_to_int_5000_digits(self, scheduler: Scheduler) -> None:
        """Convert a 5000-digit string to integer (above default 4300 limit)."""
        result = await scheduler.run(
            code="x = int('9' * 5000); print(len(str(x)))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "5000"

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
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0
        # fib(100000) has 20899 digits
        digit_count = int(result.stdout.strip())
        assert digit_count > 20000


# =============================================================================
# Edge Cases: boundary values around the default 4300-digit limit
# =============================================================================
class TestIntStrDigitsEdgeCases:
    """Boundary conditions at and around the default 4300-digit limit."""

    async def test_exactly_4300_digits(self, scheduler: Scheduler) -> None:
        """Exactly 4300 digits — would be the max under default limit."""
        result = await scheduler.run(
            code="x = int('9' * 4300); print(len(str(x)))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "4300"

    async def test_4301_digits(self, scheduler: Scheduler) -> None:
        """4301 digits — first value that breaks under the default limit."""
        result = await scheduler.run(
            code="x = int('9' * 4301); print(len(str(x)))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "4301"

    async def test_power_of_two_base_unaffected(self, scheduler: Scheduler) -> None:
        """hex/bin/oct are power-of-2 bases, never limited by CVE-2020-10735."""
        code = """
x = 2**100000
print(len(hex(x)))
print(len(bin(x)))
print(len(oct(x)))
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 3
        # hex: ~25002 chars, bin: ~100002 chars, oct: ~33335 chars
        assert all(int(line) > 1000 for line in lines)

    async def test_arithmetic_without_conversion(self, scheduler: Scheduler) -> None:
        """Pure integer arithmetic was never limited — confirm no regression."""
        code = """
x = 10**50000
y = x * x
z = y + x
print("OK", len(str(z)))
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0
        assert "OK" in result.stdout


# =============================================================================
# Weird Cases: unusual conversion paths that use the same int<->str machinery
# =============================================================================
class TestIntStrDigitsWeird:
    """Unusual conversion paths that exercise the int<->str limit."""

    async def test_roundtrip_str_int_str(self, scheduler: Scheduler) -> None:
        """str->int->str round-trip with 10000-digit number."""
        code = """
original = '9' * 10000
result = str(int(original))
print(result == original)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "True" in result.stdout

    async def test_repr_large_int(self, scheduler: Scheduler) -> None:
        """repr() goes through the same int->str path."""
        result = await scheduler.run(
            code="print(len(repr(10**5000)))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "5001"

    async def test_fstring_large_int(self, scheduler: Scheduler) -> None:
        """f-string formatting triggers int->str conversion."""
        result = await scheduler.run(
            code="x = 10**5000; print(len(f'{x}'))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "5001"

    async def test_format_large_int(self, scheduler: Scheduler) -> None:
        """format() with decimal spec triggers int->str conversion."""
        result = await scheduler.run(
            code="x = 10**5000; print(len(format(x, 'd')))",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "5001"

    async def test_value_correctness(self, scheduler: Scheduler) -> None:
        """Verify converted value is correct, not just that it runs."""
        code = """
x = 10**5000 + 42
s = str(x)
print(s[-2:])
print(s[0])
print(len(s))
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "42"  # last 2 digits
        assert lines[1] == "1"  # first digit
        assert lines[2] == "5001"  # total length


# =============================================================================
# Out of Bounds: stress tests bounded by execution timeout and output caps
# =============================================================================
class TestIntStrDigitsOutOfBounds:
    """Stress tests — resource limits (timeout, stdout cap) protect against abuse."""

    @pytest.mark.slow
    async def test_100k_digit_conversion(self, scheduler: Scheduler) -> None:
        """100,000-digit int->str conversion (well above default limit)."""
        code = """
x = 10**99999
s = str(x)
print(len(s))
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=60,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "100000"

    async def test_large_output_hits_stdout_cap(self, scheduler: Scheduler) -> None:
        """Printing a ~1M-digit number exceeds guest-agent 1MB stdout limit."""
        code = """
x = 10**999999
print(x)
"""
        with pytest.raises(OutputLimitError):
            await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=60,
            )

    async def test_user_env_var_cannot_override_limit(self, scheduler: Scheduler) -> None:
        """User env_vars are injected as Python code (os.environ), too late to affect
        PYTHONINTMAXSTRDIGITS which is read at interpreter startup."""
        result = await scheduler.run(
            code="import sys; print(sys.get_int_max_str_digits())",
            language=Language.PYTHON,
            env_vars={"PYTHONINTMAXSTRDIGITS": "1000"},
        )

        assert result.exit_code == 0
        # Still 0 — env_vars are set via os.environ at runtime, not at process spawn
        assert result.stdout.strip() == "0"

    async def test_user_can_re_enable_limit_via_code(self, scheduler: Scheduler) -> None:
        """User can re-enable the limit programmatically if desired."""
        code = """
import sys
sys.set_int_max_str_digits(1000)
try:
    print(str(10**1500))
    print("NO_ERROR")
except ValueError:
    print("VALUE_ERROR")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "VALUE_ERROR" in result.stdout
