"""The guest's shell REPL script run on this host under bash (5.1 or later: $SRANDOM): the exec
header and the reseed flag. Requires Linux, no VM."""

import subprocess
from pathlib import Path

import pytest

from tests.conftest import skip_unless_linux

_REPL = Path(__file__).parent.parent / "guest-agent" / "src" / "repl" / "scripts" / "shell_repl.sh"


def _run_execs(*execs: tuple[str, bool]) -> tuple[list[str], list[str]]:
    """Send `execs` (code, reseed flag) to the REPL as the execs s0, s1, ..., as the agent does:
    the header (sentinel id, code length in bytes, reseed flag) and the code of one exec, then
    its sentinel before the next (the agent writes a frame once the last one ended). User code
    here writes no stderr. Returns (printed lines, sentinel lines)."""
    with subprocess.Popen(
        ["bash", "--norc", "--noprofile", str(_REPL)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as repl:
        assert repl.stdin is not None
        assert repl.stdout is not None
        assert repl.stderr is not None
        assert repl.stderr.readline() == b"__REPL_READY__\n"
        sentinels: list[str] = []
        for i, (code, reseed) in enumerate(execs):
            repl.stdin.write(f"s{i} {len(code.encode())} {int(reseed)}\n{code}".encode())
            repl.stdin.flush()
            sentinels.append(repl.stderr.readline().decode().rstrip("\n"))
        repl.stdin.close()
        stdout = repl.stdout.read().decode()
    printed = [line for line in stdout.splitlines() if not line.startswith("__STDOUT_END_")]
    return printed, sentinels


@skip_unless_linux
@pytest.mark.timeout(60)
def test_the_reseed_flag_gives_random_a_new_seed() -> None:
    """RANDOM=7, then an exec without the flag: $RANDOM goes on from the seed, as in a new bash.
    The same with the flag: $RANDOM has a new seed from the kernel ($SRANDOM), another one in
    another REPL."""
    seeded = subprocess.run(
        ["bash", "--norc", "--noprofile", "-c", "RANDOM=7; echo $RANDOM $RANDOM $RANDOM"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()

    printed, sentinels = _run_execs(
        ("echo ${SRANDOM:+has-srandom}", False),  # bash 5.1+: else RANDOM= would seed with 0
        ("RANDOM=7", False),
        ("echo $RANDOM $RANDOM $RANDOM", False),
        ("RANDOM=7", False),
        ("echo $RANDOM $RANDOM $RANDOM", True),
    )
    again, _ = _run_execs(("RANDOM=7", False), ("echo $RANDOM $RANDOM $RANDOM", True))

    assert sentinels == [f"__SENTINEL_s{i}_0__" for i in range(5)]
    assert printed[0] == "has-srandom"
    assert printed[1] == seeded
    assert len(printed) == 3
    assert printed[2] != seeded  # 3 values of 15 bits: equal by chance once in 2**45
    assert again[0] != printed[2]  # another REPL in the same state gets another seed


@skip_unless_linux
@pytest.mark.timeout(60)
def test_the_code_is_read_by_bytes_and_starts_with_a_zero_status() -> None:
    """The header gives the code's length in bytes: a UTF-8 code reads whole, also after user
    code set a UTF-8 locale, and the next exec still reads its own frame. The code starts with
    $? = 0 (the reseed check leaves no status)."""
    printed, sentinels = _run_execs(
        ("echo $?", False),
        ("export LC_ALL=C.UTF-8; echo 'é€'", False),
        ("echo 'ü' $?", True),
    )

    assert sentinels == [f"__SENTINEL_s{i}_0__" for i in range(3)]
    assert printed == ["0", "é€", "ü 0"]
