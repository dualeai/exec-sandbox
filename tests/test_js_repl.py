"""The JavaScript REPL runs the code as written: import() goes to the main context's module
loader, and no text rewrite changes the user's string literals. Each exec ends with its
own output, whatever user code does to the output streams. Requires VM."""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec


async def test_import_paren_in_a_string_stays_as_written(scheduler: Scheduler) -> None:
    """A string that holds "import(" prints as written; import() still loads a built-in module
    and a file next to the user's code (a relative specifier resolves in /home/user)."""
    code = (
        'console.log("import(x)");'
        ' const path = await import("node:path"); console.log(typeof path.join);'
        ' const mod = await import("./rel_mod.mjs"); console.log(mod.answer);'
    )
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        await session.write_file("rel_mod.mjs", b"export const answer = 42;")
        result = await session.exec(code)

    assert (result.exit_code, result.stdout.splitlines()) == (0, ["import(x)", "function", "42"]), result.stderr


async def test_large_output_ends_with_its_exec(scheduler: Scheduler) -> None:
    """A large stdout and more than 64 KB of stderr (under the 100 KB stderr limit): each exec
    ends with all its own output, and none of it reaches the next exec."""
    big = "console.log('o'.repeat(600000)); process.stderr.write('e'.repeat(90000) + '\\n');"
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        first = await session.exec(big, timeout_seconds=30)
        second = await session.exec("console.log('next')", timeout_seconds=30)

    assert (first.exit_code, len(first.stdout.strip()), len(first.stderr.strip())) == (0, 600000, 90000)
    assert (second.exit_code, second.stdout.strip()) == (0, "next"), second.stderr


async def test_a_replaced_stderr_write_does_not_slow_later_execs(scheduler: Scheduler) -> None:
    """User code replaces process.stderr.write with one that never calls back: later execs
    still end (the sentinel uses the original write; with the live one, each exec would
    run to its timeout)."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        await session.exec("const w = process.stderr.write.bind(process.stderr); process.stderr.write = (s) => w(s);")
        results = [await session.exec("console.log('x')") for _ in range(5)]

    assert [(r.exit_code, r.stdout.strip()) for r in results] == [(0, "x")] * 5


@pytest.mark.parametrize(
    "setup",
    [
        "process.stdout.cork(); process.stdout.uncork = () => {};",
        "Object.defineProperty(process.stdout, 'writableCorked', { get() { return 1; } });",
        "process.stderr.end();",
        "process.stdout.end();",
    ],
    ids=["no-op-uncork", "always-corked", "stderr-ended", "stdout-ended"],
)
async def test_what_user_code_does_to_the_streams_does_not_stop_the_repl(scheduler: Scheduler, setup: str) -> None:
    """User code corks a stream for good or ends it: its exec and the next ones still end (no
    uncork loop; an ended stream's marker or sentinel goes to its fd)."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        first = await session.exec(setup, timeout_seconds=10)
        second = await session.exec("Bun.write(Bun.stderr, 'next\\n')", timeout_seconds=10)

    assert (first.exit_code, second.exit_code) == (0, 0), (first.stderr, second.stderr)


async def test_a_bun_write_that_rejects_unawaited_fails_its_exec(scheduler: Scheduler) -> None:
    """A Bun.write(Bun.stdout) whose source fails (a missing file), not awaited: its rejection
    is unhandled, so the exec fails with the error, as without the REPL's wrapper."""
    code = "Bun.write(Bun.stdout, Bun.file('/nonexistent/source'));"
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        result = await session.exec(code, timeout_seconds=10)
        after = await session.exec("console.log('next')", timeout_seconds=10)

    assert (result.exit_code, "ENOENT" in result.stderr) == (1, True), result.stderr
    assert (after.exit_code, after.stdout.strip()) == (0, "next"), after.stderr


async def test_a_bun_write_through_require_ends_with_its_exec(scheduler: Scheduler) -> None:
    """require('bun').write is the wrapped Bun.write: its unawaited large write ends before the
    exec's marker, and none of it reaches the next exec."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        first = await session.exec("require('bun').write(Bun.stdout, 'o'.repeat(600000));", timeout_seconds=30)
        second = await session.exec("console.log('next')", timeout_seconds=30)

    assert (first.exit_code, len(first.stdout)) == (0, 600000), first.stderr
    assert (second.exit_code, second.stdout.strip()) == (0, "next"), second.stderr


async def test_a_throwing_stderr_write_does_not_hide_a_user_error(scheduler: Scheduler) -> None:
    """User code replaces process.stderr.write with one that throws, then fails: the error
    shows through the original write, and the REPL runs on."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        await session.exec("process.stderr.write = () => { throw new Error('no'); };")
        failed = await session.exec("throw new Error('mine')", timeout_seconds=10)
        after = await session.exec("console.log('next')", timeout_seconds=10)

    assert (failed.exit_code, "mine" in failed.stderr) == (1, True), failed.stderr
    assert (after.exit_code, after.stdout.strip()) == (0, "next"), after.stderr


@pytest.mark.parametrize(
    ("code", "stdout_bytes", "stderr_bytes"),
    [
        ("process.stdout.write('o'.repeat(600000)); process.stdout.end();", 600000, 0),
        ("process.stderr.write('e'.repeat(90000)); process.stderr.end();", 0, 90000),
        (
            "process.stdout.removeAllListeners(); process.stdout.write('o'.repeat(600000)); process.stdout.end();",
            600000,
            0,
        ),
    ],
    ids=["stdout", "stderr", "listeners-removed"],
)
async def test_a_stream_ended_with_queued_output_ends_its_exec_after_it(
    scheduler: Scheduler, code: str, stdout_bytes: int, stderr_bytes: int
) -> None:
    """User code writes much, then ends the stream: the exec gets all of it before its end,
    and none goes into the next exec."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        first = await session.exec(code, timeout_seconds=30)
        second = await session.exec("await Bun.sleep(300); 1", timeout_seconds=30)

    assert (first.exit_code, len(first.stdout), len(first.stderr)) == (0, stdout_bytes, stderr_bytes)
    assert (second.exit_code, second.stdout, second.stderr) == (0, "", "")


@pytest.mark.parametrize(
    ("code", "shown"),
    [
        ("throw Object.create(null)", "null prototype"),
        ("throw { get stack() { throw new Error('no'); } }", "stack: [Getter]"),
        (
            "throw { get stack() { throw 1; }, [Symbol.for('nodejs.util.inspect.custom')]() { throw 2; },"
            " get [Symbol.toStringTag]() { throw 3; } }",
            "cannot be shown",
        ),
        ("Promise.reject(new Error('plain')); 1", "plain"),
    ],
    ids=["no-prototype", "throwing-stack", "everything-throws", "unhandled-rejection"],
)
async def test_any_error_fails_its_exec_and_the_repl_runs_on(scheduler: Scheduler, code: str, shown: str) -> None:
    """Whatever user code throws, or leaves rejected and unhandled, its exec fails with the
    error shown, and the next exec runs."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        failed = await session.exec(code, timeout_seconds=10)
        after = await session.exec("console.log('next')", timeout_seconds=10)

    assert (failed.exit_code, shown in failed.stderr) == (1, True), failed.stderr
    assert (after.exit_code, after.stdout.strip()) == (0, "next"), after.stderr


async def test_user_listeners_on_the_streams_get_no_warning(scheduler: Scheduler) -> None:
    """The REPL adds no listener to the output streams: user code may add its own up to the
    default limit with no MaxListenersExceededWarning."""
    async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
        result = await session.exec("for (let i = 0; i < 10; i++) process.stdout.on('close', () => {}); 1")
        after = await session.exec("await Bun.sleep(50); 1")

    assert (result.exit_code, result.stderr, after.stderr) == (0, "", ""), (result.stderr, after.stderr)
