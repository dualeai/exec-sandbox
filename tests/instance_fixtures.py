"""Real-VM saved-instance tests: per-test store and folders, VM origins, and a guest state probe.

The probe reads the guest state in one exec (variables, the digests of the files
in /home/user/state, a ticker process and its tick file) and the host keeps a
model of what the state must be. Tests compare the two with ==.

Origins (asserted on every session from StartupTiming; a wrong origin fails):
- cold: a private, empty memory-snapshot dir and no warm pool -> (F, F)
- l1: a second Scheduler on that dir after the first exited (its exit awaits
  the background L1 save), or the primed dir -> (F, T)
- pool: the primed dir and warm_pool_size=1 -> (T, T)
- reuse: Scheduler.reuse() -> (F, F) and instance_load_ms set

Fixtures: import `instance_world_fixture` and `primed_l1_dir_fixture` into a test
module (``# noqa: F401``); tests request them as `instance_world` and `primed_l1_dir`.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import socket
import tempfile
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest

from exec_sandbox import (
    InstanceNotSavedWarning,
    InstanceRef,
    Language,
    SandboxError,
    Scheduler,
    SchedulerConfig,
    Session,
    VmSpec,
)
from tests.instance_support import new_keyring
from tests.s3_faults import Bucket

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

STATE_DIR = "/home/user/state"
"""Files the probe digests (write_file paths are relative to /home/user)."""
TICKER = ": tickmark; i=0; while :; do i=$((i+1)); echo $i >> /home/user/tick; sleep 0.1; done"
"""One line per tick: the lines stay 1..n only if RAM and /home/user come from one paused instant."""
VAR_NAMES = ("pv0", "pv1", "pv2", "plost")
UNSET = "__unset__"
BIG_FILE_BYTES = 1 << 20

L2_PACKAGES = {Language.PYTHON: ("six==1.17.0",), Language.JAVASCRIPT: ("lodash@4.17.21",)}
"""One small package per language (both in the bundled allow-lists); raw has no allow-list."""
_L2_IMPORT = {
    Language.PYTHON: ("import six; print(six.__version__)", "1.17.0"),
    Language.JAVASCRIPT: ("console.log(require('lodash').VERSION)", "4.17.21"),
}
_L2_REGISTRY = {Language.PYTHON: "pypi.org", Language.JAVASCRIPT: "registry.npmjs.org"}

Origin = Literal["cold", "l1", "pool", "reuse"]
_ORIGIN_FLAGS: dict[Origin, tuple[bool, bool]] = {
    "cold": (False, False),
    "l1": (False, True),
    "pool": (True, True),
    "reuse": (False, False),
}


# ============================================================================
# Store, folders and Schedulers of one test
# ============================================================================


@dataclass(frozen=True, kw_only=True)
class InstanceWorld:
    """One test's instance store (moto bucket, fresh prefix, own keyring) and local folders."""

    base: SchedulerConfig
    endpoint_url: str
    bucket: Bucket
    """The moto bucket under this test's fresh prefix."""
    keyring: str
    home_root: Path
    """EXEC_SANDBOX_HOME_DISK_DIR: each VmManager's folder, with its instances/ work folder, is under it."""
    asset_dir: Path
    """EXEC_SANDBOX_INSTANCE_ASSET_DIR: guest images downloaded by digest."""
    tmp: Path

    @contextlib.asynccontextmanager
    async def scheduler(
        self, *, memory_snapshot_cache_dir: Path, warm_pool_size: int, disk_snapshot_cache_dir: Path | None
    ) -> AsyncIterator[Scheduler]:
        """A Scheduler on the test config plus the store; disk_snapshot_cache_dir None keeps the
        machine's L2 cache."""
        fields: dict[str, Any] = {
            **self.base.model_dump(exclude_unset=True),
            "instance_store": "s3",
            "s3_bucket": self.bucket.name,
            "s3_endpoint_url": self.endpoint_url,
            "s3_instance_prefix": self.bucket.prefix,
            "instance_keyring": self.keyring,
            "memory_snapshot_cache_dir": memory_snapshot_cache_dir,
            "warm_pool_size": warm_pool_size,
        }
        if disk_snapshot_cache_dir is not None:
            fields["disk_snapshot_cache_dir"] = disk_snapshot_cache_dir
        async with Scheduler(SchedulerConfig.model_validate(fields)) as scheduler:
            yield scheduler

    def private_dir(self, name: str) -> Path:
        """A new empty folder of this test (a private L1 or L2 cache): 0700 whatever the umask,
        as the library makes its own cache folders (it refuses one others can write)."""
        path = self.tmp / f"{name}-{uuid.uuid4().hex}"
        path.mkdir(mode=0o700, parents=True)
        return path

    def instance_folders(self, instance_id: str | None = None) -> list[Path]:
        """Local instance folders of every live VmManager of this test (of one instance, or all)."""
        pattern = f"{instance_id}-*" if instance_id is not None else "*"
        return sorted(self.home_root.glob(f"vmm-*/instances/{pattern}"))

    def keys(self, sub: str) -> list[str]:
        """Every store key under prefix + sub, relative to the prefix."""
        return self.bucket.listing(sub)

    def instance_keys(self, instance_id: str) -> set[str]:
        """Head and body object names of one instance ("head", "b/<commit>.<f|p>")."""
        own = f"s/{instance_id}/"
        return {key[len(own) :] for key in self.keys(own)}


@pytest.fixture(name="instance_world")
def instance_world_fixture(
    scheduler_config: SchedulerConfig, moto_s3_server: tuple[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[InstanceWorld]:
    """Per test: moto bucket (in process), fresh prefix and keyring, home-disk root, asset dir."""
    endpoint_url, s3 = moto_s3_server
    home_root = Path(tempfile.mkdtemp(prefix="hd-"))  # short and 0700 (PrivateRunDir checks the root)
    asset_dir = tmp_path / "assets"
    monkeypatch.setenv("EXEC_SANDBOX_HOME_DISK_DIR", str(home_root))
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_ASSET_DIR", str(asset_dir))
    try:
        yield InstanceWorld(
            base=scheduler_config,
            endpoint_url=endpoint_url,
            bucket=Bucket(s3),  # a fresh inst-<uuid>/ prefix
            keyring=new_keyring(),
            home_root=home_root,
            asset_dir=asset_dir,
            tmp=tmp_path,
        )
    finally:
        shutil.rmtree(home_root, ignore_errors=True)


@pytest.fixture(name="primed_l1_dir")
async def primed_l1_dir_fixture(images_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """L1 memory snapshots of the three default specs, made once per run for every xdist worker.

    One warm-pool start (warm_pool_size=1) on an empty dir saves an L1 per language
    (WarmVMPool._boot_and_add_vm, index 0). A file lock in the run's base temp dir
    makes the other workers wait for it, then use it.
    """
    base = tmp_path_factory.getbasetemp()
    root = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    directory = root / "instance-l1-primed"
    ready = directory / "ready"
    fd = os.open(root / "instance-l1-primed.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        if not ready.exists():
            directory.mkdir(mode=0o700, exist_ok=True)  # an L1 cache folder: private whatever the umask
            config = SchedulerConfig(
                images_dir=images_dir,
                auto_download_assets=False,
                warm_pool_size=1,
                memory_snapshot_cache_dir=directory,
            )
            async with Scheduler(config) as scheduler:
                await scheduler.wait_pool_ready()
            saved = sorted(p.name for p in directory.glob("*.vmstate"))
            assert len(saved) == len(Language), f"the warm pool saved {saved}, not one L1 per language"
            ready.touch()
    finally:
        os.close(fd)  # releases the lock
    return directory


# ============================================================================
# Origins, close, L2
# ============================================================================


def assert_origin(session: Session, origin: Origin) -> None:
    """How the VM started, from StartupTiming: a wrong origin fails the test."""
    timing = session.startup_timing
    assert (timing.warm_pool_hit, timing.l1_cache_hit) == _ORIGIN_FLAGS[origin], f"wanted {origin}: {timing}"
    assert (timing.instance_load_ms is not None) == (origin == "reuse"), f"wanted {origin}: {timing}"


async def close_saved(session: Session) -> None:
    """Close a session saved after its last change: no InstanceNotSavedWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await session.close()
    assert not [w for w in caught if issubclass(w.category, InstanceNotSavedWarning)]


async def close_unsaved(session: Session) -> None:
    """Close a session of an instance with changes since its last save: InstanceNotSavedWarning."""
    with pytest.warns(InstanceNotSavedWarning):
        await session.close()


def _reachable(host: str) -> bool:
    try:
        socket.create_connection((host, 443), timeout=5).close()
    except OSError:
        return False
    return True


async def new_session(scheduler: Scheduler, spec: VmSpec) -> Session:
    """scheduler.new(spec). An L2 build that fails while its registry is unreachable skips the test."""
    try:
        return await scheduler.new(spec)
    except SandboxError as e:
        registry = _L2_REGISTRY.get(spec.language)
        if spec.packages and registry is not None and not await asyncio.to_thread(_reachable, registry):
            pytest.skip(f"the L2 of {list(spec.packages)} is not cached and its build needs {registry}: {e}")
        raise


async def assert_packages_import(session: Session) -> None:
    code, version = _L2_IMPORT[session.spec.language]
    result = await session.exec(code)
    assert (result.exit_code, result.stdout.strip()) == (0, version), result.stderr


# ============================================================================
# Guest state: language drivers, the host model, the probe
# ============================================================================


def _seeded(seed: str, n: int) -> bytes:
    return random.Random(seed).randbytes(n)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RoundOps:
    """What one round does in the guest (the host writes the big file with write_file)."""

    number: int
    var: str
    token: str
    note: str
    truncate: tuple[str, int] | None
    delete: str | None


class Driver:
    """Guest code for one language. Every mutating snippet prints "ok" last."""

    language: Language

    def init(self) -> str:
        raise NotImplementedError

    def round(self, ops: RoundOps) -> str:
        raise NotImplementedError

    def start_ticker(self) -> str:
        raise NotImplementedError

    def lost(self, *, kill_pid: int, delete: str) -> str:
        raise NotImplementedError

    def probe(self, *, ticker_pid: int, wait_ticks: int) -> str:
        raise NotImplementedError


_PY_PROBE = r"""
import hashlib as _hl, os as _os, time as _tm
_g = globals()
_out = []
for _n in __NAMES__:
    _out.append('var\t%s\t%s' % (_n, _g[_n] if _n in _g else '__unset__'))
_out.append('items\t' + ' '.join(map(str, _g.get('pitems', []))))
_d = '/home/user/state'
for _f in sorted(_os.listdir(_d)) if _os.path.isdir(_d) else []:
    with open(_os.path.join(_d, _f), 'rb') as _fh:
        _out.append('file\t%s\t%s' % (_f, _hl.sha256(_fh.read()).hexdigest()))
def _ticks():
    try:
        with open('/home/user/tick') as _fh:
            return _fh.read().split('\n')[:-1]
    except FileNotFoundError:
        return []
_t0 = len(_ticks())
_end = _tm.monotonic() + 20
while len(_ticks()) < _t0 + __WAIT__ and _tm.monotonic() < _end:
    _tm.sleep(0.05)
_tk = _ticks()
_out.append('ticks\t%d\t%d\t%d' % (_t0, len(_tk), _tk == [str(_i) for _i in range(1, len(_tk) + 1)]))
def _cmd(_p):
    try:
        with open('/proc/%s/cmdline' % _p, 'rb') as _fh:
            return _fh.read().replace(b'\0', b' ').decode().strip()
    except OSError:
        return ''
_cmds = [_cmd(_p) for _p in _os.listdir('/proc') if _p.isdigit()]
_out.append('ticker\t%d' % ('tickmark' in _cmd(__PID__)))
_out.append('sleepers\t%d' % sum(_c == 'sleep 7771' for _c in _cmds))
print('\n'.join(_out))
"""


class PythonDriver(Driver):
    language = Language.PYTHON

    def init(self) -> str:
        return "pitems = []\nprint('ok')"

    def round(self, ops: RoundOps) -> str:
        lines = [
            f"{ops.var} = {ops.token!r}",
            f"pitems.append({ops.number})",
            "import os as _os",
            f"_os.makedirs({STATE_DIR!r}, exist_ok=True)",
            f"with open('{STATE_DIR}/{ops.note}', 'w') as _fh:",
            f"    _fh.write({ops.token!r})",
        ]
        if ops.truncate is not None:
            lines.append(f"_os.truncate('{STATE_DIR}/{ops.truncate[0]}', {ops.truncate[1]})")
        if ops.delete is not None:
            lines.append(f"_os.remove('{STATE_DIR}/{ops.delete}')")
        return "\n".join([*lines, "print('ok')"])

    def start_ticker(self) -> str:
        return (
            "import subprocess as _sp\n"
            f"_ticker = _sp.Popen(['sh', '-c', {TICKER!r}], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,"
            " stderr=_sp.DEVNULL, start_new_session=True)\n"
            "print(_ticker.pid)"
        )

    def lost(self, *, kill_pid: int, delete: str) -> str:
        return (
            "pv0 = 'lost'\nplost = 'lost'\nimport os as _os, signal as _sig, subprocess as _sp\n"
            f"with open('{STATE_DIR}/lost.txt', 'w') as _fh:\n    _fh.write('lost')\n"
            f"_os.remove('{STATE_DIR}/{delete}')\n"
            f"_os.kill({kill_pid}, _sig.SIGKILL)\n"
            "_sp.Popen(['sleep', '7771'], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,"
            " start_new_session=True)\n"
            "print('ok')"
        )

    def probe(self, *, ticker_pid: int, wait_ticks: int) -> str:
        return (
            _PY_PROBE.replace("__NAMES__", json.dumps(list(VAR_NAMES)))
            .replace("__WAIT__", str(wait_ticks))
            .replace("__PID__", str(ticker_pid))
        )


_JS_PROBE = r"""
const _fs = require('fs');
const _cr = require('crypto');
const _out = [];
for (const _n of __NAMES__) _out.push('var\t' + _n + '\t' + (_n in globalThis ? String(globalThis[_n]) : '__unset__'));
_out.push('items\t' + (globalThis.pitems || []).join(' '));
const _d = '/home/user/state';
if (_fs.existsSync(_d)) {
    for (const _f of _fs.readdirSync(_d).sort()) {
        _out.push('file\t' + _f + '\t' + _cr.createHash('sha256').update(_fs.readFileSync(_d + '/' + _f)).digest('hex'));
    }
}
const _ticks = () => {
    try { return _fs.readFileSync('/home/user/tick', 'utf8').split('\n').slice(0, -1); } catch (e) { return []; }
};
const _t0 = _ticks().length;
const _end = Date.now() + 20000;
while (_ticks().length < _t0 + __WAIT__ && Date.now() < _end) await new Promise((r) => setTimeout(r, 50));
const _tk = _ticks();
_out.push('ticks\t' + _t0 + '\t' + _tk.length + '\t' + (_tk.every((v, i) => v === String(i + 1)) ? 1 : 0));
const _cmd = (p) => {
    try { return _fs.readFileSync('/proc/' + p + '/cmdline', 'utf8').split('\0').join(' ').trim(); } catch (e) { return ''; }
};
const _cmds = _fs.readdirSync('/proc').filter((p) => /^\d+$/.test(p)).map(_cmd);
_out.push('ticker\t' + (_cmd(__PID__).includes('tickmark') ? 1 : 0));
_out.push('sleepers\t' + _cmds.filter((c) => c === 'sleep 7771').length);
console.log(_out.join('\n'));
"""


class JavascriptDriver(Driver):
    language = Language.JAVASCRIPT

    def init(self) -> str:
        return "globalThis.pitems = [];\nconsole.log('ok');"

    def round(self, ops: RoundOps) -> str:
        lines = [
            f"globalThis.{ops.var} = {json.dumps(ops.token)};",
            f"globalThis.pitems.push({ops.number});",
            "const _fs = require('fs');",
            f"_fs.mkdirSync({json.dumps(STATE_DIR)}, {{ recursive: true }});",
            f"_fs.writeFileSync('{STATE_DIR}/{ops.note}', {json.dumps(ops.token)});",
        ]
        if ops.truncate is not None:
            lines.append(f"_fs.truncateSync('{STATE_DIR}/{ops.truncate[0]}', {ops.truncate[1]});")
        if ops.delete is not None:
            lines.append(f"_fs.unlinkSync('{STATE_DIR}/{ops.delete}');")
        return "\n".join([*lines, "console.log('ok');"])

    def start_ticker(self) -> str:
        return (
            f"const _ticker = require('child_process').spawn('sh', ['-c', {json.dumps(TICKER)}],"
            " { detached: true, stdio: 'ignore' });\n"
            "_ticker.unref();\nconsole.log(_ticker.pid);"
        )

    def lost(self, *, kill_pid: int, delete: str) -> str:
        return (
            "globalThis.pv0 = 'lost';\nglobalThis.plost = 'lost';\nconst _fs = require('fs');\n"
            f"_fs.writeFileSync('{STATE_DIR}/lost.txt', 'lost');\n"
            f"_fs.unlinkSync('{STATE_DIR}/{delete}');\n"
            f"process.kill({kill_pid}, 'SIGKILL');\n"
            "require('child_process').spawn('sleep', ['7771'], { detached: true, stdio: 'ignore' }).unref();\n"
            "console.log('ok');"
        )

    def probe(self, *, ticker_pid: int, wait_ticks: int) -> str:
        return (
            _JS_PROBE.replace("__NAMES__", json.dumps(list(VAR_NAMES)))
            .replace("__WAIT__", str(wait_ticks))
            .replace("__PID__", str(ticker_pid))
        )


_SH_PROBE = r"""
for _n in __NAMES__; do eval "_v=\${$_n-__unset__}"; printf 'var\t%s\t%s\n' "$_n" "$_v"; done
printf 'items\t%s\n' "$(echo $pitems)"
if [ -d /home/user/state ]; then
    for _f in $(ls /home/user/state); do
        printf 'file\t%s\t%s\n' "$_f" "$(sha256sum < "/home/user/state/$_f" | cut -d' ' -f1)"
    done
fi
_ticks() { if [ -f /home/user/tick ]; then echo $(( $(wc -l < /home/user/tick) )); else echo 0; fi; }
_t0=$(_ticks); _i=0
while [ "$(_ticks)" -lt $((_t0 + __WAIT__)) ] && [ "$_i" -lt 400 ]; do sleep 0.05; _i=$((_i + 1)); done
_n=$(_ticks)
if [ "$_n" -eq 0 ]; then _ok=1; else
    _ok=$(head -n "$_n" /home/user/tick | awk -v n="$_n" '$0 != NR { b = 1 } END { print (b || NR != n) ? 0 : 1 }')
fi
printf 'ticks\t%s\t%s\t%s\n' "$_t0" "$_n" "$_ok"
_cmd() { tr '\0' ' ' 2>/dev/null < "/proc/$1/cmdline"; }
case "$(_cmd __PID__ 2>/dev/null)" in *tickmark*) printf 'ticker\t1\n' ;; *) printf 'ticker\t0\n' ;; esac
_s=0
for _p in /proc/[0-9]*; do
    case "$(_cmd "${_p#/proc/}" 2>/dev/null)" in "sleep 7771 ") _s=$((_s + 1)) ;; esac
done
printf 'sleepers\t%s\n' "$_s"
"""


class RawDriver(Driver):
    language = Language.RAW

    def init(self) -> str:
        return "pitems='' && echo ok"

    def round(self, ops: RoundOps) -> str:
        steps = [
            f"{ops.var}='{ops.token}'",
            f'pitems="$pitems {ops.number}"',
            f"mkdir -p {STATE_DIR}",
            f"printf %s '{ops.token}' > {STATE_DIR}/{ops.note}",
        ]
        if ops.truncate is not None:
            steps.append(f"truncate -s {ops.truncate[1]} {STATE_DIR}/{ops.truncate[0]}")
        if ops.delete is not None:
            steps.append(f"rm {STATE_DIR}/{ops.delete}")
        return " && ".join([*steps, "echo ok"])

    def start_ticker(self) -> str:
        return f"sh -c '{TICKER}' </dev/null >/dev/null 2>&1 &\necho $!"

    def lost(self, *, kill_pid: int, delete: str) -> str:
        return (
            f"pv0=lost && plost=lost && printf lost > {STATE_DIR}/lost.txt && rm {STATE_DIR}/{delete}"
            f" && kill -9 {kill_pid} && echo ok\n"
            "sleep 7771 </dev/null >/dev/null 2>&1 &"
        )

    def probe(self, *, ticker_pid: int, wait_ticks: int) -> str:
        return (
            _SH_PROBE.replace("__NAMES__", " ".join(VAR_NAMES))
            .replace("__WAIT__", str(wait_ticks))
            .replace("__PID__", str(ticker_pid))
        )


DRIVERS: dict[Language, Driver] = {
    Language.PYTHON: PythonDriver(),
    Language.JAVASCRIPT: JavascriptDriver(),
    Language.RAW: RawDriver(),
}


@dataclass(frozen=True, kw_only=True)
class GuestState:
    """The compared part of the guest state."""

    variables: dict[str, str]
    items: tuple[int, ...]
    files: dict[str, str]
    """File name in /home/user/state -> sha256."""


@dataclass(frozen=True, kw_only=True)
class Probe:
    state: GuestState
    ticker_alive: bool
    """The ticker runs under the pid it got when it started."""
    ticks_at_start: int
    ticks: int
    ticks_contiguous: bool
    """The tick file holds exactly the lines 1..ticks."""
    sleepers: int
    """Processes whose command line is `sleep 7771`."""


def parse_probe(stdout: str) -> Probe:
    variables: dict[str, str] = {}
    files: dict[str, str] = {}
    fields: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        kind, *rest = line.split("\t")
        if kind == "var":
            variables[rest[0]] = rest[1]
        elif kind == "file":
            files[rest[0]] = rest[1]
        else:
            fields[kind] = rest
    at_start, ticks, contiguous = (int(v) for v in fields["ticks"])
    return Probe(
        state=GuestState(variables=variables, items=tuple(int(v) for v in fields["items"][0].split()), files=files),
        ticker_alive=fields["ticker"] == ["1"],
        ticks_at_start=at_start,
        ticks=ticks,
        ticks_contiguous=contiguous == 1,
        sleepers=int(fields["sleepers"][0]),
    )


@dataclass(kw_only=True)
class GuestModel:
    """What the guest state must be: the host side of the probe. Seeded: same seed, same data."""

    driver: Driver
    seed: str
    variables: dict[str, str] = field(default_factory=lambda: dict.fromkeys(VAR_NAMES, UNSET))
    items: list[int] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)
    ticker_pid: int = 0
    """0: no ticker."""
    rounds: int = 0

    def expected(self) -> GuestState:
        return GuestState(
            variables=dict(self.variables),
            items=tuple(self.items),
            files={name: _sha256(data) for name, data in self.files.items()},
        )

    def fork(self, seed: str) -> GuestModel:
        """A copy that continues with other data (a second session of the same saved state)."""
        return GuestModel(
            driver=self.driver,
            seed=seed,
            variables=dict(self.variables),
            items=list(self.items),
            files=dict(self.files),
            ticker_pid=self.ticker_pid,
            rounds=self.rounds,
        )


def new_model(language: Language, seed: str) -> GuestModel:
    return GuestModel(driver=DRIVERS[language], seed=seed)


async def _run_ok(session: Session, code: str) -> str:
    result = await session.exec(code)
    assert result.exit_code == 0, (result.exit_code, result.stdout, result.stderr)
    return result.stdout


async def start_guest(session: Session, model: GuestModel, *, ticker: bool) -> None:
    """Empty list; with `ticker`, start the ticker and record its pid."""
    assert (await _run_ok(session, model.driver.init())).strip() == "ok"
    if ticker:
        model.ticker_pid = int((await _run_ok(session, model.driver.start_ticker())).strip())
        assert model.ticker_pid > 1


async def play_round(session: Session, model: GuestModel) -> None:
    """Set a var, append to the list, write a note (guest) and a seeded 1 MiB file (write_file),
    truncate the previous big file, delete the note of two rounds ago."""
    n = model.rounds + 1
    rng = random.Random(f"{model.seed}/{n}")
    previous_big = f"big{n - 1}.bin"
    old_note = f"note{n - 2}.txt"
    ops = RoundOps(
        number=n,
        var=f"pv{n % 3}",
        token=rng.randbytes(8).hex(),
        note=f"note{n}.txt",
        truncate=(previous_big, rng.randrange(1, BIG_FILE_BYTES)) if previous_big in model.files else None,
        delete=old_note if old_note in model.files else None,
    )
    assert (await _run_ok(session, model.driver.round(ops))).strip() == "ok"
    big = _seeded(f"{model.seed}/{n}/big", BIG_FILE_BYTES)
    await session.write_file(f"state/big{n}.bin", big)

    model.rounds = n
    model.variables[ops.var] = ops.token
    model.items.append(n)
    model.files[ops.note] = ops.token.encode()
    if ops.truncate is not None:
        name, size = ops.truncate
        model.files[name] = model.files[name][:size]
    if ops.delete is not None:
        del model.files[ops.delete]
    model.files[f"big{n}.bin"] = big


async def probe(session: Session, model: GuestModel, *, wait_ticks: int) -> Probe:
    """One exec; with wait_ticks, it first waits (bounded, in the guest) for that many more ticks."""
    code = model.driver.probe(ticker_pid=model.ticker_pid, wait_ticks=wait_ticks)
    return parse_probe(await _run_ok(session, code))


async def assert_restored(session: Session, model: GuestModel) -> Probe:
    """The guest equals the model; a ticker runs on under its pid, its lines still 1..n; no lost sleeper."""
    got = await probe(session, model, wait_ticks=5 if model.ticker_pid else 0)
    assert got.state == model.expected()
    assert got.ticks_contiguous, got
    assert got.sleepers == 0, got
    if model.ticker_pid:
        assert got.ticker_alive, got
        assert got.ticks >= got.ticks_at_start + 5, got
    return got


async def lose_changes(session: Session, model: GuestModel) -> None:
    """Changes after the last save (the model does not take them): change a var, write lost.txt,
    delete a kept file, kill the ticker, start `sleep 7771`. The probe confirms they happened."""
    kept = sorted(model.files)[0]
    code = model.driver.lost(kill_pid=model.ticker_pid, delete=kept)
    assert (await _run_ok(session, code)).strip() == "ok"
    got = await probe(session, model, wait_ticks=0)
    assert got.state.variables["plost"] == "lost"
    assert "lost.txt" in got.state.files
    assert kept not in got.state.files
    assert not got.ticker_alive
    assert got.sleepers == 1


async def reuse_and_check(scheduler: Scheduler, ref: InstanceRef, model: GuestModel) -> None:
    """reuse() -> the state of `ref` (probe == model), then close (the probe counts as a change: warns)."""
    async with await scheduler.reuse(ref.instance_id) as vm:
        assert_origin(vm, "reuse")
        assert vm.instance == ref
        await assert_restored(vm, model)
        await close_unsaved(vm)
