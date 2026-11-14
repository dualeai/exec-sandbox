"""Patch targets in the tests: no test replaces a public function or class of asyncio or time
(the subprocess spawns aside), or a loop method, for the whole process."""

from __future__ import annotations

import ast
import asyncio
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SPAWNS: Final = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
"""Faked spawns are left out: each fakes the spawns of the code under test, where nothing
else spawns."""


def _public(module: ModuleType) -> frozenset[str]:
    return frozenset(name for name in dir(module) if not name.startswith("_") and callable(getattr(module, name)))


_SHARED_FUNCTIONS: Final[dict[str, frozenset[str]]] = {"asyncio": _public(asyncio) - _SPAWNS, "time": _public(time)}
"""Public functions and classes of the asyncio and time modules, which the event loop,
pytest-asyncio and every module share: patching one (patch("asyncio.sleep"),
setattr(asyncio, "sleep", ...), patch.object(mod.asyncio, "sleep"), the string
exec_sandbox.qemu_vm.asyncio.sleep, which names the same module object, or
`asyncio.sleep = f`) replaces it in the whole process, and a frozen time.monotonic stops the
loop's clock. Use tests/conftest.py asyncio_with / time_with on the module's own name instead."""

_LOOP_METHODS: Final = frozenset({"time", "call_later", "call_at", "run_in_executor"})
"""Event loop methods: patching one on a loop or a loop class changes that loop's clock or
timers for every task (patch.object(asyncio.get_running_loop(), "time"), BaseEventLoop.time)."""

_LOOP_CALL: Final = re.compile(r"(?:get_running_loop|get_event_loop|new_event_loop)\(\)")

_LOOP_TARGET: Final = re.compile(rf"{_LOOP_CALL.pattern}|EventLoop\b|loop\b")
"""A loop or a loop class by its usual names (asyncio.get_running_loop(), BaseEventLoop,
loop, event_loop, self._loop); a name assigned from a loop call counts too (_names)."""

_ASYNCIO_SUBMODULES: Final = r"(?:events|base_events|tasks|timeouts|runners)"
"""Where asyncio's shared functions are defined: asyncio.sleep calls events.get_running_loop(),
so a patch of asyncio.events.get_running_loop changes every caller."""

_STRING_TARGET: Final = re.compile(rf"(?:^|\.)(asyncio|time)(?:\.{_ASYNCIO_SUBMODULES})?\.(\w+)$")

_ASYNCIO_INTERNAL: Final = re.compile(rf"(?:^|\.)asyncio\.{_ASYNCIO_SUBMODULES}(?:\.\w+)?$")
"""A submodule of asyncio or any of its names (asyncio.base_events.time: the clock of every
loop; asyncio.tasks.events: what asyncio.sleep calls): shared by every loop and task."""

_ASYNCIO_SUBMODULE: Final = "asyncio submodule"

_LOOP_STRING: Final = re.compile(r"EventLoop\.(\w+)$")

_PATCHERS: Final = frozenset({"patch", "setattr", "delattr"})

_UNJUDGED_KNOWN: Final = frozenset(
    {
        # exec_sandbox.vm_manager.cleanup_process / exec_sandbox.cgroup.Cgroup.remove (parametrize)
        ("test_vm_manager.py", "patch(failing, new_callable=AsyncMock, return_value=False)"),
    }
)
"""Patch calls whose target is a variable, so the scan cannot judge them: each was read and
is not shared. A new one fails the scan until it is read and listed here."""


class _UnjudgedError(Exception):
    """A patch call whose target or attribute is not a literal string."""


def _names(tree: ast.Module) -> dict[str, str]:
    """Local name -> 'asyncio' or 'time' for those modules (`import asyncio as aio`),
    _ASYNCIO_SUBMODULE for one of its submodules, 'loop' for a name assigned from a loop call
    (`running = asyncio.get_running_loop()`), 'patch' for an alias of unittest.mock.patch."""
    names = {module: module for module in _SHARED_FUNCTIONS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _SHARED_FUNCTIONS:
                    names[alias.asname or alias.name] = alias.name
                elif alias.asname and _ASYNCIO_INTERNAL.fullmatch(alias.name):  # import asyncio.events as ev
                    names[alias.asname] = _ASYNCIO_SUBMODULE
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":  # from asyncio import events
            for alias in node.names:
                if re.fullmatch(_ASYNCIO_SUBMODULES, alias.name):
                    names[alias.asname or alias.name] = _ASYNCIO_SUBMODULE
        elif (
            isinstance(node, ast.ImportFrom) and node.module == "unittest.mock"
        ):  # from unittest.mock import patch as p
            for alias in node.names:
                if alias.name == "patch" and alias.asname:
                    names[alias.asname] = "patch"
        elif (
            isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr)
            and node.value is not None
            and _LOOP_CALL.search(ast.unparse(node.value))
        ):  # loop = ..., loop: T = ..., (loop := ...)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update((ast.unparse(target), "loop") for target in targets)
    return names


def _string(node: ast.expr | None) -> str | None:
    """A str constant, or a string built from literal parts with each unknown part read as
    "x" (an f-string, `+`, `%`, `.format`). A name or another expression is None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "x"
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (_string(node.left) or "x") + (_string(node.right) or "x")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and (left := _string(node.left)) is not None:
        values: tuple[str, ...] | dict[str, str] = (
            {
                _string(key) or "x": _string(value) or "x"
                for key, value in zip(node.right.keys, node.right.values, strict=True)
            }
            if isinstance(node.right, ast.Dict)
            else tuple(
                _string(part) or "x"
                for part in (node.right.elts if isinstance(node.right, ast.Tuple) else [node.right])
            )
        )
        try:
            return left % values
        except (KeyError, TypeError, ValueError):
            return re.sub(r"%(?:\(\w+\))?[sdr]", "x", left)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and (template := _string(node.func.value)) is not None
    ):
        args = [_string(arg) or "x" for arg in node.args]
        kwargs = {keyword.arg: _string(keyword.value) or "x" for keyword in node.keywords if keyword.arg}
        try:
            return template.format(*args, **kwargs)
        except (IndexError, KeyError, ValueError):
            return re.sub(r"\{[^}]*\}", "x", template)
    return None


def _module_dict(node: ast.expr | None, names: dict[str, str]) -> str | None:
    """'asyncio' or 'time' when `node` is that module's namespace: sys.modules (any module),
    vars(time), asyncio.__dict__. sys.modules gives '*'."""
    if node is None:
        return None
    if ast.unparse(node) == "sys.modules" or _string(node) == "sys.modules":
        return "*"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "vars" and node.args:
        return _module_of(node.args[0], names)
    if isinstance(node, ast.Attribute) and node.attr == "__dict__":
        return _module_of(node.value, names)
    return None


def _shared_key(namespace: str, key: str | None) -> bool:
    """A key of a module namespace (_module_dict) that names a shared module or function."""
    return key in _SHARED_FUNCTIONS if namespace == "*" else key in _SHARED_FUNCTIONS[namespace]


def _shared_string(target: str) -> bool:
    if (loop := _LOOP_STRING.search(target)) is not None:  # patch("asyncio.BaseEventLoop.time")
        return loop.group(1) in _LOOP_METHODS
    if _ASYNCIO_INTERNAL.search(target):  # patch("asyncio.base_events.time")
        return True
    match = _STRING_TARGET.search(target)
    return match is not None and match.group(2) in _SHARED_FUNCTIONS[match.group(1)]


def _module_of(node: ast.expr, names: dict[str, str]) -> str | None:
    """'asyncio' or 'time' when `node` names that module (asyncio, aio, mod.asyncio,
    sys.modules['time'], asyncio.events)."""
    if isinstance(node, ast.Subscript) and (key := _string(node.slice)) in _SHARED_FUNCTIONS:
        return key
    parts = ast.unparse(node).split(".")
    if len(parts) >= 2 and re.fullmatch(_ASYNCIO_SUBMODULES, parts[-1]) and names.get(parts[-2]) == "asyncio":
        return "asyncio"
    module = names.get(parts[-1])
    return module if module in _SHARED_FUNCTIONS else None


def _shared_attribute(target: ast.expr, attribute: str | None, imports: dict[str, str]) -> bool:
    """Whether `attribute` of `target` is shared. A variable attribute (None) of an object
    that may be shared (asyncio, time, a submodule, a loop) cannot be judged; of any other
    object it is not shared."""
    name = ast.unparse(target)
    if attribute is None:
        shared = (
            _module_of(target, imports) is not None
            or imports.get(name) in (_ASYNCIO_SUBMODULE, "loop")
            or _ASYNCIO_INTERNAL.search(name) is not None
            or _LOOP_TARGET.search(name) is not None
        )
        if shared:
            raise _UnjudgedError
        return False
    if imports.get(name) == _ASYNCIO_SUBMODULE or _ASYNCIO_INTERNAL.search(name):  # patch.object(asyncio.events, "x")
        return True
    if (module := _module_of(target, imports)) is not None:
        return attribute in _SHARED_FUNCTIONS[module]
    return attribute in _LOOP_METHODS and (_LOOP_TARGET.search(name) is not None or imports.get(name) == "loop")


def _keyword(call: ast.Call, *names: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg in names), None)


def _keys(values: ast.expr | None, call: ast.Call, *, skip: frozenset[str]) -> list[str | None]:
    """The keys a dict write sets: a dict display, pairs, one key, and keyword arguments."""
    keys: list[ast.expr | None] = []
    if isinstance(values, ast.Dict):
        keys.extend(values.keys)
    elif isinstance(values, ast.List | ast.Tuple):  # [("time", t)]
        keys.extend(
            pair.elts[0] if isinstance(pair, ast.Tuple | ast.List) and pair.elts else pair for pair in values.elts
        )
    elif values is not None:
        keys.append(values)
    for keyword in call.keywords:
        if keyword.arg is None and isinstance(keyword.value, ast.Dict):  # **{"time": t}
            keys.extend(keyword.value.keys)
        elif keyword.arg is not None and keyword.arg not in skip:
            keys.append(ast.Constant(keyword.arg))
    return [_string(key) for key in keys]


def _assign_targets(node: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    flat: list[ast.expr] = []
    for target in targets:
        flat.extend(target.elts if isinstance(target, ast.Tuple | ast.List) else [target])
    return flat


def _assignment_patch(node: ast.Assign | ast.AnnAssign, imports: dict[str, str]) -> bool:
    """asyncio.sleep = f, sys.modules["time"] = t, vars(time)["monotonic"] = f."""
    for target in _assign_targets(node):
        if isinstance(target, ast.Attribute) and _shared_attribute(target.value, target.attr, imports):
            return True
        if (
            isinstance(target, ast.Subscript)
            and (namespace := _module_dict(target.value, imports)) is not None
            and _shared_key(namespace, _string(target.slice))
        ):
            return True
    return False


def _global_patch(node: ast.AST, imports: dict[str, str]) -> bool:
    """Whether `node` patches a shared asyncio or time function, or a loop method. Raises
    _UnjudgedError for a patch call whose target is a variable."""
    if isinstance(node, ast.Assign | ast.AnnAssign):
        return _assignment_patch(node, imports)
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    name = "patch" if imports.get(name) == "patch" else name
    args = node.args
    if name == "update" and isinstance(func, ast.Attribute) and (namespace := _module_dict(func.value, imports)):
        return any(_shared_key(namespace, key) for key in _keys(args[0] if args else None, node, skip=frozenset()))
    if name in ("dict", "setitem"):  # patch.dict(sys.modules, {...}), monkeypatch.setitem(sys.modules, "time", t)
        first = args[0] if args else _keyword(node, "in_dict", "dic")
        if (namespace := _module_dict(first, imports)) is None:
            return False
        values = args[1] if len(args) >= 2 else _keyword(node, "values", "name")
        skip = frozenset({"in_dict", "dic", "values", "name", "value", "clear"})
        return any(_shared_key(namespace, key) for key in _keys(values, node, skip=skip))
    first = args[0] if args else _keyword(node, "target")
    target = _string(first)
    if name == "multiple" and first is not None:  # patch.multiple("x.asyncio", sleep=f), patch.multiple(time, ...)
        keys = _keys(None, node, skip=frozenset({"target", "spec", "create", "spec_set", "autospec", "new_callable"}))
        return any(
            key is not None
            and (_shared_string(f"{target}.{key}") if target is not None else _shared_attribute(first, key, imports))
            for key in keys
        )
    if (
        name in ("object", "setattr", "delattr") and first is not None and target is None
    ):  # patch.object(asyncio, "sleep"); a string first is the string form below
        attribute = args[1] if len(args) >= 2 else _keyword(node, "attribute", "name")
        return _shared_attribute(first, _string(attribute), imports)
    if name in _PATCHERS and first is not None:  # patch("x.asyncio.sleep"), monkeypatch.setattr("x.time.monotonic", f)
        if target is None:
            raise _UnjudgedError
        return _shared_string(target)
    return False


def _scan(source: str) -> tuple[list[int], list[str]]:
    """The lines that patch a shared function, and the patch calls the scan cannot judge."""
    tree = ast.parse(source)
    imports = _names(tree)
    found: list[int] = []
    unjudged: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call | ast.Assign | ast.AnnAssign):
            try:
                if _global_patch(node, imports):
                    found.append(node.lineno)
            except _UnjudgedError:
                unjudged.append(ast.unparse(node))
    return sorted(found), unjudged


def _found(source: str) -> list[int]:
    return _scan(source)[0]


def test_no_test_patches_asyncio_or_time_for_the_whole_process() -> None:
    """Every .py file of tests/ (conftest, helpers, benchmarks too), this one aside. A patch
    call whose target is a variable fails too, unless it was read and listed."""
    here = Path(__file__)
    found: list[str] = []
    unjudged: set[tuple[str, str]] = set()
    for path in sorted(here.parent.rglob("*.py")):
        if path == here:
            continue
        lines, calls = _scan(path.read_text())
        found.extend(f"{path.relative_to(here.parent)}:{line}" for line in lines)
        unjudged.update((path.name, call) for call in calls)
    assert found == []
    assert unjudged == _UNJUDGED_KNOWN


@pytest.mark.parametrize(
    ("source", "caught"),
    [
        ('patch("asyncio.sleep")', True),
        ('patch(target="asyncio.sleep")', True),
        ('patch("exec_sandbox.qemu_vm.asyncio.sleep", new_callable=AsyncMock)', True),
        ('patch(f"exec_sandbox.{module}.time.monotonic")', True),
        ('monkeypatch.setattr(asyncio, "sleep", f)', True),
        ('monkeypatch.setattr("time.monotonic", f)', True),
        ('monkeypatch.delattr(asyncio, "to_thread")', True),
        ('patch.object(mod.asyncio, "sleep")', True),
        ('patch.object(asyncio, attribute="sleep")', True),
        ('patch.multiple("asyncio", sleep=f)', True),
        ("patch.multiple(time, monotonic=f)", True),
        ('import asyncio as aio\nmonkeypatch.setattr(aio, "sleep", f)', True),
        ('patch.object(sys.modules["time"], "monotonic")', True),
        ('patch.object(asyncio.get_running_loop(), "time")', True),
        ('patch.object(asyncio.BaseEventLoop, "time")', True),
        ('patch("asyncio.BaseEventLoop.time")', True),
        ('patch("asyncio.base_events.BaseEventLoop.call_later")', True),
        ('patch("asyncio.events.get_running_loop")', True),
        ('patch.object(asyncio.events, "get_running_loop")', True),
        ('patch("asyncio.tasks.sleep")', True),
        ('patch.object(event_loop, "time")', True),
        ('running = asyncio.get_running_loop()\npatch.object(running, "time")', True),
        ('patch.dict(sys.modules, {"time": fake})', True),
        ('monkeypatch.setattr("asyncio.base_events.time", time_with(monotonic=f))', True),
        ('patch.object(asyncio.base_events, "time", f)', True),
        ('patch("asyncio.tasks.events", fake)', True),
        ('from asyncio import events\npatch.object(events, "get_running_loop")', True),
        ('import asyncio.events as ev\npatch.object(ev, "get_running_loop")', True),
        ('patch.multiple("exec_sandbox.qemu_vm.asyncio", sleep=f)', True),
        ('running: AbstractEventLoop = asyncio.get_running_loop()\npatch.object(running, "time")', True),
        ('if (running := asyncio.get_running_loop()):\n    patch.object(running, "time")', True),
        ('monkeypatch.setitem(sys.modules, "time", f)', True),
        ('sys.modules["time"] = f', True),
        ('patch("asyncio." + "sleep")', True),
        ('patch("exec_sandbox.%s.asyncio.sleep" % module)', True),
        ('patch("exec_sandbox.{}.time.monotonic".format(module))', True),
        ('patch.multiple("asyncio.events", get_running_loop=f)', True),
        ('patch.multiple("asyncio.base_events", time=t)', True),
        ("patch.multiple(asyncio.base_events, time=t)", True),
        ("patch.multiple(asyncio.tasks, events=e)", True),
        ('patch.dict("sys.modules", {"time": fake})', True),
        ("patch.dict(sys.modules, time=f)", True),
        ('sys.modules.update({"time": f})', True),
        ('monkeypatch.setitem(sys.modules, name="time", value=f)', True),
        ('patch("asyncio.%s" % "sleep")', True),
        ('patch("asyncio.{}".format("sleep"))', True),
        ('patch("asyncio.gather")', True),
        ('patch("exec_sandbox.qemu_vm.asyncio.open_unix_connection")', True),
        ('patch("time.perf_counter_ns")', True),
        ('monkeypatch.setattr(asyncio, name="sleep", value=f)', True),
        ('monkeypatch.setattr(target=asyncio, name="sleep", value=f)', True),
        ('patch.multiple("asyncio", **{"sleep": f})', True),
        ('patch("asyncio.{n}".format(n="sleep"))', True),
        ('patch("{1}.{0}".format("sleep", "asyncio"))', True),
        ('patch("asyncio.%(n)s" % {"n": "sleep"})', True),
        ('from unittest.mock import patch as p\np("asyncio.sleep")', True),
        ("asyncio.sleep: Any = f", True),
        ("asyncio.sleep, time.monotonic = f, g", True),
        ('patch.dict(in_dict=sys.modules, values={"time": f})', True),
        ('monkeypatch.setitem(dic=sys.modules, name="time", value=f)', True),
        ('patch.dict(sys.modules, [("time", f)])', True),
        ('sys.modules.update([("time", f)])', True),
        ('patch.dict(asyncio.__dict__, {"sleep": f})', True),
        ('vars(time)["monotonic"] = f', True),
        ("asyncio.sleep = f", True),
        ('monkeypatch.setattr("exec_sandbox.qemu_vm.asyncio", asyncio_with(sleep=s))', False),
        ('patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec")', False),
        ('monkeypatch.setattr(os, "getpid", f)', False),
        ('patch.object(manager, "time")', False),
        ('patch("exec_sandbox.scheduler.EventLoopPolicy")', False),
        ('patch.dict(sys.modules, {"boto3": fake})', False),
        ('patch.multiple("exec_sandbox.qemu_vm", sleep=f)', False),
        ('monkeypatch.setitem(sys.modules, "boto3", f)', False),
        ('patch("exec_sandbox." + module + ".helper")', False),
        ('patch.multiple("exec_sandbox.qemu_vm", helper=f)', False),
        ('sys.modules.update({"boto3": f})', False),
        ('patch("asyncio.%s" % "subprocess")', False),
        ('patch("exec_sandbox.vm_manager.asyncio.create_subprocess_shell")', False),
        ('patch.dict(os.environ, {"time": "1"})', False),
        ('vars(module)["helper"] = f', False),
    ],
)
def test_the_scan_judges_each_form(source: str, caught: bool) -> None:
    """Each form that replaces a shared function or loop method is caught; a patch of the
    module's own name, of another module or of a spawn is not."""
    assert bool(_found(source)) is caught


@pytest.mark.parametrize(
    "source",
    ["patch(TARGET)", "monkeypatch.setattr(asyncio, name, f)", "patch.object(loop, attribute)"],
)
def test_a_target_in_a_variable_is_not_judged(source: str) -> None:
    """The scan cannot judge such a call: it fails the whole-tests scan until it is listed."""
    assert _scan(source) == ([], [source])


@pytest.mark.parametrize("source", ["monkeypatch.setattr(ctrl, attr, None)", "delattr(channel._writer, name)"])
def test_a_variable_attribute_of_a_plain_object_is_not_shared(source: str) -> None:
    """The object is no shared module or loop: whatever the attribute, nothing shared changes."""
    assert _scan(source) == ([], [])
