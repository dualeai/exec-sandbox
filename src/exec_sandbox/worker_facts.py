"""This worker's facts, in the keys (RouteKey) and value forms of RouteError.required.

A broker reads them once per worker (Scheduler.describe_worker) and sends a call that
raised RouteError to a worker whose facts satisfy the error's `required`
(WorkerFacts.satisfies). The library places nothing: the broker picks the worker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from exec_sandbox.exceptions import RouteKey
from exec_sandbox.platform_utils import HostArch, detect_host_arch, detect_host_x86_vendor
from exec_sandbox.qemu_cmd import guest_cpu_model
from exec_sandbox.system_probes import detect_accel_type, require_qemu_version

if TYPE_CHECKING:
    from collections.abc import Mapping


async def machine_facts(*, force_emulation: bool) -> dict[RouteKey, str | None]:
    """This host's EQUAL facts: arch, accelerator, CPU vendor (x86 only, else None) and guest
    CPU model. Every probe is cached for the process, so a call after the first costs no
    subprocess."""
    arch = detect_host_arch()
    accel = await detect_accel_type(force_emulation=force_emulation)
    return {
        RouteKey.ARCH: arch.name.lower(),
        RouteKey.ACCELERATOR: str(accel.value),
        RouteKey.CPU_VENDOR: detect_host_x86_vendor() if arch == HostArch.X86_64 else None,
        RouteKey.CPU_MODEL: await guest_cpu_model(arch, accel),
    }


async def qemu_version_fact() -> str:
    """This worker's QEMU version as MAJOR.MINOR.PATCH (require_qemu_version: cached; raises
    ConfigError HOST_DEPENDENCY without one)."""
    return ".".join(map(str, await require_qemu_version()))


class WorkerFacts(BaseModel):
    """What this worker can serve (Scheduler.describe_worker): one value per RouteKey, in the
    value forms of RouteError.required. A key whose rule is CONTAINS maps to the set of values
    this worker holds; the other keys map to its value (None: no limit, for memory_mb and
    cpu). A fact that does not apply (cpu_vendor on ARM) is absent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: dict[RouteKey, str | frozenset[str] | None] = Field(
        description="This worker's value of each key: a set for machine_type, base_image and packages_image "
        "(the machine types of its QEMU, the digests of the guest images on its disk)"
    )
    key_ids: tuple[str, ...] = Field(
        description="Ids of this worker's instance keys, newest first (empty without an instance store): "
        'the key an INSTANCE_KEY_MISSING error names is in its context["key_id"]'
    )

    @field_serializer("values", when_used="json")
    def _sorted_sets(self, values: dict[RouteKey, str | frozenset[str] | None]) -> dict[str, str | list[str] | None]:
        """JSON lists in sorted order: the same facts always give the same text."""
        return {str(key): sorted(value) if isinstance(value, frozenset) else value for key, value in values.items()}

    def satisfies(self, required: Mapping[RouteKey, str] | Mapping[str, str]) -> bool:
        """True when this worker can serve a call whose RouteError gave `required`: each key
        holds by its rule (RouteKey.rule). Keys may be plain strings (a `required` decoded
        from JSON). A key these facts do not list, or that this release does not know (a
        newer release added it), is not satisfied."""
        for name, need in required.items():
            try:
                key = RouteKey(name)
            except ValueError:
                return False
            if key not in self.values or not key.holds(self.values[key], need):
                return False
        return True
