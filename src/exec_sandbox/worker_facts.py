"""This worker's facts, in the keys (RouteKey) and value forms of RouteError.required.

A broker reads them once per worker (Scheduler.describe_worker) and sends a call that
raised RouteError to a worker whose facts satisfy the error's `required`
(WorkerFacts.satisfies). The library places nothing: the broker picks the worker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from exec_sandbox.exceptions import ROUTE_ANY, RouteKey
from exec_sandbox.platform_utils import detect_host_arch
from exec_sandbox.qemu_cmd import counter_rate, guest_cpu_id, guest_cpu_model
from exec_sandbox.system_probes import detect_accel_type, require_qemu_version

if TYPE_CHECKING:
    from collections.abc import Mapping


async def machine_facts(*, force_emulation: bool) -> dict[RouteKey, str | None]:
    """This host's EQUAL facts, which a save records as its host's (HostFacts): arch,
    accelerator, guest CPU model and CPU identity (None where the guest CPU is a fixed
    model). Every probe is cached for the process, so a call after the first costs no
    subprocess. Raises as detect_accel_type and guest_cpu_id."""
    arch = detect_host_arch()
    accel = await detect_accel_type(force_emulation=force_emulation)
    return {
        RouteKey.ARCH: arch.name.lower(),
        RouteKey.ACCELERATOR: str(accel.value),
        RouteKey.CPU_MODEL: await guest_cpu_model(arch, accel),
        RouteKey.CPU_ID: await guest_cpu_id(arch, accel),
    }


async def counter_fact(*, force_emulation: bool) -> str | None:
    """This worker's counter_hz: "any" (ROUTE_ANY) with x86 KVM TSC scaling, else the rate a
    guest that boots here gets; None: no exact rate here (qemu_cmd.counter_rate). Not a
    machine fact: a save keeps the rate of the guest's boot (GuestFacts), not its own. Raises
    as detect_accel_type and counter_rate."""
    rate = await counter_rate(detect_host_arch(), await detect_accel_type(force_emulation=force_emulation))
    if rate.any_rate:
        return ROUTE_ANY
    return None if rate.hz is None else str(rate.hz)


async def qemu_version_fact() -> str:
    """This worker's QEMU version as MAJOR.MINOR.PATCH (require_qemu_version: cached; raises
    ConfigError HOST_DEPENDENCY without one)."""
    return ".".join(map(str, await require_qemu_version()))


class WorkerFacts(BaseModel):
    """What this worker can serve (Scheduler.describe_worker): one value per RouteKey, in the
    value forms of RouteError.required. A key whose rule is CONTAINS maps to the set of values
    this worker holds; the other keys map to its value (None: no limit, for memory_mb and
    cpu). A fact that does not apply (cpu_id on x86) or that this worker lacks (an exact
    counter rate) is absent."""

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
        newer release added it), is not satisfied. An empty `required` names no worker
        (another release wrote the instance): False, so a broker never sends the call back to
        a worker like this one; describe_instance(id).writer_release names the release, or the
        error's context["writer_release"] when the data names it and describe raises too."""
        if not required:
            return False
        for name, need in required.items():
            try:
                key = RouteKey(name)
            except ValueError:
                return False
            if key not in self.values or not key.holds(self.values[key], need):
                return False
        return True
