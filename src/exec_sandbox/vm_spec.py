"""VmSpec: what a VM is made of.

One frozen object holds everything that shapes the guest: the L1 memory
snapshot cache key holds the fields that change the guest state (the domain
list and the port numbers stay on the host side, so they are not in it), and a
saved instance stores it, so a load makes the same VM again. Values of one launch only (vm id, work paths, host ports
picked at launch, timeouts) are not part of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exec_sandbox import constants
from exec_sandbox.dns_filter import normalize_domain
from exec_sandbox.models import Language, PortMapping

if TYPE_CHECKING:
    from collections.abc import Sequence


class VmSpec(BaseModel):
    """What a VM is made of. Pass one to `Scheduler.new()`.

    Frozen and hashable: keep a catalog of specs, one per workload, and
    set only the fields that a workload needs:

        SMALL = VmSpec(language=Language.PYTHON)
        DATA = VmSpec(language=Language.PYTHON, memory_mb=1024, disk_mb=4096, packages=("pandas==3.0.1",))
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: Language = Field(description="Guest runtime: python, javascript or raw (shell)")
    memory_mb: int = Field(
        default=constants.DEFAULT_MEMORY_MB,
        ge=constants.MIN_MEMORY_MB,
        description=f"Guest RAM in MB ({constants.MIN_MEMORY_MB} or more)",
    )
    cpu_cores: int = Field(default=constants.DEFAULT_VM_CPU_CORES, ge=1, description="Guest vCPUs (1 or more)")
    disk_mb: int = Field(
        default=constants.DEFAULT_DISK_MB,
        ge=constants.MIN_DISK_MB,
        description=f"Size of the home disk (/home/user) in MB ({constants.MIN_DISK_MB} or more); "
        "sparse on the host, so only written blocks take space",
    )
    packages: tuple[str, ...] = Field(
        default=(),
        description="Packages to install, each with a version (for example 'pandas==3.0.1', 'lodash@4.17.21'); "
        "sorted, since order never changes the VM",
    )
    allow_network: bool = Field(default=False, description="Outbound network access")
    allowed_domains: tuple[str, ...] | None = Field(
        default=None,
        description="With allow_network: the only domains the guest may reach, and their subdomains. "
        "None: the language's package registries only (PyPI for python, npm for javascript; no limit for raw). "
        "Empty: no limit. "
        "Each entry must be a domain name (RFC 1035, alphabetic TLD), checked when the spec is built; "
        "the spec keeps each name once, lowercase, with no final dot, in sorted order.",
    )
    expose_ports: tuple[PortMapping, ...] = Field(
        default=(),
        max_length=constants.MAX_EXPOSED_PORTS,
        description="Guest ports to expose on the host; an int is a PortMapping with that internal port",
    )

    @field_validator("packages")
    @classmethod
    def _sort_packages(cls, packages: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(packages))

    @field_validator("allowed_domains")
    @classmethod
    def _check_domains(cls, domains: tuple[str, ...] | None) -> tuple[str, ...] | None:
        # Checked here, not when gvproxy starts: a bad entry fails when the spec is
        # built (pydantic ValidationError), before any VM or reservation.
        # As the network filter compares names: no final dot, lowercase; each
        # name once and sorted, as packages are, so the same set is the same spec
        return None if domains is None else tuple(sorted({normalize_domain(domain) for domain in domains}))

    @field_validator("expose_ports", mode="before")
    @classmethod
    def _ports_from_ints(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            items = cast("Sequence[object]", value)
            return tuple(PortMapping(internal=p) if isinstance(p, int) else p for p in items)
        return value

    @field_validator("expose_ports")
    @classmethod
    def _check_host_ports(cls, ports: tuple[PortMapping, ...]) -> tuple[PortMapping, ...]:
        # gvproxy takes the forwards as one JSON object keyed by host port and protocol: a
        # host port given twice would keep one forward and drop the other with no error
        given = [(p.external, p.protocol) for p in ports if p.external is not None]
        twice = sorted({key for key in given if given.count(key) > 1})
        if twice:
            raise ValueError(f"Host port given twice: {', '.join(f'{port}/{protocol}' for port, protocol in twice)}")
        return ports

    @property
    def has_network_device(self) -> bool:
        """The VM has a virtio-net device: network access or exposed ports."""
        return self.allow_network or bool(self.expose_ports)
