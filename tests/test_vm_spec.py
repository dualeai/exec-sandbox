"""VmSpec: the public, frozen object that holds what a VM is made of."""

import pytest
from pydantic import ValidationError

from exec_sandbox.models import Language, PortMapping
from exec_sandbox.vm_spec import VmSpec


class TestVmSpec:
    def test_defaults(self) -> None:
        """192 MB of RAM, 1 vCPU, a 1024 MB home disk, no package, no network."""
        spec = VmSpec(language=Language.PYTHON)
        assert (spec.memory_mb, spec.cpu_cores, spec.disk_mb) == (192, 1, 1024)
        assert (spec.packages, spec.allow_network, spec.allowed_domains, spec.expose_ports) == ((), False, None, ())

    def test_packages_are_canonical(self) -> None:
        """Package order never changes the VM: the spec sorts them."""
        assert VmSpec(language=Language.PYTHON, packages=("b==1", "a==2")).packages == ("a==2", "b==1")
        assert VmSpec(language=Language.PYTHON, packages=("b==1", "a==2")) == VmSpec(
            language=Language.PYTHON, packages=("a==2", "b==1")
        )

    @pytest.mark.parametrize(
        ("allow_network", "ports", "expected"),
        [
            (False, (), False),
            (True, (), True),
            (False, (PortMapping(internal=8080),), True),
        ],
    )
    def test_network_device(self, allow_network: bool, ports: tuple[PortMapping, ...], expected: bool) -> None:
        """Network access or an exposed port adds the virtio-net device."""
        spec = VmSpec(language=Language.PYTHON, allow_network=allow_network, expose_ports=ports)
        assert spec.has_network_device is expected

    def test_int_port_is_internal_port(self) -> None:
        spec = VmSpec(language=Language.PYTHON, expose_ports=[8080, PortMapping(internal=9000, external=3000)])
        assert spec.expose_ports == (PortMapping(internal=8080), PortMapping(internal=9000, external=3000))

    def test_a_host_port_given_twice_is_refused(self) -> None:
        """gvproxy keys its forwards by host port and protocol: a second forward on the same
        host port would replace the first with no error."""
        ports = [
            PortMapping(internal=80, external=3000),
            PortMapping(internal=81, external=3000),
            PortMapping(internal=82, external=3001, protocol="udp"),
            PortMapping(internal=83, external=3001, protocol="udp"),
        ]
        with pytest.raises(ValidationError, match=r"Host port given twice: 3000/tcp, 3001/udp"):
            VmSpec(language=Language.PYTHON, expose_ports=ports)

    def test_one_host_port_per_protocol_and_shared_guest_ports_are_accepted(self) -> None:
        """TCP and UDP on the same host port are two forwards; a guest port on two host ports
        too; dynamic host ports (None) are picked apart by the OS."""
        ports = (
            PortMapping(internal=80, external=3000),
            PortMapping(internal=80, external=3000, protocol="udp"),
            PortMapping(internal=80, external=3001),
            PortMapping(internal=81),
            PortMapping(internal=81),
        )
        assert VmSpec(language=Language.PYTHON, expose_ports=ports).expose_ports == ports

    def test_frozen_and_hashable(self) -> None:
        """A catalogue of specs can be dict keys; a spec never changes after creation."""
        spec = VmSpec(language=Language.PYTHON, expose_ports=[8080])
        assert hash(spec) == hash(VmSpec(language="python", expose_ports=[8080]))
        with pytest.raises(ValidationError, match="Instance is frozen"):
            spec.memory_mb = 1024  # type: ignore[misc]

    def test_domains_are_kept_as_the_network_filter_compares_them(self) -> None:
        """Lowercase and with no final dot: gvproxy matches lowercase names with no final dot,
        so "GitHub.com." must allow github.com. Each name once and sorted: the
        same set of names, in any order or form, is the same spec."""
        spec = VmSpec(
            language=Language.PYTHON, allow_network=True, allowed_domains=("PyPI.org", "GitHub.com.", "github.com")
        )
        assert spec.allowed_domains == ("github.com", "pypi.org")
        assert spec == VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=("pypi.org", "github.com"))

    def test_json_round_trip(self) -> None:
        """A saved instance stores the spec as JSON and rebuilds the same VM from it."""
        spec = VmSpec(
            language=Language.RAW, memory_mb=512, packages=("a",), expose_ports=[80], allowed_domains=("x.y",)
        )
        assert VmSpec.model_validate_json(spec.model_dump_json()) == spec

    @pytest.mark.parametrize(
        ("fields", "error_type"),
        [
            pytest.param({"memory_mb": 127}, "greater_than_equal", id="memory-below-min"),
            pytest.param({"disk_mb": 127}, "greater_than_equal", id="disk-below-min"),
            pytest.param({"cpu_cores": 0}, "greater_than_equal", id="no-cpu"),
            pytest.param({"expose_ports": list(range(1, 12))}, "too_long", id="too-many-ports"),
            pytest.param({"expose_ports": None}, "tuple_type", id="ports-not-a-sequence"),
            pytest.param({"language": "cobol"}, "enum", id="unknown-language"),
            pytest.param({"memroy_mb": 256}, "extra_forbidden", id="typo-field"),
            pytest.param({"allowed_domains": ["pypi.org", "bad..domain"]}, "value_error", id="domain-empty-label"),
            pytest.param({"allowed_domains": ["example.123"]}, "value_error", id="domain-numeric-tld"),
            pytest.param({"allowed_domains": ["(a|b)*.com"]}, "value_error", id="domain-regex"),
        ],
    )
    def test_invalid(self, fields: dict[str, object], error_type: str) -> None:
        """One error, on the field given, for the rule that field breaks."""
        with pytest.raises(ValidationError) as exc:
            VmSpec.model_validate({"language": "python", **fields})
        assert [(error["loc"], error["type"]) for error in exc.value.errors()] == [(tuple(fields), error_type)]

    @pytest.mark.parametrize("memory_mb", [128, 65536])
    def test_memory_bounds_accept(self, memory_mb: int) -> None:
        """128 MB is the floor; there is no ceiling (the host admission decides)."""
        assert VmSpec(language=Language.PYTHON, memory_mb=memory_mb).memory_mb == memory_mb

    def test_smallest_disk_and_most_ports_accept(self) -> None:
        """128 MB is the smallest home disk; 10 is the most exposed ports."""
        spec = VmSpec(language=Language.PYTHON, disk_mb=128, expose_ports=list(range(1, 11)))
        assert (spec.disk_mb, len(spec.expose_ports)) == (128, 10)
