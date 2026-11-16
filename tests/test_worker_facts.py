"""WorkerFacts: the rule of each RouteKey, applied to a RouteError's `required`."""

import json

import pytest

from exec_sandbox import RouteKey, WorkerFacts

_FACTS = WorkerFacts(
    values={
        RouteKey.ARCH: "aarch64",
        RouteKey.ACCELERATOR: "hvf",
        RouteKey.CPU_MODEL: "host",
        RouteKey.COUNTER_HZ: "24000000",
        RouteKey.QEMU_VERSION: "11.10.0",
        RouteKey.MACHINE_TYPE: frozenset({"virt", "virt-11.10"}),
        RouteKey.BASE_IMAGE: frozenset({"a" * 64}),
        RouteKey.PACKAGES_IMAGE: frozenset(),
        RouteKey.MEMORY_MB: "4096",
        RouteKey.CPU: None,
    },
    key_ids=("b" * 64,),
)


@pytest.mark.parametrize(
    ("required", "satisfied"),
    [
        pytest.param({}, False, id="no-target-another-release"),
        pytest.param({RouteKey.ARCH: "aarch64", RouteKey.CPU_MODEL: "host"}, True, id="equal"),
        pytest.param({RouteKey.ARCH: "x86_64"}, False, id="equal-differs"),
        pytest.param(
            {RouteKey.CPU_ID: "midr_el1=0x00000000410fd4f1,revidr_el1=0x0000000000000000"},
            False,
            id="fact-this-worker-lacks",
        ),
        pytest.param({RouteKey.COUNTER_HZ: "24000000"}, True, id="same-counter-rate"),
        pytest.param({RouteKey.COUNTER_HZ: "1000000000"}, False, id="other-counter-rate"),
        pytest.param({RouteKey.BASE_IMAGE: "a" * 64}, True, id="contains"),
        pytest.param({RouteKey.BASE_IMAGE: "c" * 64}, False, id="image-not-here"),
        pytest.param({RouteKey.PACKAGES_IMAGE: "a" * 64}, False, id="image-of-another-role"),
        pytest.param({RouteKey.MEMORY_MB: "4096"}, True, id="at-least-equal"),
        pytest.param({RouteKey.MEMORY_MB: "4097"}, False, id="above-the-budget"),
        pytest.param({RouteKey.MEMORY_MB: "512"}, True, id="fewer-digits-below"),
        pytest.param({RouteKey.MEMORY_MB: "40960"}, False, id="more-digits-above"),
        pytest.param({RouteKey.CPU: "512.00"}, True, id="no-limit"),
        pytest.param({RouteKey.QEMU_VERSION: "11.9.3"}, True, id="older-qemu-saved-it"),
        pytest.param({RouteKey.QEMU_VERSION: "11.10.0"}, True, id="same-qemu"),
        pytest.param({RouteKey.QEMU_VERSION: "11.10.1"}, False, id="newer-qemu-saved-it"),
        pytest.param({RouteKey.MACHINE_TYPE: "virt-11.10"}, True, id="machine-type"),
    ],
)
def test_satisfies(required: dict[RouteKey, str], satisfied: bool) -> None:
    assert _FACTS.satisfies(required) is satisfied


def test_json_round_trip() -> None:
    """A broker in another language reads the JSON form: sets become sorted lists (the same
    facts always give the same text), None stays null."""
    text = _FACTS.model_dump_json()
    assert '"cpu":null' in text
    assert '"machine_type":["virt","virt-11.10"]' in text
    assert WorkerFacts.model_validate_json(text) == _FACTS
    many = WorkerFacts(values={RouteKey.BASE_IMAGE: frozenset(c * 64 for c in "fedcba")}, key_ids=())
    assert json.loads(many.model_dump_json())["values"]["base_image"] == sorted(c * 64 for c in "abcdef")


def test_required_decoded_from_json() -> None:
    """A RouteError's `required` that a broker decoded from JSON has plain string keys; a key
    this release does not know (a newer release added it) is not satisfied."""
    assert _FACTS.satisfies({"arch": "aarch64", "qemu_version": "11.9.3", "base_image": "a" * 64})
    assert not _FACTS.satisfies({"arch": "x86_64"})
    assert not _FACTS.satisfies({"arch": "aarch64", "added_later": "1"})


def test_any_counter_rate_takes_every_rate() -> None:
    """x86 KVM with TSC scaling: the worker's counter_hz "any" satisfies every saved rate."""
    scaling = WorkerFacts(values={RouteKey.COUNTER_HZ: "any"}, key_ids=())
    assert scaling.satisfies({RouteKey.COUNTER_HZ: "2969960000"}) is True
    assert scaling.satisfies({"counter_hz": "2700000000"}) is True
