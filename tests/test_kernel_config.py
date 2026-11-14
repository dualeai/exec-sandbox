"""exec-sandbox.config disables the built-in drivers that wait for hardware microvm lacks.

Alpine's base config builds legacy PC drivers that probe absent hardware with timeouts
(floppy: 3 s; CMOS RTC under rtc=off: ~1.6 s), and MODULES=n turns its modules into
built-ins: each one delays every x86 cold boot before /init.
"""

import re
from pathlib import Path
from typing import Final

import pytest

_KERNEL_DIR = Path(__file__).parent.parent / "images" / "kernel"

# Drivers whose probe waits for hardware the microvm machine does not have.
_SLOW_PROBE_DRIVERS = ("CONFIG_BLK_DEV_FD", "CONFIG_RTC_DRV_CMOS", "CONFIG_RTC_DRV_TEST")


@pytest.mark.parametrize("option", _SLOW_PROBE_DRIVERS)
def test_fragment_disables_slow_probe_driver(option: str) -> None:
    fragment = (_KERNEL_DIR / "exec-sandbox.config").read_text()
    base = (_KERNEL_DIR / "alpine-virt-x86_64.config").read_text()

    assert re.search(rf"^{option}=[ym]$", base, re.MULTILINE)  # the base really builds it
    assert f"\n# {option} is not set\n" in fragment


# Fragment directives whose symbol is in neither base config: symbols of the pinned kernel's
# Kconfig (checked against it) whose dependencies Alpine leaves off, so they are absent from its
# .config; kept as a guard if a base update turns them on.
_ABSENT_FROM_BASE: Final = frozenset(
    {
        "CONFIG_LEDS_CLASS",
        "CONFIG_NVME_RDMA",
        "CONFIG_NVME_APPLE",
        "CONFIG_NVME_TARGET_RDMA",
        "CONFIG_NVME_AUTH",
        "CONFIG_SMC",
        "CONFIG_PHYLINK",
        "CONFIG_HWPOISON_INJECT",
    }
)


def test_each_disabled_symbol_is_a_symbol_of_the_base() -> None:
    """A "# CONFIG_X is not set" line whose X no base config knows does nothing: a wrong name
    (CONFIG_NFNETLINK for NETFILTER_NETLINK, CONFIG_SCTP for IP_SCTP) left those built in
    (MODULES=n builds every =m in) and the build check passed. Each such line is either a
    symbol of a base config or one of the listed real symbols that Alpine leaves out."""
    fragment = (_KERNEL_DIR / "exec-sandbox.config").read_text()
    bases = [(_KERNEL_DIR / f"alpine-virt-{arch}.config").read_text() for arch in ("aarch64", "x86_64")]
    disabled = re.findall(r"^# (CONFIG_[A-Z0-9_]+) is not set$", fragment, re.MULTILINE)
    unknown = {
        symbol
        for symbol in disabled
        if not any(re.search(rf"^(# )?{symbol}[= ]", base, re.MULTILINE) for base in bases)
    }

    assert unknown == _ABSENT_FROM_BASE
