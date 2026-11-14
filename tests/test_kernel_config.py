"""exec-sandbox.config disables the built-in drivers that wait for hardware microvm lacks.

Alpine's base config builds legacy PC drivers that probe absent hardware with timeouts
(floppy: 3 s; CMOS RTC under rtc=off: ~1.6 s), and MODULES=n turns its modules into
built-ins: each one delays every x86 cold boot before /init.
"""

import re
from pathlib import Path

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
