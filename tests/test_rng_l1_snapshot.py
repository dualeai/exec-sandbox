"""L1 snapshot restore must produce unique RNG output per VM.

QEMU migration replays exact CPU+RAM+device state, including the kernel CRNG.
The host's first ping after a restore carries 32 random bytes: the guest agent adds
them to the kernel input pool, then reseeds the CRNG (guest-agent/src/entropy.rs).
Without them, on HVF (a reseed there gets no fresh secret bytes from the guest
itself), VMs restored from the same vmstate would produce the same or predictable
os.urandom() output — a critical security vulnerability.

These tests force L1 restores of one template and assert that the random output of
the VMs differs: os.urandom (the kernel CRNG), JavaScript Math.random and
crypto.getRandomValues, and bash $RANDOM. Different output is not unpredictable output:
that the guest takes the host's bytes is tested on the ping itself
(test_rng_quality.TestHostEntropyPing, and the entropy tests of the guest agent).
"""

import asyncio
import json

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, run_timed, skip_unless_hwaccel, wait_l1_saves

# Code that reads urandom IMMEDIATELY — before any entropy can accumulate
# from CPU jitter / interrupts / etc.
IMMEDIATE_URANDOM = """\
import os, hashlib
# Read immediately on restore — minimal time for new entropy
data = os.urandom(1024)
print(hashlib.sha256(data).hexdigest())
"""


@skip_unless_hwaccel
class TestL1SnapshotRNG:
    """Verify RNG uniqueness after L1 memory snapshot restore.

    Requires hwaccel: L1 memory snapshots use QEMU migration which requires
    hardware-accelerated CPU state save/restore (KVM/HVF).

    The L1 snapshot saves full CPU+RAM+device state via QEMU migration.
    On restore, the kernel CRNG resumes from the saved state. The guest agent
    adds the host entropy of the first ping after the restore to the input pool
    and reseeds the CRNG, so each restored VM diverges. These tests check that the
    outputs differ — failure means identical random output across VMs, indicating
    a broken or missing reseed.
    """

    async def test_l1_restored_vms_different_random(self, scheduler: Scheduler) -> None:
        """Two VMs restored from the same L1 snapshot must produce different random."""
        # Step 1: Prime the L1 cache with a cold boot
        prime_result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('prime')")
        assert prime_result.exit_code == 0

        # Step 2: Wait for background L1 save to complete
        # The scheduler schedules a background save after first cold boot.
        # We need it to finish before the next calls can hit L1 cache.
        await wait_l1_saves(scheduler)

        # Step 3: Run two VMs concurrently — both should restore from L1
        runs = await asyncio.gather(
            run_timed(scheduler, VmSpec(language=Language.PYTHON), IMMEDIATE_URANDOM),
            run_timed(scheduler, VmSpec(language=Language.PYTHON), IMMEDIATE_URANDOM),
        )
        results = [run.result for run in runs]

        # Verify both came from L1 restore
        for i, run in enumerate(runs):
            assert run.result.exit_code == 0, f"VM {i} failed: {run.result.stderr}"
            assert run.startup.l1_cache_hit, f"VM {i} did NOT come from L1 restore (test is invalid without L1)"

        # Step 4: Assert RNG uniqueness — this is the bug reproduction
        hashes = [r.stdout.strip() for r in results]
        assert all(len(h) == 64 for h in hashes), f"Invalid SHA256 output: {hashes}"
        assert hashes[0] != hashes[1], (
            f"CRITICAL: L1-restored VMs produced identical random output!\n"
            f"  VM 0: {hashes[0]}\n"
            f"  VM 1: {hashes[1]}\n"
            f"Kernel CRNG state was cloned — no post-restore entropy re-seeding."
        )

    async def test_l1_restored_vms_sequential_different_random(self, scheduler: Scheduler) -> None:
        """Even sequential L1 restores must produce different random — no state replay."""
        # Step 1: Prime L1 cache
        prime_result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('prime')")
        assert prime_result.exit_code == 0

        # Step 2: Wait for L1 save
        await wait_l1_saves(scheduler)

        # Step 3: Run 3 VMs sequentially from L1
        hashes: list[str] = []
        for i in range(3):
            result, startup, _ports = await run_timed(scheduler, VmSpec(language=Language.PYTHON), IMMEDIATE_URANDOM)
            assert result.exit_code == 0, f"VM {i} failed: {result.stderr}"
            assert startup.l1_cache_hit, f"VM {i} did NOT come from L1 restore"
            hashes.append(result.stdout.strip())

        # All three must be unique
        assert len(set(hashes)) == 3, (
            f"CRITICAL: L1-restored VMs produced duplicate random output!\n"
            f"  Hashes: {hashes}\n"
            f"Kernel CRNG state was replayed from snapshot."
        )


# Math.random lives in the JS runtime, not the kernel: the guest CRNG reseed does
# not cover it. The REPL replaces it with a function that reads getrandom(2) in
# both realms (the exec context and the outer realm that require'd modules and
# process.constructor.constructor reach).
JS_RANDOM = """\
const words = new Uint32Array(4);
crypto.getRandomValues(words);
const outer = process.constructor.constructor('return [Math.random(), Math.random()]')();
console.log(JSON.stringify({ math: [Math.random(), Math.random()], outer, crypto: Array.from(words) }));
"""


async def _two_l1_restores(scheduler: Scheduler, spec: VmSpec, *, prime: str, draw: str) -> tuple[str, str]:
    """A cold boot of `spec` runs `prime` (its L1 template is saved after), then two VMs restored
    from that template run `draw`: their two stdouts."""
    result = await run_once(scheduler, spec, prime)
    assert result.exit_code == 0, result.stderr
    await wait_l1_saves(scheduler)
    runs = [await run_timed(scheduler, spec, draw) for _ in range(2)]
    for i, run in enumerate(runs):
        assert run.result.exit_code == 0, f"VM {i} failed: {run.result.stderr}"
        assert run.startup.l1_cache_hit, f"VM {i} did NOT come from L1 restore"
    return runs[0].result.stdout, runs[1].result.stdout


@skip_unless_hwaccel
class TestL1SnapshotJsRNG:
    async def test_l1_restored_js_vms_different_random(self, scheduler: Scheduler) -> None:
        outputs = await _two_l1_restores(
            scheduler, VmSpec(language=Language.JAVASCRIPT), prime="console.log('prime')", draw=JS_RANDOM
        )
        first, second = (json.loads(output.strip()) for output in outputs)
        assert first["math"] != second["math"], "Math.random state cloned from the L1 template"
        assert first["outer"] != second["outer"], "outer-realm Math.random state cloned from the L1 template"
        assert first["crypto"] != second["crypto"], "crypto.getRandomValues state cloned from the L1 template"


# bash $RANDOM lives in the shell process, not the kernel: the guest CRNG reseed does
# not cover it. The first exec after a restore carries the reseed flag, and the shell
# REPL then sets RANDOM from $SRANDOM (the kernel).
@skip_unless_hwaccel
class TestL1SnapshotShellRNG:
    async def test_l1_restored_raw_vms_different_random(self, scheduler: Scheduler) -> None:
        """bash $RANDOM gets a new seed at each restore of one L1 template."""
        first, second = await _two_l1_restores(
            scheduler, VmSpec(language=Language.RAW), prime="echo prime", draw="echo $RANDOM $RANDOM $RANDOM"
        )
        assert first != second, "bash $RANDOM state cloned from the L1 template"
