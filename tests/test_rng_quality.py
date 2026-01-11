"""RNG quality validation for VM images.

Tests that cryptographic randomness works correctly in VMs:
1. Kernel entropy pool is properly seeded (random.trust_cpu=on working)
2. /dev/urandom provides non-blocking, high-quality randomness
3. Language-specific crypto APIs work correctly
4. Different VMs produce different random sequences

Based on:
- NIST SP 800-90B entropy estimation principles
- Fourmilab ENT statistical tests
- Firecracker microVM entropy best practices

References:
- https://github.com/usnistgov/SP800-90B_EntropyAssessment
- https://www.fourmilab.ch/random/
- https://github.com/firecracker-microvm/firecracker/blob/main/docs/entropy.md
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Level 1: Kernel Entropy Health (Fast - run on every boot)
# =============================================================================
class TestKernelEntropyHealth:
    """Verify kernel CRNG is properly initialized."""

    @pytest.mark.parametrize(
        "language,code",
        [
            pytest.param(
                Language.PYTHON,
                "print(open('/proc/sys/kernel/random/entropy_avail').read().strip())",
                id="python",
            ),
            pytest.param(
                Language.JAVASCRIPT,
                "console.log(require('fs').readFileSync('/proc/sys/kernel/random/entropy_avail', 'utf8').trim())",
                id="javascript",
            ),
            pytest.param(
                Language.RAW,
                "cat /proc/sys/kernel/random/entropy_avail",
                id="raw",
            ),
        ],
    )
    async def test_entropy_pool_seeded(self, scheduler: Scheduler, language: Language, code: str) -> None:
        """CRNG has 256 bits entropy (random.trust_cpu=on working)."""
        result = await scheduler.run(code=code, language=language)

        assert result.exit_code == 0
        entropy = int(result.stdout.strip())
        # Modern kernels with CONFIG_RANDOM_TRUST_CPU maintain 256 bits
        assert entropy >= 256, f"Entropy starvation: only {entropy} bits"

    async def test_urandom_nonblocking(self, scheduler: Scheduler) -> None:
        """getrandom() doesn't block - CRNG ready at boot."""
        code = """
import os
import time

# Read 1MB - should be instant if CRNG initialized
start = time.perf_counter()
data = os.urandom(1024 * 1024)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"TIME_MS:{elapsed_ms:.2f}")
# Should complete in <100ms, blocking would take seconds
print("PASS" if elapsed_ms < 100 else "FAIL_BLOCKED")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout, "getrandom() blocked - entropy starvation"


# =============================================================================
# Level 2: ENT-style Statistical Tests (Fast - run in CI)
# =============================================================================
class TestEntStatistics:
    """Fourmilab ENT-style statistical tests.

    Reference: https://www.fourmilab.ch/random/
    """

    async def test_chi_square_byte_distribution(self, scheduler: Scheduler) -> None:
        """Chi-square test for uniform byte distribution.

        Chi-square is extremely sensitive to RNG errors.
        For 255 DOF: values 200-310 are normal (p=0.01 to p=0.99)
        """
        code = """
import os

# Generate 256KB (matches ENT default)
data = os.urandom(256 * 1024)

# Count byte frequencies
freq = [0] * 256
for b in data:
    freq[b] += 1

# Chi-square statistic
expected = len(data) / 256
chi_sq = sum((f - expected) ** 2 / expected for f in freq)

# For 255 DOF:
# - < 200: suspiciously uniform (may indicate weak RNG)
# - > 310: non-uniform distribution (definitely broken)
# - 200-310: normal range
print(f"CHI_SQ:{chi_sq:.2f}")

if chi_sq < 200:
    print("SUSPECT_TOO_UNIFORM")
elif chi_sq > 310:
    print("FAIL_NON_UNIFORM")
else:
    print("PASS")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_entropy_bits_per_byte(self, scheduler: Scheduler) -> None:
        """Shannon entropy should be ~7.99 bits/byte for random data."""
        code = """
import os
import math

data = os.urandom(256 * 1024)

# Calculate Shannon entropy
freq = [0] * 256
for b in data:
    freq[b] += 1

entropy = 0.0
for f in freq:
    if f > 0:
        p = f / len(data)
        entropy -= p * math.log2(p)

print(f"ENTROPY:{entropy:.6f}")
# Perfect random = 8.0 bits/byte, >7.9 is excellent
print("PASS" if entropy > 7.9 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_serial_correlation(self, scheduler: Scheduler) -> None:
        """Serial correlation coefficient should be near zero."""
        code = """
import os

data = os.urandom(256 * 1024)

# Serial correlation: measures dependency between consecutive bytes
n = len(data)
sum_xy = sum(data[i] * data[i+1] for i in range(n-1))
sum_x = sum(data[:-1])
sum_y = sum(data[1:])
sum_x2 = sum(b*b for b in data[:-1])
sum_y2 = sum(b*b for b in data[1:])

# Pearson correlation coefficient
num = (n-1) * sum_xy - sum_x * sum_y
den_x = ((n-1) * sum_x2 - sum_x * sum_x) ** 0.5
den_y = ((n-1) * sum_y2 - sum_y * sum_y) ** 0.5

if den_x * den_y > 0:
    corr = num / (den_x * den_y)
else:
    corr = 0

print(f"SERIAL_CORR:{corr:.6f}")
# Should be very close to 0 (< 0.01 in absolute value)
print("PASS" if abs(corr) < 0.01 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_compression_ratio(self, scheduler: Scheduler) -> None:
        """Random data should be incompressible (ratio > 0.99)."""
        code = """
import os
import zlib

data = os.urandom(256 * 1024)
compressed = zlib.compress(data, level=9)

ratio = len(compressed) / len(data)
print(f"COMPRESS_RATIO:{ratio:.4f}")
# Random data compresses poorly (ratio > 0.99)
# Weak RNG may have patterns that compress better
print("PASS" if ratio > 0.99 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_monte_carlo_pi(self, scheduler: Scheduler) -> None:
        """Monte Carlo pi estimation - tests 2D uniformity."""
        code = """
import os
import struct
import math

# Generate coordinate pairs
data = os.urandom(4 * 100000)  # 100k pairs of 16-bit coords
coords = struct.unpack(f"{len(data)//2}H", data)

inside = 0
for i in range(0, len(coords), 2):
    x = coords[i] / 65535.0
    y = coords[i+1] / 65535.0
    if x*x + y*y <= 1.0:
        inside += 1

pi_estimate = 4.0 * inside / (len(coords) // 2)
error = abs(pi_estimate - math.pi)

print(f"PI_ESTIMATE:{pi_estimate:.6f}")
print(f"ERROR:{error:.6f}")
# Error should be < 0.01 with 100k samples
print("PASS" if error < 0.01 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout


# =============================================================================
# Level 3: Language-Specific Crypto API Tests
# =============================================================================
class TestCryptoAPIs:
    """Verify crypto APIs work correctly on each runtime."""

    async def test_python_secrets_module(self, scheduler: Scheduler) -> None:
        """Python secrets module (CSPRNG) works."""
        code = """
import secrets

# Test token generation
token = secrets.token_hex(32)
assert len(token) == 64
assert all(c in "0123456789abcdef" for c in token)

# Test secure comparison (timing-safe)
a = secrets.token_bytes(32)
b = secrets.token_bytes(32)
assert not secrets.compare_digest(a, b)  # Different
assert secrets.compare_digest(a, a)      # Same

# Test randbelow
for _ in range(100):
    n = secrets.randbelow(1000)
    assert 0 <= n < 1000

print("PASS")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_python_hashlib_random(self, scheduler: Scheduler) -> None:
        """Python hashlib with random data produces unique hashes."""
        code = """
import os
import hashlib

# Generate 100 random hashes - all should be unique
hashes = set()
for _ in range(100):
    data = os.urandom(32)
    h = hashlib.sha256(data).hexdigest()
    hashes.add(h)

print(f"UNIQUE_HASHES:{len(hashes)}")
print("PASS" if len(hashes) == 100 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_javascript_crypto_random(self, scheduler: Scheduler) -> None:
        """Node/Bun crypto.randomBytes works."""
        code = """
const crypto = require("crypto");

// randomBytes (synchronous)
const buf1 = crypto.randomBytes(1024);
console.log(`randomBytes:${buf1.length}`);

// Verify different calls produce different data
const buf2 = crypto.randomBytes(1024);
const same = buf1.equals(buf2);
console.log(`different:${!same}`);

// UUID generation
const uuid = crypto.randomUUID();
console.log(`UUID_LEN:${uuid.length}`);

console.log(buf1.length === 1024 && !same && uuid.length === 36 ? "PASS" : "FAIL");
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_javascript_compression_test(self, scheduler: Scheduler) -> None:
        """JavaScript random data is incompressible."""
        code = """
const crypto = require("crypto");
const zlib = require("zlib");

// Generate 256KB random data
const data = crypto.randomBytes(256 * 1024);

// Compress it
const compressed = zlib.deflateSync(data, { level: 9 });
const ratio = compressed.length / data.length;

console.log(`RATIO:${ratio.toFixed(4)}`);
console.log(ratio > 0.99 ? "PASS" : "FAIL");
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_raw_dev_urandom(self, scheduler: Scheduler) -> None:
        """Shell access to /dev/urandom works."""
        code = """
# Test /dev/urandom read (256KB)
BYTES=$(dd if=/dev/urandom bs=1024 count=256 2>/dev/null | wc -c)
echo "BYTES:$BYTES"

# Check if we got the expected amount
if [ "$BYTES" -eq 262144 ]; then
    echo "PASS"
else
    echo "FAIL"
fi
"""
        result = await scheduler.run(code=code, language=Language.RAW)

        assert result.exit_code == 0
        assert "PASS" in result.stdout


# =============================================================================
# Level 4: Uniqueness Across VMs (Critical for Security)
# =============================================================================
class TestCrossVMUniqueness:
    """Verify different VMs produce different random sequences.

    This catches the catastrophic VM clone/snapshot vulnerability
    where all clones would generate identical keys.
    """

    async def test_different_vms_different_random(self, scheduler: Scheduler) -> None:
        """Two VMs must produce different random outputs."""
        import asyncio

        code = """
import os
import hashlib
# Generate 1KB and hash it for comparison
data = os.urandom(1024)
print(hashlib.sha256(data).hexdigest())
"""
        # Run same code in two separate VMs
        results = await asyncio.gather(
            scheduler.run(code=code, language=Language.PYTHON),
            scheduler.run(code=code, language=Language.PYTHON),
        )

        hashes = [r.stdout.strip() for r in results]
        assert len(hashes) == 2
        assert all(len(h) == 64 for h in hashes), "Invalid SHA256 output"
        assert hashes[0] != hashes[1], "CRITICAL: VMs produced identical random!"

    async def test_multiple_vms_all_unique(self, scheduler: Scheduler) -> None:
        """Five VMs must all produce unique random outputs."""
        import asyncio

        code = """
import os
import hashlib
data = os.urandom(1024)
print(hashlib.sha256(data).hexdigest())
"""
        # Run in 5 VMs concurrently
        results = await asyncio.gather(*[scheduler.run(code=code, language=Language.PYTHON) for _ in range(5)])

        hashes = [r.stdout.strip() for r in results]
        assert len(hashes) == 5
        assert len(set(hashes)) == 5, f"Duplicate hashes found: {hashes}"


# =============================================================================
# Level 5: NIST SP 800-90B Style Tests (Thorough)
# =============================================================================
class TestNistStyle:
    """NIST SP 800-90B inspired min-entropy tests.

    Reference: https://github.com/usnistgov/SP800-90B_EntropyAssessment
    """

    async def test_repetition_count(self, scheduler: Scheduler) -> None:
        """No long runs of identical bytes (IID assumption)."""
        code = """
import os

data = os.urandom(1024 * 1024)  # 1MB

# Find longest run of identical bytes
max_run = 1
current_run = 1
for i in range(1, len(data)):
    if data[i] == data[i-1]:
        current_run += 1
        max_run = max(max_run, current_run)
    else:
        current_run = 1

print(f"MAX_RUN:{max_run}")
# For 1MB of random data, runs > 4 are extremely rare (p < 10^-10)
# Runs > 6 indicate a broken RNG
print("PASS" if max_run <= 6 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_adaptive_proportion(self, scheduler: Scheduler) -> None:
        """No single byte value dominates (checks for stuck bits)."""
        code = """
import os

data = os.urandom(1024 * 1024)

# Count most frequent byte
freq = [0] * 256
for b in data:
    freq[b] += 1

max_freq = max(freq)
proportion = max_freq / len(data)

print(f"MAX_PROPORTION:{proportion:.6f}")
# Expected: ~1/256 = 0.00390625
# Allow up to 2x expected (0.0078) for statistical variation
print("PASS" if proportion < 0.008 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout

    async def test_bit_balance(self, scheduler: Scheduler) -> None:
        """Bits should be roughly 50% zeros and 50% ones."""
        code = """
import os

data = os.urandom(1024 * 1024)  # 1MB = 8M bits

# Count 1-bits
ones = sum(bin(b).count('1') for b in data)
total_bits = len(data) * 8
zeros = total_bits - ones

ratio = ones / total_bits
print(f"ONES_RATIO:{ratio:.6f}")

# Should be very close to 0.5 (within 0.001 for 8M samples)
print("PASS" if 0.499 < ratio < 0.501 else "FAIL")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "PASS" in result.stdout
