"""Cross-platform permission utilities for qemu-vm user isolation, and the
private-folder checks (owner, mode, ACL entries, folders above it) of the local
instance store, the private run folders and the cache folders.

Centralizes chmod, chown, setfacl, sudo operations with:
- Platform guards (Linux vs macOS)
- Consistent error handling
- Async subprocess wrappers
- Read-back checks (get_mode, qemu_vm_acl_perms)

Security:
- Gives qemu-vm access by ACL (setfacl), owner or group, never by world-writable
  permissions (0666)
- QEMU as qemu-vm makes its sockets 0770, group qemu-vm (QEMU_VM_SUDO), in a VM
  folder of mode 2770 (grant_qemu_vm_dir_access)
- Follows libvirt group membership pattern
"""

import asyncio
import contextlib
import ctypes
import ctypes.util
import errno
import grp
import os
import pwd
import re
import shutil
import stat
import struct
import sys
import tempfile
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Final, NamedTuple

import psutil

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.cgroup import can_contain_processes
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.files.disk import make_private_dir
from exec_sandbox.platform_utils import HostOS, ProcessWrapper, child_env, detect_host_os
from exec_sandbox.system_probes import posix_acl_xattr

# macOS <sys/acl.h>
_ACL_TYPE_EXTENDED = 0x00000100
_ACL_FIRST_ENTRY = 0

logger = get_logger(__name__)

# =============================================================================
# Cache for probes
# =============================================================================


class _ProbeCache:
    """Container for probe results to avoid global statements."""

    qemu_vm_user: bool | None = None
    qemu_as_qemu_vm: bool | None = None  # The QEMU identity of this process (one decision)
    setfacl_available: bool | None = None

    def reset(self) -> None:
        """Forget every result (tests: a probe a test mocked must not stay for the next test)."""
        self.qemu_vm_user = None
        self.qemu_as_qemu_vm = None
        self.setfacl_available = None


_probe_cache = _ProbeCache()

_AS_QEMU_VM: tuple[str, ...] = ("-u", "qemu-vm", "/bin/sh", "-c", 'umask 007 && exec "$@"', "qemu-vm")

QEMU_VM_SUDO: tuple[str, ...] = ("sudo", "-n", *_AS_QEMU_VM)
"""Command prefix that runs a command as qemu-vm with umask 007, so the sockets QEMU makes
are 0770, group qemu-vm. The shell sets the umask after sudo, because the umask sudo gives
depends on the sudo in use: sudo-rs (default sudo on Ubuntu 25.10 and later) ORs the
caller's umask with its own 022 (umask 027: sockets 0750, no group write, so the host
cannot connect); classic sudo lets PAM pam_umask set it. "$@" passes each argument
unchanged: the shell parses only the fixed script."""

# POSIX ACL xattr (system_probes.POSIX_ACL_ACCESS_XATTR): a version, then (tag, perm, id) entries
_ACL_XATTR_VERSION = 2
_ACL_USER = 0x02
_ACL_MASK = 0x10
_ACL_PERM_BITS = {"r": 4, "w": 2, "x": 1}


# =============================================================================
# Private folders (local instance store, private run folders, cache folders)
# =============================================================================


def make_owned_dir(path: Path, *, setting: str) -> None:
    """make_private_dir for a folder whose files QEMU uses (caches, assets, overlays), then
    refuse it unless it is a real folder of this user that group and others cannot write,
    with no ACL entry, in folders that only root and this user can change
    (private_tree_problem): a folder made before with another umask or group (`sg qemu-vm`
    under umask 002 gives 0775 qemu-vm), or such a folder above it, lets QEMU as qemu-vm
    rename or replace what it holds.

    Raises:
        ConfigError: SETTING_INVALID, naming `setting` and the fix
        OSError: the folder cannot be made or opened, or a check cannot run
    """
    make_private_dir(path)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        problem = private_tree_problem(path, fd, forbidden=stat.S_IWGRP | stat.S_IWOTH)
    finally:
        os.close(fd)
    if problem:
        raise ConfigError(
            f"Folder {path} {problem[0]}",
            code=ErrorCode.SETTING_INVALID,
            context={"folder": str(path), "setting": setting},
            hint=problem[1],
            instance_id=None,
        )


def private_dir_problem(root: Path, st: os.stat_result, *, forbidden: int) -> tuple[str, str] | None:
    """Why `root` (its lstat result `st`) is not a safe folder of this user: (problem, fix).

    None when it is a real folder (not a symlink), owned by this user, with
    none of the `forbidden` mode bits set. Callers raise their own error class.
    """
    uid = os.geteuid()
    if not stat.S_ISDIR(st.st_mode):
        return "is a symlink or not a folder", "Use a real folder, not a link to one."
    if st.st_uid != uid:
        return (
            f"belongs to uid {st.st_uid}, not to this user (uid {uid})",
            f"A folder made with sudo belongs to root: run `sudo chown -R {uid} {root}`, or use a folder of this user.",
        )
    if st.st_mode & forbidden:
        return f"has mode {oct(stat.S_IMODE(st.st_mode))}: group or others have access", f"Run `chmod 700 {root}`."
    return None


def private_tree_problem(root: Path, fd: int, *, forbidden: int) -> tuple[str, str] | None:
    """Why the folder open at `fd` (opened through the path `root`) is not safe: (problem, fix).

    private_dir_problem of the open folder, then:
    - no ACL entry on it: mode bits do not show ACL entries, and a parent with an
      inheritable ACL gives them to each new folder (macOS);
    - the real path of `root` still names the open folder (same device and inode);
    - no folder above it that another user can change (OpenSSH StrictModes, auth.c
      safe_path): each one belongs to root or this user, and has no other write bit,
      nor a group write bit unless its group is this user's private group and it has no
      ACL entry (_group_is_this_users; on Linux the group bits of a folder with ACL
      entries are the ACL mask, which a named user's write sets; a ~/.cache made under
      umask 002 is 0775), unless it is sticky (as /tmp: others cannot rename what they
      do not own); on macOS, also no ACL entry that lets another user or a group change
      it (_macos_acl_write_entry). Else another user can put another folder in its place.
    Raises OSError when a check cannot run. Callers raise their own error class.
    """
    st = os.fstat(fd)
    if problem := private_dir_problem(root, st, forbidden=forbidden):
        return problem
    if has_extended_acl(fd):
        return (
            "has ACL entries, which can give other users access",
            f"Remove them with `chmod -R -N {root}` (macOS) or `setfacl -R -b {root}` (Linux), "
            "and make the folder in a parent folder without inheritable ACL entries.",
        )
    real = root.resolve(strict=True)
    at_real = real.stat()
    if (at_real.st_dev, at_real.st_ino) != (st.st_dev, st.st_ino):
        return "was moved or replaced during its check", "Find out who else can change its parent folders."
    uid = os.geteuid()
    macos = detect_host_os() is HostOS.MACOS
    for parent in real.parents:
        pst = parent.lstat()
        mode = pst.st_mode
        if macos and (entry := _macos_acl_write_entry(parent)) is not None:
            return (
                f"is in {parent}, whose ACL entry `{entry}` lets another user change it",
                f"Remove the entry with `chmod -a# <index> {parent}` (see `ls -led {parent}`), or use another folder.",
            )
        if not stat.S_ISDIR(mode) or pst.st_uid not in (0, uid):  # its owner can change it, whatever its mode
            writable = not mode & stat.S_ISVTX and mode & (stat.S_IWGRP | stat.S_IWOTH)
            return (
                f"is in {parent} (uid {pst.st_uid}, mode {oct(stat.S_IMODE(mode))}), which another user can change",
                f"Use a folder whose parent folders belong to root or this user, or give {parent} to root "
                f"(`sudo chown root {parent}`: uid {pst.st_uid} then no longer owns it)"
                + (f", then `sudo chmod go-w {parent}` or `sudo chmod +t {parent}`." if writable else "."),
            )
        if not mode & stat.S_ISVTX and (
            mode & stat.S_IWOTH
            or (mode & stat.S_IWGRP and not (_group_is_this_users(pst.st_gid) and not _has_acl_at(parent)))
        ):
            return (
                f"is in {parent} (uid {pst.st_uid}, mode {oct(stat.S_IMODE(mode))}), which another user can change",
                f"Use a folder whose parent folders only root and this user can write, run `chmod go-w {parent}`, "
                f"or, when others must keep writing there (a shared volume), `chmod +t {parent}` (a sticky folder passes, as /tmp). "
                "Only the folder's owner can run these: root (sudo) for a folder of root, an init container that runs as root "
                "for a Kubernetes emptyDir.",
            )
    return None


def _group_is_this_users(gid: int) -> bool:
    """Whether `gid` is this user's private group, so that its write bit gives no other user
    access: the user's primary group, named as the user (Debian, Ubuntu and Fedora give each
    user one, and pam_umask gives umask 002 only then), with no other listed member and no
    other local account (/etc/passwd) with it as primary group, and no second local record
    with its gid (/etc/group: groupadd -o) under another name or with members. A group with no listed
    member is not enough: setgid programs use such groups, and accounts from LDAP or SSSD
    can have it as primary group. Never on macOS: grp does not show its computed and nested
    members (everyone, localaccounts). False when a lookup fails."""
    if detect_host_os() is HostOS.MACOS:
        return False
    try:
        me = pwd.getpwuid(os.geteuid())
        group = grp.getgrgid(gid)
        primary = _local_users_with_primary_group(gid)
        records = _local_group_records(gid)
    except (KeyError, OSError):
        return False
    return (
        gid == me.pw_gid
        and group.gr_name == me.pw_name
        and set(group.gr_mem) <= {me.pw_name}
        and primary <= {me.pw_name}
        and all(name == me.pw_name and members <= {me.pw_name} for name, members in records)
    )


_MACOS_ACL_WRITE_RIGHTS: Final = frozenset({"write", "append", "delete", "delete_child", "writesecurity", "chown"})
"""macOS ACL rights (acl_to_text names: add_file is `write`, add_subdirectory `append`) that
let their holder replace what a folder holds, rename it away, or change who may."""


@lru_cache(maxsize=1)
def _macos_libc() -> ctypes.CDLL:
    """macOS libc with the prototypes of its ACL calls, loaded once: find_library costs
    hundreds of microseconds, too much for each checked parent folder."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.acl_get_link_np.restype = ctypes.c_void_p
    libc.acl_get_link_np.argtypes = (ctypes.c_char_p, ctypes.c_int)
    libc.acl_get_fd_np.restype = ctypes.c_void_p
    libc.acl_get_fd_np.argtypes = (ctypes.c_int, ctypes.c_int)
    libc.acl_to_text.restype = ctypes.c_void_p
    libc.acl_to_text.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    libc.acl_get_entry.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
    libc.acl_free.argtypes = (ctypes.c_void_p,)
    return libc


def _macos_acl_write_entry(path: Path) -> str | None:
    """The first ACL entry of the folder `path` (its last name not followed) that allows one
    of _MACOS_ACL_WRITE_RIGHTS to a user other than this one, or to a group (macOS groups
    have computed members), or None. Mode bits do not show ACL entries on macOS. Deny
    entries pass: each home has `group:everyone deny delete`. An entry that only applies to
    new children (only_inherit) does not apply to the folder. Raises OSError when the ACL
    cannot be read."""
    libc = _macos_libc()
    acl = libc.acl_get_link_np(os.fsencode(path), _ACL_TYPE_EXTENDED)
    if not acl:
        err = ctypes.get_errno()
        if err in (errno.ENOENT, errno.ENOTSUP, errno.EOPNOTSUPP):  # no ACL, or a filesystem without ACLs
            return None
        raise OSError(err, os.strerror(err))
    try:
        text = libc.acl_to_text(acl, None)
        if not text:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        try:
            lines = ctypes.string_at(text).decode(errors="replace").splitlines()
        finally:
            libc.acl_free(text)
    finally:
        libc.acl_free(acl)
    for line in lines[1:]:  # the first line is the "!#acl 1" header
        fields = line.split(":")
        if len(fields) < 6:  # noqa: PLR2004 - tag:uuid:name:id:allow|deny:rights
            continue
        # tag:uuid:name:id:allow|deny[,flags]:rights; the flags (inherited, file_inherit,
        # directory_inherit, only_inherit, limit_inherit) follow the kind in the same field
        tag, qualifier_id, flags, rights = fields[0], fields[3], set(fields[4].split(",")), set(fields[5].split(","))
        if "allow" not in flags or "only_inherit" in flags or not rights & _MACOS_ACL_WRITE_RIGHTS:
            continue
        if tag == "user" and qualifier_id == str(os.geteuid()):
            continue
        return line
    return None


def _has_acl_at(path: Path) -> bool:
    """has_extended_acl of the folder at `path` (its last name not followed); True when it
    cannot be opened, so that the caller refuses it."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return True
    try:
        return has_extended_acl(fd)
    finally:
        os.close(fd)


def has_extended_acl(fd: int) -> bool:
    """True when the file open at `fd` has ACL entries beyond its mode bits.

    macOS: acl_get_fd_np(fd, ACL_TYPE_EXTENDED) from libc (sys/acl.h). With no ACL
    it returns NULL and errno ENOENT, the same as gnulib (lib/acl-errno-valid.c,
    lib/file-has-acl.c, which coreutils `ls` uses); an ACL with no entry is no ACL.
    Linux: the kernel keeps an access ACL with more than the 3 base entries, or a
    default ACL, in the xattr system.posix_acl_access or system.posix_acl_default
    (acl(5)); a base-only ACL is never stored. A filesystem without ACLs has none.
    """
    if sys.platform == "darwin":
        libc = _macos_libc()
        acl = libc.acl_get_fd_np(fd, _ACL_TYPE_EXTENDED)
        if not acl:
            err = ctypes.get_errno()
            if err in (errno.ENOENT, errno.ENOTSUP, errno.EOPNOTSUPP):  # no ACL, or a filesystem without ACLs
                return False
            raise OSError(err, os.strerror(err))
        try:
            entry = ctypes.c_void_p()
            return libc.acl_get_entry(acl, _ACL_FIRST_ENTRY, ctypes.byref(entry)) == 0  # 0: an entry, -1: none
        finally:
            libc.acl_free(acl)
    if sys.platform == "linux":
        for name in ("system.posix_acl_access", "system.posix_acl_default"):
            try:
                os.getxattr(fd, name)
            except OSError as e:
                if e.errno in (errno.ENODATA, errno.EOPNOTSUPP):
                    continue
                raise
            return True
    return False


# =============================================================================
# qemu-vm User State
# =============================================================================


@lru_cache(maxsize=1)
def get_qemu_vm_uid() -> int | None:
    """Get UID for qemu-vm user (cached).

    Returns:
        UID of qemu-vm user, or None if user doesn't exist
    """
    try:
        return pwd.getpwnam("qemu-vm").pw_uid
    except KeyError:
        return None


def get_qemu_vm_gid() -> int | None:
    """GID of the qemu-vm user's group: the group QEMU runs with as qemu-vm (sudo -u),
    which probe_qemu_as_qemu_vm checks this process is in. One passwd lookup (once per VM).

    Returns:
        GID of the qemu-vm user's group, or None if the user doesn't exist
    """
    try:
        return pwd.getpwnam("qemu-vm").pw_gid
    except KeyError:
        return None


def get_expected_socket_uid(use_qemu_vm_user: bool) -> int:
    """Get expected UID for socket authentication.

    Used to verify QEMU process identity before sending commands.
    Falls back to current user if qemu-vm user not available.

    Args:
        use_qemu_vm_user: Whether QEMU is running as qemu-vm user

    Returns:
        Expected UID of QEMU process for socket peer verification
    """
    if use_qemu_vm_user:
        uid = get_qemu_vm_uid()
        if uid is not None:
            return uid
    return os.getuid()


async def probe_qemu_vm_user() -> bool:
    """Check if qemu-vm user exists for process isolation (cached).

    Returns False on macOS since qemu-vm user is Linux-only.

    Returns:
        True if qemu-vm user exists, False otherwise
    """
    # Fast path: return cached result
    if _probe_cache.qemu_vm_user is not None:
        return _probe_cache.qemu_vm_user

    # Skip on non-Linux - not applicable
    if detect_host_os() != HostOS.LINUX:
        _probe_cache.qemu_vm_user = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/id",
            "qemu-vm",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.qemu_vm_user = proc.returncode == 0
        if _probe_cache.qemu_vm_user:
            logger.info("qemu-vm user available (process isolation enabled)")
        else:
            logger.debug("qemu-vm user not found (process isolation disabled)")
    except (OSError, TimeoutError) as e:
        logger.debug(f"qemu-vm user probe failed: {e}")
        _probe_cache.qemu_vm_user = False

    return _probe_cache.qemu_vm_user


async def probe_qemu_as_qemu_vm() -> bool:
    """Whether this process runs QEMU as qemu-vm: one decision per process, cached.

    True only when the whole qemu-vm path works for this process:
    - the qemu-vm user exists (Linux), and this process is another user;
    - this process can put a process in a VM cgroup (a delegated cgroup v2 subtree):
      only a cgroup ends a qemu-vm QEMU and proves its end;
    - `sudo -n` runs a command as qemu-vm through QEMU_VM_SUDO; the command reports its
      umask (it must leave group write: QEMU makes the guest and QMP sockets) and its gid;
    - this process can connect to those sockets: it is root, or that gid is in its groups
      now (a group added after login, or a launcher that drops groups, such as CodSpeed
      memtrack, gives no access);
    - that group has no member other than qemu-vm and this user (_other_members_of);
    - `sudo -n chown` and `sudo -n rm` work as root (VM disks, VmWorkingDirectory.cleanup).
    Else every VM of this process runs QEMU as this user, with one warning when the
    qemu-vm user exists. A process never mixes the two identities.
    """
    if _probe_cache.qemu_as_qemu_vm is not None:
        return _probe_cache.qemu_as_qemu_vm
    if detect_host_os() != HostOS.LINUX or not await probe_qemu_vm_user():
        decision, problem = False, None
    elif os.geteuid() == get_qemu_vm_uid():
        decision, problem = False, None  # QEMU already runs as qemu-vm: no sudo
    else:
        problem = await _qemu_vm_path_problem()
        decision = problem is None
    # First finished probe wins: concurrent callers get one answer and one log line.
    if _probe_cache.qemu_as_qemu_vm is None:
        _probe_cache.qemu_as_qemu_vm = decision
        if decision:
            logger.info("QEMU runs as qemu-vm")
        elif problem is not None:
            logger.warning(f"QEMU runs as this user, not as qemu-vm (less isolation): {problem}")
    return _probe_cache.qemu_as_qemu_vm


def _read_passwd() -> str:
    """/etc/passwd. Raises OSError when it cannot be read: no answer is not "no member"."""
    return Path("/etc/passwd").read_text(errors="replace")


_PASSWD_ID: Final = re.compile(r"[ \t\n\v\f\r]*([+-]?)([0-9]+)")
_ID_MAX: Final = 0xFFFFFFFF
_ULONG_DIGITS: Final = 20
"""Digits of 2**64 - 1, past its leading zeros: a longer number overflows strtoul."""


def _parse_id(field: str) -> int | None:
    """A uid or gid field as glibc reads it (strtoul in base 10, ended by the colon, then the
    32-bit check of nss_files): "0986" and " 986" are 986; a minus wraps modulo 2**64, so
    "-0" is 0 and "-18446744073709550630" is 986; a value above 0xffffffff ("-1"), "9_86",
    "986 " and "" make the line invalid (None)."""
    if (match := _PASSWD_ID.fullmatch(field)) is None:
        return None
    digits = match.group(2).lstrip("0") or "0"
    if len(digits) > _ULONG_DIGITS:  # strtoul: ULONG_MAX and ERANGE on overflow, with a sign too
        return None  # (and before int(): Python refuses strings of more than 4300 digits)
    value = int(digits)
    if value > 2**64 - 1:
        return None
    if match.group(1) == "-":
        value = -value % 2**64
    return value if value <= _ID_MAX else None


def _files_records(text: str, *, compat_lines: bool) -> Iterator[list[str]]:
    """The records of /etc/passwd or /etc/group as glibc's nss_files reads them: lines end at
    "\n" only (not \r, \x1c, \x85), a NUL ends the line (C string), C-locale blanks are
    skipped, then blank and '#' lines. NIS compat lines (+name, -name) are records only with
    `compat_lines`: getent passwd skips them, but initgroups reads such /etc/group lines
    (a member there gets the gid). Each record: its fields split at ':'."""
    for raw in text.split("\n"):
        line = raw.partition("\0")[0].lstrip(" \t\n\v\f\r")
        if not line or line.startswith("#") or (not compat_lines and line[:1] in ("+", "-")):
            continue
        yield line.split(":")


def _local_users_with_primary_group(gid: int) -> set[str]:
    """The local accounts (/etc/passwd) whose primary group is `gid`. Not pwd.getpwall: it
    holds the GIL for the whole listing, which with LDAP or SSSD enumeration goes over the
    network and freezes the event loop; qemu-vm's group is a local system group."""
    users: set[str] = set()
    for fields in _files_records(_read_passwd(), compat_lines=False):
        if len(fields) < 4:  # noqa: PLR2004 - name:password:uid:gid
            continue
        if _parse_id(fields[2]) is not None and _parse_id(fields[3]) == gid:
            users.add(fields[0])
    return users


def _read_group() -> str:
    """/etc/group. Raises OSError when it cannot be read."""
    return Path("/etc/group").read_text(errors="replace")


def _local_group_records(gid: int) -> list[tuple[str, set[str]]]:
    """Every local group record (/etc/group) with `gid`: (name, members). getgrgid gives the
    first one only, but initgroups gives the gid to the members of each (groupadd -o)."""
    records: list[tuple[str, set[str]]] = []
    for fields in _files_records(_read_group(), compat_lines=True):
        if len(fields) < 3:  # noqa: PLR2004 - name:password:gid[:members]
            continue
        # a compat line's empty gid reads as 0 (initgroups); a bad gid makes the line invalid
        record_gid = 0 if fields[0][:1] in ("+", "-") and not fields[2].strip() else _parse_id(fields[2])
        if record_gid == gid:
            members: set[str] = {m for m in fields[3].split(",") if m} if len(fields) > 3 else set()  # noqa: PLR2004
            records.append((fields[0], members))
    return records


def _other_members_of(gid: int) -> str | None:
    """Why the group QEMU runs with (gid) is not private to qemu-vm and this process's user,
    or None. Its members can open the VM folders (2770) and connect to the guest and QMP
    sockets (0770): a group shared with other users would hand them every VM."""
    try:
        group = grp.getgrgid(gid)
    except KeyError:
        return f"the group of qemu-vm ({gid}) has no name; give qemu-vm its own group, qemu-vm"
    allowed = {"qemu-vm", pwd.getpwuid(os.geteuid()).pw_name}
    try:
        records = _local_group_records(gid)  # getgrgid shows the first record only (groupadd -o)
        members = set(group.gr_mem) | _local_users_with_primary_group(gid)
    except OSError as e:
        return f"/etc/passwd or /etc/group cannot be read to check who else has group {group.gr_name} ({gid}): {e.strerror or e}"
    for _name, record_members in records:
        members |= record_members
    if group.gr_name != "qemu-vm" or any(name != "qemu-vm" for name, _ in records) or members - allowed:
        others = ", ".join(sorted(members - allowed)) or "none"
        return (
            f"QEMU would run with group {group.gr_name} ({gid}), which other users share (other members: {others}); "
            "they could reach every VM's sockets: give qemu-vm its own group, qemu-vm, with only this user in it"
        )
    return None


async def _qemu_vm_path_problem() -> str | None:
    """Why the qemu-vm path does not work for this process (probe_qemu_as_qemu_vm), or None."""
    if not await can_contain_processes():
        return (
            "this process cannot put a process in a VM cgroup (no delegated cgroup v2 subtree it runs "
            "inside), and only a cgroup ends a qemu-vm QEMU and proves its end; see the README's cgroup setup"
        )
    # Each sudo of the probe runs with -k: cached credentials (a sudo typed in this terminal
    # minutes ago) would pass now and fail every later VM start when they expire.
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "-n",
            "-k",
            *_AS_QEMU_VM,
            "/bin/sh",
            "-c",
            "umask; id -g",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except (OSError, TimeoutError) as e:
        return f"`sudo -n -u qemu-vm` could not run ({e}); add `<user> ALL=(qemu-vm) NOPASSWD: ALL` to sudoers"
    lines = stdout.decode(errors="replace").split()
    if proc.returncode != 0 or len(lines) != 2 or not all(line.isdigit() for line in lines):  # noqa: PLR2004
        return (
            f"`sudo -n -u qemu-vm` failed (exit {proc.returncode}); add `<user> ALL=(qemu-vm) NOPASSWD: ALL` to sudoers"
        )
    umask, gid = int(lines[0], 8), int(lines[1])
    if umask & stat.S_IWGRP:
        return f"commands run as qemu-vm get umask {lines[0]}: QEMU sockets would have no group write"
    if os.geteuid() != 0 and gid != os.getegid() and gid not in os.getgroups():
        return (
            f"this process is not in group {gid} (qemu-vm) now, so it cannot connect to the sockets QEMU makes; "
            "add the user to the group and log in again, or start the service with `sg qemu-vm`"
        )
    if (shared := await asyncio.to_thread(_other_members_of, gid)) is not None:
        return shared
    folder = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="exec-sandbox-qemu-vm-probe-"))
    try:
        probe = folder / "probe"
        probe.touch()
        for command, run in (
            ("chown", ["-k", "chown", "qemu-vm:qemu-vm", str(probe)]),
            ("rm", ["-k", "rm", "-rf", str(folder)]),
        ):
            proc = await sudo_exec(run)
            await proc.communicate()
            if proc.returncode != 0:
                return f"`sudo -n {command}` as root does not work; add the sudoers lines of the README"
    finally:
        await asyncio.to_thread(shutil.rmtree, folder, ignore_errors=True)
    return None


# =============================================================================
# ACL Operations (Linux only, no-op on macOS)
# =============================================================================


async def _probe_setfacl() -> bool:
    """Check if setfacl is available (cached)."""
    if _probe_cache.setfacl_available is not None:
        return _probe_cache.setfacl_available

    if detect_host_os() != HostOS.LINUX:
        _probe_cache.setfacl_available = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "setfacl",
            "--version",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.setfacl_available = proc.returncode == 0
    except (OSError, TimeoutError):
        _probe_cache.setfacl_available = False

    return _probe_cache.setfacl_available


async def set_acl_user(paths: Sequence[Path], *, user: str, perms: str) -> bool:
    """Set the ACL entry of one user on files: setfacl -m u:{user}:{perms} {paths} (one process).

    Args:
        paths: File or directory paths
        user: Username to grant access
        perms: Permission string (e.g., "rw", "rwx", "r"); replaces the user's entry

    Returns:
        True if successful, False otherwise (including on macOS)
    """
    if not await _probe_setfacl():
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "setfacl",
            "-m",
            f"u:{user}:{perms}",
            *(str(path) for path in paths),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"setfacl failed: {stderr.decode().strip()}")
            return False
        logger.debug(f"Set ACL u:{user}:{perms} on {[str(path) for path in paths]}")
        return True
    except OSError as e:
        logger.debug(f"setfacl failed: {e}")
        return False


# =============================================================================
# Mode Operations (chmod)
# =============================================================================


async def chmod_async(path: Path, mode: str) -> bool:
    """Change file mode bits asynchronously.

    Args:
        path: File or directory path
        mode: Mode string - octal ('755') or symbolic ('a+x', 'u+rw')

    Returns:
        True if successful, False otherwise
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "chmod",
            mode,
            str(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"chmod {mode} {path} failed: {stderr.decode().strip()}")
            return False
        return True
    except OSError as e:
        logger.debug(f"chmod failed: {e}")
        return False


async def chmod_executable(path: Path) -> None:
    """Set file as executable (0o755).

    Uses Path.chmod() in a thread to avoid blocking.

    Args:
        path: File path to make executable
    """
    await asyncio.to_thread(path.chmod, 0o755)


def get_mode(path: Path) -> int | None:
    """Get file mode bits.

    One metadata call: direct, a worker thread costs 30 times more.

    Args:
        path: File or directory path

    Returns:
        Mode bits as integer, or None if file doesn't exist
    """
    try:
        return path.stat().st_mode
    except OSError:
        return None


# =============================================================================
# Ownership Operations (chown)
# =============================================================================


async def chown_async(path: Path, user: str, group: str, *, owner_uid: int) -> bool:
    """Give the regular file `path` of uid `owner_uid` to user:group, as root (sudo -n).

    Root follows a link: a link planted in its place would give any file of the host away.
    So the path must be a regular file (not a link) of `owner_uid` when it is checked, and
    chown -h changes a link that replaces it after the check, never its target.

    Args:
        path: File path
        user: New owner username
        group: New owner group
        owner_uid: The uid the file must have now

    Returns:
        True if successful, False otherwise
    """
    try:
        st = path.lstat()
    except OSError as e:
        logger.debug(f"chown {user}:{group} {path} refused: {e.strerror or e}")
        return False
    if not stat.S_ISREG(st.st_mode) or st.st_uid != owner_uid:
        logger.debug(f"chown {user}:{group} {path} refused: not a regular file of uid {owner_uid}")
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "-n",  # never wait for a password
            "chown",
            "-h",  # a link that replaced the file after the check: the link itself
            f"{user}:{group}",
            "--",
            str(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"chown {user}:{group} {path} failed: {stderr.decode().strip()}")
            return False
        return True
    except OSError as e:
        logger.debug(f"chown failed: {e}")
        return False


# =============================================================================
# Privileged Execution (sudo)
# =============================================================================


async def sudo_exec(
    args: list[str],
    stdout: int | None = asyncio.subprocess.PIPE,
    stderr: int | None = asyncio.subprocess.PIPE,
) -> ProcessWrapper:
    """Run command with `sudo -n` in a new session: never a password prompt (with a terminal,
    sudo would wait for one and the caller would hang); without a NOPASSWD rule the command fails.

    Args:
        args: Command and arguments (without 'sudo -n' prefix)
        stdout: stdout handling (default PIPE)
        stderr: stderr handling (default PIPE)

    Returns:
        ProcessWrapper for the sudo process
    """
    return ProcessWrapper(
        await asyncio.create_subprocess_exec(
            "sudo",
            "-n",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            stdout=stdout,
            stderr=stderr,
            env=child_env(),
        )
    )


class _Left(NamedTuple):
    """A process that a sudo rm left running, and whether this process may reap it: an
    orphan root rm that became this process's child (as PID 1 or a subreaper), not sudo,
    which asyncio's child watcher reaps."""

    process: psutil.Process
    reap: bool


_left_rm: dict[str, list[_Left]] = {}
"""Path -> the processes a timed-out or cancelled sudo rm left (the root rm runs on after
the kill): while one runs, a new call starts no other rm on the same tree."""


_ERRORS_KEPT_BYTES: Final = 1024
"""The part of sudo rm's stderr that its failure warning quotes."""


@dataclass(frozen=True)
class SudoErrors:
    """Where a sudo rm writes its stderr (sudo_errors_file): `file`, or None when none could
    be made; `problem`, why its text may be missing ("" when its first KiB is reserved)."""

    file: BinaryIO | None
    problem: str

    def text(self) -> str:
        """The first KiB that rm wrote, or why there is none."""
        if self.file is None:
            return f"(not kept: {self.problem})"
        self.file.seek(0)
        text = self.file.read(_ERRORS_KEPT_BYTES).rstrip(b"\0").decode(errors="replace").strip()
        return f"(not kept: {self.problem})" if not text and self.problem else text


@contextlib.contextmanager
def sudo_errors_file(path: str) -> Generator[SudoErrors, None, None]:
    """An unlinked file for the stderr of a sudo rm of `path`: a file, not a pipe, as the root
    rm would hold a pipe after sudo ends. Its first KiB is reserved where the system can
    (Linux), so a full disk still keeps the text. A reserve that fails keeps the file: the
    root rm may still write where this user cannot (the ext4 root reserve, a user quota). No
    file when the temp folder refuses one (no inode, read-only): the rm then runs without its
    error text, as the folders that would free room may be the ones it removes."""
    errors: BinaryIO | None = None
    problem = ""
    try:
        errors = tempfile.TemporaryFile()  # noqa: SIM115 - closed by the with below
    except OSError as e:
        logger.debug("No file for the errors of sudo rm", extra={"path": path, "error": str(e)})
        problem = f"no file in the temp folder: {e.strerror}"
    if errors is None:  # outside the except: an error thrown in gets no OSError context
        yield SudoErrors(file=None, problem=problem)
        return
    with errors:
        if sys.platform == "linux":  # macOS has no posix_fallocate
            try:
                os.posix_fallocate(errors.fileno(), 0, _ERRORS_KEPT_BYTES)
            except OSError as e:
                problem = f"no room reserved in the temp folder: {e.strerror}"
        yield SudoErrors(file=errors, problem=problem)


def _ended(left: _Left) -> bool:
    """Whether a kept process ended: gone, or a zombie (reaped here when this process may).
    A process this user may not inspect (hidepid=1) still runs."""
    process = left.process
    try:
        if not process.is_running():
            return True
        if process.status() != psutil.STATUS_ZOMBIE:
            return False
        if left.reap and process.ppid() == os.getpid():
            with contextlib.suppress(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)
    except psutil.AccessDenied:
        return False
    except psutil.NoSuchProcess:  # ZombieProcess too
        pass
    return True


def _ended_left_rm() -> list[tuple[str, int]]:
    """Each path of _left_rm whose processes all ended, with how many were judged (a thread:
    it reads /proc)."""
    return [(path, len(left)) for path, left in list(_left_rm.items()) if all(_ended(entry) for entry in left)]


def _drop_ended(judged: list[tuple[str, int]]) -> None:
    """Drop the entries that _ended_left_rm judged ended, unless a call added to one since."""
    for path, count in judged:
        if len(_left_rm.get(path, ())) == count:
            _left_rm.pop(path, None)


def _keep(path: str, processes: list[psutil.Process]) -> None:
    """Keep what a sudo rm of `path` left running (this process's own children, such as sudo,
    are left to asyncio to reap)."""
    kept: list[_Left] = []
    for process in processes:
        try:
            kept.append(_Left(process=process, reap=process.ppid() != os.getpid()))
        except psutil.Error:
            kept.append(_Left(process=process, reap=False))
    _left_rm.setdefault(path, []).extend(kept)


def _sudo_children(pid: int | None) -> list[psutil.Process]:
    try:
        return psutil.Process(pid).children(recursive=True)
    except (psutil.Error, ValueError):  # ended (NoSuchProcess), or not readable
        return []


def _sudo_and_children(pid: int | None) -> list[psutil.Process]:
    """sudo and what it started, for a sudo_rm that was cancelled before its end."""
    try:
        return [psutil.Process(pid), *_sudo_children(pid)]
    except (psutil.Error, ValueError):
        return []


async def _spawn_sudo_rm(key: str, errors: SudoErrors) -> ProcessWrapper:
    """Start `sudo -n rm -rf -- key`. A cancel waits for the spawn (shielded: the errors file
    must stay open until the child has its copy), keeps what it started, then goes on."""
    spawn = asyncio.ensure_future(
        sudo_exec(
            ["rm", "-rf", "--", key],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL if errors.file is None else errors.file.fileno(),
        )
    )
    try:
        return await asyncio.shield(spawn)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            proc = await asyncio.shield(spawn)
            _keep(key, _sudo_and_children(proc.pid))  # it runs on: no second rm
        raise


async def sudo_rm(path: Path) -> bool:
    """Remove file or directory with sudo, bounded by constants.SUDO_RM_TIMEOUT_SECONDS.

    sudo's stdout goes to /dev/null and its stderr to an unlinked file: sudo forks the root
    rm, which would hold sudo's pipes, so a wait for the pipes would wait for rm, which this
    user cannot kill (a slow or stuck rm would then hold the destroy). On expiry sudo is killed and
    the root rm runs on by itself: it is kept, and while it runs a new call for the same path
    starts no other (a cancel keeps sudo and its rm the same way). The path is checked again
    after the kill, as rm may have ended meanwhile.

    Args:
        path: Path to remove

    Returns:
        True if the path is gone, False otherwise
    """
    key = str(path)
    if _left_rm:  # drop what ended, for any path: the caller may never call again for it
        _drop_ended(await asyncio.to_thread(_ended_left_rm))
    if key in _left_rm:
        logger.debug("sudo rm of an earlier call still runs", extra={"path": key})
        return False
    try:
        with sudo_errors_file(key) as errors:
            proc = await _spawn_sudo_rm(key, errors)
            left: list[psutil.Process] = []
            try:
                try:
                    async with asyncio.timeout(constants.SUDO_RM_TIMEOUT_SECONDS):
                        returncode = await proc.wait()
                except TimeoutError:
                    left = await asyncio.to_thread(_sudo_children, proc.pid)
                    # A stuck sudo (hostname, PAM, SSSD) or a slow rm: the caller keeps the folder
                    with contextlib.suppress(ProcessLookupError):  # sudo ended meanwhile
                        await proc.kill()
                    await proc.wait()
                    if not await asyncio.to_thread(os.path.lexists, path):
                        return True
                    _keep(key, left)  # a concurrent call's processes stay too
                    logger.warning(
                        "sudo rm did not end in time; the folder stays",
                        extra={"path": key, "timeout_seconds": constants.SUDO_RM_TIMEOUT_SECONDS},
                    )
                    return False
            except asyncio.CancelledError:  # at any await: sudo or its rm runs on, no second rm
                _keep(key, [*left, *_sudo_and_children(proc.pid)])
                raise
            if returncode != 0:
                logger.warning("sudo rm failed", extra={"path": key, "exit_code": returncode, "stderr": errors.text()})
                return False
            return True
    except OSError as e:
        logger.debug(f"sudo rm failed: {e}")
        return False


# =============================================================================
# High-level Helpers
# =============================================================================


async def grant_qemu_vm_access(path: Path) -> bool:
    """Grant qemu-vm user read/write access to a file via ACL.

    This is more secure than chmod 666 as it only grants access to
    the specific user that needs it.

    No-op if qemu-vm user doesn't exist or on macOS.

    Args:
        path: File path (typically a socket)

    Returns:
        True if ACL was set, False if skipped or failed
    """
    if not await probe_qemu_vm_user():
        logger.debug("qemu-vm user not available, skipping ACL")
        return False

    return await set_acl_user([path], user="qemu-vm", perms="rw")


async def grant_qemu_vm_file_access(path: Path, *, writable: bool) -> bool:
    """Grant qemu-vm user access to a file, including parent directory traversal.

    Makes all parent directories traversable (a+x) up to the filesystem root (not
    included: ensure_traversable), then:
    - writable=True: gives the file of this user to qemu-vm:qemu-vm (chown_async: read+write)
    - writable=False: makes the file world-readable (a+r); a file that already is
      needs nothing (chmod by a user who does not own the file fails, even with no change)

    Args:
        path: File path to grant access to
        writable: Whether qemu-vm needs write access

    Returns:
        True if access was granted, False if any step failed
    """
    if not await ensure_traversable(path):
        return False

    if writable:
        return await chown_async(path, "qemu-vm", "qemu-vm", owner_uid=os.geteuid())
    return _make_others_readable(path)


def _make_others_readable(path: Path) -> bool:
    """a+r on `path` through one descriptor opened with O_NOFOLLOW: only a regular file of
    this user, never the target of a link (a link planted in its place would make a
    private file of this user readable by all). A file already readable needs nothing."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        logger.debug("a+r refused", extra={"path": str(path), "error": e.strerror or str(e)})
        return False
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return False
        if st.st_mode & stat.S_IROTH:
            return True
        if st.st_uid != os.geteuid():
            return False
        os.fchmod(fd, stat.S_IMODE(st.st_mode) | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError as e:
        logger.debug("a+r failed", extra={"path": str(path), "error": e.strerror or str(e)})
        return False
    finally:
        os.close(fd)
    return True


QEMU_VM_DIR_MODE = 0o2770
"""Mode of a folder this process shares with QEMU as qemu-vm (grant_qemu_vm_dir_access)."""


def grant_qemu_vm_dir_access(path: Path) -> bool:
    """Give a folder of this process to group qemu-vm with mode 2770 (QEMU_VM_DIR_MODE).

    QEMU as qemu-vm makes its sockets in the folder. This process and qemu-vm get rwx
    through owner and group. Other users get nothing: they cannot remove or replace a
    socket, or put a folder at a socket path. The setgid bit gives each new file the
    group qemu-vm.

    No sudo: the owner of a file may give it to a group it is in (chown(2)), and
    probe_qemu_as_qemu_vm checks that this process is in the group (or is root). The
    changes go through one descriptor opened with O_NOFOLLOW (never the target of a
    link). Then the mode is read back: chmod drops the setgid bit with no error when
    this process is not in the group.

    Returns:
        True when the folder has group qemu-vm and mode 2770; False when the qemu-vm
        user is missing or a step failed (EPERM: this process is not in the group)
    """
    gid = get_qemu_vm_gid()
    if gid is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchown(fd, -1, gid)
            os.fchmod(fd, QEMU_VM_DIR_MODE)
            st = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as e:
        logger.debug(f"Could not give {path} to group qemu-vm: {e}")
        return False
    return st.st_gid == gid and stat.S_IMODE(st.st_mode) == QEMU_VM_DIR_MODE


async def grant_qemu_vm_private_file_access(path: Path) -> bool:
    """Grant qemu-vm read+write on a private (0600) file by ACL, plus parent traversal.

    The host keeps owning the file (it can still read, copy and delete it
    without sudo); other accounts get nothing. Needs setfacl.

    Returns:
        True if access was granted, False if any step failed
    """
    return await grant_qemu_vm_acl([path], perms="rw")


async def grant_qemu_vm_acl(paths: Sequence[Path], *, perms: str) -> bool:
    """Give qemu-vm exactly `perms` ("r", "rw") on files this process owns, by ACL, and make
    the folders above them traversable (ensure_traversable).

    The host keeps owning each file; other accounts get nothing. One setfacl for the
    files whose ACL does not give qemu-vm `perms` already (an in-process xattr read
    each), none when all do. "r" after "rw" removes the write right.

    Returns:
        True if every file has the entry, False if a step failed (no setfacl: package acl)
    """
    for folder_member in {path.parent: path for path in paths}.values():
        if not await ensure_traversable(folder_member):
            return False
    wanted = sum(_ACL_PERM_BITS[p] for p in perms)
    missing = [path for path in paths if qemu_vm_acl_perms(path) != wanted]
    return not missing or await set_acl_user(missing, user="qemu-vm", perms=perms)


def qemu_vm_acl_perms(path: Path) -> int | None:
    """What the ACL gives qemu-vm on `path` (bits r=4 w=2 x=1, within the mask), or None:
    no u:qemu-vm entry, no ACL, or no POSIX ACLs here (system_probes.posix_acl_xattr: a
    capability of the OS and of the filesystem). One xattr read (acl(5))."""
    uid = get_qemu_vm_uid()
    if uid is None or (raw := posix_acl_xattr(path)) is None:
        return None
    if len(raw) < 4 or struct.unpack_from("<I", raw)[0] != _ACL_XATTR_VERSION:  # noqa: PLR2004
        return None
    user, mask = None, 0o7
    for offset in range(4, len(raw) - 7, 8):
        tag, perm, entry_id = struct.unpack_from("<HHI", raw, offset)
        if tag == _ACL_USER and entry_id == uid:
            user = perm
        elif tag == _ACL_MASK:
            mask = perm
    return None if user is None else user & mask


async def ensure_traversable(path: Path) -> bool:
    """Make every folder above `path` traversable (a+x) for the qemu-vm user, innermost
    first; the filesystem root is left as it is.

    Checks the mode of each folder first and spawns `chmod a+x` only when an
    execute bit is missing (every VM start checks ~6 parents; most already pass).

    Args:
        path: File whose parent folders qemu-vm must traverse

    Returns:
        True if all folders were made traversable, False if any failed
    """
    all_x = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    success = True
    for dir_path in list(path.parents)[:-1]:
        mode = get_mode(dir_path)
        if mode is not None and mode & all_x == all_x:
            continue
        if not await chmod_async(dir_path, "a+x"):
            success = False
    return success
