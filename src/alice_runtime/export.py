"""Create-only transcript export through pinned Linux directory descriptors.

No existing destination inode is opened, truncated, replaced or removed. The
guarantee covers these export operations and path substitution while they run;
it cannot prevent a same-UID process from independently modifying protected
files or moving an already exported file. Linux local filesystems, O_TMPFILE,
hard-link publication, directory fsync and /proc/self/fd are required.
Writing through subtree mounts (including subtree bind mounts or subvolumes)
is refused; existing-file protection remains create-only on every mount.
The unprivileged O_TMPFILE publication route is documented by Linux open(2)
and link(2): https://man7.org/linux/man-pages/man2/open.2.html and
https://man7.org/linux/man-pages/man2/link.2.html.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import threading


class TranscriptExportError(ValueError):
    """Unsafe, unavailable or failed transcript export."""


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_RESERVED_NAMES = frozenset({
    "memory.json", "bridge.json", "config.json", "tokenizer_config.json",
    "model.safetensors.index.json",
})


def _identity(descriptor):
    result = os.fstat(descriptor)
    if not stat.S_ISDIR(result.st_mode):
        raise TranscriptExportError("Expected a regular directory descriptor")
    return result.st_dev, result.st_ino


def _read_proc_metadata(path, limit):
    with open(path, "r", encoding="utf-8") as handle:
        result = handle.read(limit + 1)
    if len(result) > limit:
        raise TranscriptExportError("Mount metadata exceeds the supported bound")
    return result


def _mount_root(descriptor):
    """Resolve a pinned FD's mount ID to its root in this mount namespace."""
    _identity(descriptor)
    info = _read_proc_metadata(f"/proc/self/fdinfo/{descriptor}", 65536)
    ids = [line.split(":", 1)[1].strip() for line in info.splitlines()
           if line.startswith("mnt_id:")]
    if len(ids) != 1 or not ids[0].isdigit() or int(ids[0]) <= 0:
        raise TranscriptExportError("Cannot identify the export directory mount")
    matches = []
    for line in _read_proc_metadata("/proc/self/mountinfo", 4 * 2**20).splitlines():
        prefix, separator, _suffix = line.partition(" - ")
        fields = prefix.split()
        if fields and fields[0] == ids[0]:
            if not separator or len(fields) < 6:
                raise TranscriptExportError("Malformed export mount metadata")
            matches.append(fields[3])
    if len(matches) != 1:
        raise TranscriptExportError("Export mount is unavailable or ambiguous")
    return matches[0]


def assert_whole_filesystem_mount(directory_fd):
    """Fail closed on subtree mounts; protected read-only FD opens remain allowed.

    Bind mounting a protected subdirectory hides its actual parents from '..'.
    mountinfo's root field exposes that case. Unsupported or unavailable mount
    metadata is an error. Privileged mount-namespace changes are outside this
    process's guarantee; no check authorizes overwriting an existing inode.
    """
    try:
        if _mount_root(directory_fd) != "/":
            raise TranscriptExportError("Safe export does not support subtree/bind mounts or subvolumes")
    except (OSError, ValueError) as exc:
        raise TranscriptExportError(f"Cannot establish a safe writable mount: {exc}") from exc


def _ancestors(descriptor):
    """Identify actual directory ancestry, including aliases of the same inode."""
    current = os.dup(descriptor)
    seen = set()
    try:
        for _ in range(1024):
            identity = _identity(current)
            if identity in seen:
                return seen
            seen.add(identity)
            parent = os.open("..", _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = parent
        raise TranscriptExportError("Directory ancestry exceeds the supported bound")
    finally:
        os.close(current)


def assert_separate_directory(directory_fd, protected_fds, *, reject_ancestor=True):
    """Reject an equal/descendant protected inode; optionally reject ancestors.

    Descriptors remain owned by the caller. A fresh destination's parent may use
    reject_ancestor=False, then check the new child with the default strict mode.
    """
    ancestry = _ancestors(directory_fd)
    identity = _identity(directory_fd)
    for protected in protected_fds:
        if (_identity(protected) in ancestry
                or (reject_ancestor and identity in _ancestors(protected))):
            raise TranscriptExportError("Transcript export directory overlaps protected model, Bridge or memory state")


def _open_directory(path, *, create, forbidden_ids=frozenset(), guard_mounts=False):
    raw = Path(path).expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    if ".." in absolute.parts:
        raise TranscriptExportError("Directory paths must not contain '..' components")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        if guard_mounts:
            assert_whole_filesystem_mount(descriptor)
        for part in absolute.parts[1:]:
            if _identity(descriptor) in forbidden_ids:
                raise TranscriptExportError("Transcript export directory is inside protected state")
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                if guard_mounts:
                    assert_whole_filesystem_mount(descriptor)
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass  # A concurrent creator still has to pass O_NOFOLLOW.
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            try:
                if guard_mounts:
                    assert_whole_filesystem_mount(child)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def open_directory_fd(path, *, create=False):
    """Return an owned FD after opening every path component with O_NOFOLLOW."""
    try:
        return _open_directory(path, create=create)[0]
    except (OSError, ValueError) as exc:
        raise TranscriptExportError(f"Cannot open a safe local directory: {exc}") from exc


def _filename(name):
    text = os.fspath(name)
    if (not isinstance(text, str) or not text or text.startswith(".")
            or not text.endswith(".json") or len(text.encode("utf-8")) > 200
            or any(character in text for character in "/\\")
            or any(ord(character) < 32 for character in text)
            or text.casefold() in _RESERVED_NAMES):
        raise TranscriptExportError("Use one new transcript filename ending in .json, without paths or reserved state names")
    return text


class TranscriptExportDirectory:
    """One explicit export directory, pinned for its whole configured lifetime.

    Protected roots must exist and must be actual no-symlink paths. The runtime
    supplies its resolved model/Bridge/M roots after attachment. Destinations
    are new leaf .json files only. There is no unsafe replace/fallback mode.
    """

    def __init__(self, path, *, protected_paths=(), create=True):
        self._lock = threading.RLock()
        self._fd = None
        self._protected_fds = []
        try:
            for protected in protected_paths:
                self._protected_fds.append(open_directory_fd(protected))
            forbidden = frozenset(_identity(fd) for fd in self._protected_fds)
            self._fd, self.path = _open_directory(path, create=create, forbidden_ids=forbidden,
                                                guard_mounts=True)
            assert_separate_directory(self._fd, self._protected_fds)
        except BaseException as exc:
            self.close()
            if isinstance(exc, (OSError, ValueError)):
                raise TranscriptExportError(f"Cannot configure transcript export: {exc}") from exc
            raise

    def _require_current_directory(self):
        if self._fd is None:
            raise TranscriptExportError("Transcript export directory is closed")
        assert_whole_filesystem_mount(self._fd)
        descriptor, _path = _open_directory(self.path, create=False, guard_mounts=True)
        try:
            if _identity(descriptor) != _identity(self._fd):
                raise TranscriptExportError("Transcript export directory moved or changed; reopen its configuration")
        finally:
            os.close(descriptor)
        assert_separate_directory(self._fd, self._protected_fds)

    def write(self, name, content: bytes) -> Path:
        name = _filename(name)
        if not isinstance(content, bytes):
            raise TranscriptExportError("Transcript export content must be serialized bytes")
        with self._lock:
            descriptor = None
            published = False
            try:
                self._require_current_directory()
                flags = getattr(os, "O_TMPFILE", 0)
                if not flags:
                    raise TranscriptExportError("Safe transcript export requires Linux O_TMPFILE support")
                descriptor = os.open(".", os.O_WRONLY | os.O_CLOEXEC | flags, 0o600, dir_fd=self._fd)
                view = memoryview(content)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Transcript write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                # Source is the pinned anonymous inode, not a replaceable temp
                # pathname. linkat's no-replace behavior rejects every existing
                # destination, including a symlink or hardlink to protected data.
                os.link(f"/proc/self/fd/{descriptor}", name,
                        dst_dir_fd=self._fd, follow_symlinks=True)
                published = True
                os.fsync(self._fd)
                self._require_current_directory()
                return self.path / name
            except FileExistsError as exc:
                raise TranscriptExportError("Transcript destination already exists; choose a new filename") from exc
            except (OSError, ValueError) as exc:
                outcome = (" A complete export may already exist in the pinned directory; "
                           "inspect it before retrying." if published else " No export was published.")
                raise TranscriptExportError(f"Cannot safely export transcript: {exc}.{outcome}") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)  # Anonymous staging needs no pathname cleanup.

    def close(self):
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            for descriptor in self._protected_fds:
                os.close(descriptor)
            self._protected_fds.clear()

    def __enter__(self):
        with self._lock:
            self._require_current_directory()
        return self

    def __exit__(self, *_args):
        self.close()
