"""Transcript export cannot replace protected state, even through path races.

Every protected file and lock in these tests is a disposable temporary copy.
No Qwen weights, real memory or runtime checkpoint are opened or modified.
"""

from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import json
import os
from pathlib import Path
import threading

import pytest

from alice_runtime import export as module
from alice_runtime.export import (
    TranscriptExportDirectory, TranscriptExportError,
    assert_separate_directory, assert_whole_filesystem_mount, open_directory_fd,
)
from alice_runtime.sessions import SessionError, SessionStore


@pytest.fixture
def protected(tmp_path):
    roots = [tmp_path / name for name in ("model", "bridge", "memory")]
    for root in roots:
        root.mkdir()
    files = [roots[0] / "config.json", roots[0] / "model.safetensors",
             roots[1] / "bridge.json", roots[1] / "bridge.safetensors",
             roots[2] / "memory.json", roots[2] / "memory.safetensors",
             roots[2] / ".alice-memory.lock"]
    for index, file in enumerate(files):
        file.write_bytes(f"protected-state-{index}".encode())
    before = {path: (path.read_bytes(), path.stat().st_dev, path.stat().st_ino) for path in files}
    lock = os.open(files[-1], os.O_RDWR)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield roots, files, before
    assert_protected(before)
    other = os.open(files[-1], os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(other)
        os.close(lock)


def assert_protected(before):
    assert {path: (path.read_bytes(), path.stat().st_dev, path.stat().st_ino)
            for path in before} == before


def store_with(text="private transcript"):
    store = SessionStore()
    session = store.new_session("conversation")
    store.add_message(session, "user", text)
    return store


def test_public_save_requires_explicit_exporter_and_never_accepts_arbitrary_protected_path(protected):
    _roots, files, before = protected
    store = store_with()
    for target in files:
        with pytest.raises(SessionError, match="export is disabled"):
            store.save(target)
    assert_protected(before)


def test_new_allowed_transcript_restores_and_existing_file_is_never_replaced(tmp_path, protected):
    roots, _files, before = protected
    store = store_with()
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        saved = store.save("chat.json", exporter=exporter)
        assert saved == tmp_path / "exports" / "chat.json"
        assert saved.stat().st_mode & 0o777 == 0o600
        assert SessionStore.load(saved).messages("conversation") == store.messages("conversation")
        content, inode = saved.read_bytes(), saved.stat().st_ino
        with pytest.raises(SessionError, match="already exists"):
            store.save("chat.json", exporter=exporter)
        assert saved.read_bytes() == content and saved.stat().st_ino == inode
    assert_protected(before)


@pytest.mark.parametrize("name", [
    "../memory/memory.json", "/tmp/config.json", "sub/chat.json", "..", ".alice-memory.lock",
    "memory.json", "bridge.json", "config.json", "tokenizer_config.json", "model.safetensors.index.json",
])
def test_unsafe_or_reserved_leaf_names_are_refused(tmp_path, protected, name):
    roots, _files, _before = protected
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        with pytest.raises(SessionError, match="filename"):
            store_with().save(name, exporter=exporter)
        assert list(exporter.path.iterdir()) == []


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_existing_leaf_alias_cannot_modify_memory_or_lock(tmp_path, protected, alias):
    roots, files, before = protected
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        for index, file in enumerate(files):
            target = exporter.path / f"alias-{index}.json"
            if alias == "symlink":
                target.symlink_to(file)
            else:
                os.link(file, target)
            with pytest.raises(SessionError, match="already exists"):
                store_with().save(target.name, exporter=exporter)
    assert_protected(before)


def test_equal_child_ancestor_and_symlink_export_directories_are_rejected_before_creation(tmp_path, protected):
    roots, _files, before = protected
    for target in (roots[2], roots[2] / "new-export", tmp_path):
        with pytest.raises(TranscriptExportError, match="protected"):
            TranscriptExportDirectory(target, protected_paths=roots)
    assert not (roots[2] / "new-export").exists()
    alias = tmp_path / "alias"
    alias.symlink_to(roots[2], target_is_directory=True)
    for target in (alias, alias / "new-export"):
        with pytest.raises(TranscriptExportError):
            TranscriptExportDirectory(target, protected_paths=roots)
    assert not (roots[2] / "new-export").exists()
    assert_protected(before)


def test_directory_fd_alias_and_parent_components_are_not_a_string_path_bypass(protected):
    roots, _files, _before = protected
    descriptor = open_directory_fd(roots[2])
    duplicate = os.dup(descriptor)
    try:
        with pytest.raises(TranscriptExportError, match="protected"):
            assert_separate_directory(duplicate, [descriptor])
        with pytest.raises(TranscriptExportError, match="components"):
            open_directory_fd(roots[2] / ".." / "bridge")
    finally:
        os.close(duplicate)
        os.close(descriptor)


def test_path_swap_after_check_writes_only_pinned_directory_and_never_cleans_user_paths(
        tmp_path, protected, monkeypatch):
    roots, files, before = protected
    directory, moved = tmp_path / "exports", tmp_path / "original-exports"
    with TranscriptExportDirectory(directory, protected_paths=roots) as exporter:
        original_open = os.open
        swapped = False

        def swap_parent(path, flags, *args, **kwargs):
            nonlocal swapped
            if path == "." and not swapped:
                swapped = True
                directory.rename(moved)
                directory.symlink_to(roots[2], target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)

        def forbidden_cleanup(*_args, **_kwargs):
            raise AssertionError("Export must never clean up by a mutable filename")

        monkeypatch.setattr(module.os, "open", swap_parent)
        monkeypatch.setattr(module.os, "unlink", forbidden_cleanup)
        with pytest.raises(SessionError, match="may already exist"):
            store_with().save("chat.json", exporter=exporter)
        assert swapped and (moved / "chat.json").is_file()
        assert not (roots[2] / "chat.json").exists()
        assert json.loads((moved / "chat.json").read_text())["sessions"]
    assert_protected(before)


def test_leaf_hardlink_inserted_at_publication_wins_without_being_overwritten(tmp_path, protected, monkeypatch):
    roots, files, before = protected
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        original_link = os.link

        def insert_alias(source, name, **kwargs):
            original_link(files[-1], name, dst_dir_fd=kwargs["dst_dir_fd"])
            return original_link(source, name, **kwargs)

        monkeypatch.setattr(module.os, "link", insert_alias)
        with pytest.raises(SessionError, match="already exists"):
            store_with().save("raced.json", exporter=exporter)
        assert (exporter.path / "raced.json").stat().st_ino == files[-1].stat().st_ino
    assert_protected(before)


def test_parent_component_swapped_during_creation_is_not_followed(tmp_path, protected, monkeypatch):
    roots, _files, before = protected
    original_mkdir = os.mkdir

    def substitute_created_directory(name, *args, **kwargs):
        result = original_mkdir(name, *args, **kwargs)
        if name == "new-parent":
            (tmp_path / "new-parent").rename(tmp_path / "old-parent")
            (tmp_path / "new-parent").symlink_to(roots[2], target_is_directory=True)
        return result

    monkeypatch.setattr(module.os, "mkdir", substitute_created_directory)
    with pytest.raises(TranscriptExportError):
        TranscriptExportDirectory(tmp_path / "new-parent" / "exports", protected_paths=roots)
    assert not (roots[2] / "exports").exists()
    assert_protected(before)


def test_two_independent_exporters_racing_same_name_publish_only_one_complete_snapshot(tmp_path, protected):
    roots, _files, before = protected
    barrier = threading.Barrier(2)
    with (TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as first,
          TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as second):
        def export(exporter, text):
            barrier.wait(timeout=3)
            try:
                store_with(text).save("same.json", exporter=exporter)
                return "saved"
            except SessionError as exc:
                assert "already exists" in str(exc)
                return "refused"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(export, first, "first snapshot"),
                       pool.submit(export, second, "second snapshot")]
            assert sorted(result.result(timeout=5) for result in results) == ["refused", "saved"]
        content = json.loads((first.path / "same.json").read_text())
        assert content["sessions"][0]["messages"][0]["text"] in {"first snapshot", "second snapshot"}
        assert list(first.path.iterdir()) == [first.path / "same.json"]
    assert_protected(before)


def test_unsupported_tmpfile_or_failed_write_leaves_no_named_partial_export(tmp_path, protected, monkeypatch):
    roots, _files, before = protected
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        original_open = os.open

        def unavailable(path, flags, *args, **kwargs):
            if path == ".":
                raise OSError(errno.EOPNOTSUPP, "O_TMPFILE unavailable")
            return original_open(path, flags, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(module.os, "open", unavailable)
            with pytest.raises(SessionError, match="O_TMPFILE unavailable"):
                store_with().save("unsupported.json", exporter=exporter)
        with monkeypatch.context() as patch:
            patch.setattr(module.os, "write", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
            with pytest.raises(SessionError, match="disk full"):
                store_with().save("partial.json", exporter=exporter)
        assert list(exporter.path.iterdir()) == []
    assert_protected(before)


def test_post_publication_sync_error_preserves_complete_export_without_cleanup(tmp_path, protected, monkeypatch):
    roots, _files, before = protected
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        original_sync = os.fsync

        def fail_directory_sync(descriptor):
            if descriptor == exporter._fd:
                raise OSError("export directory sync failed")
            return original_sync(descriptor)

        monkeypatch.setattr(module.os, "fsync", fail_directory_sync)
        with pytest.raises(SessionError, match="may already exist"):
            store_with().save("complete.json", exporter=exporter)
        assert json.loads((exporter.path / "complete.json").read_text())["sessions"]
    assert_protected(before)


@pytest.mark.parametrize("root", ["/protected/subtree", "/btrfs-subvolume", r"/path\040with-space"])
def test_subtree_mount_metadata_is_refused_before_export_directory_creation(tmp_path, protected, monkeypatch, root):
    roots, _files, before = protected

    def metadata(path, _limit):
        if "/fdinfo/" in path:
            return "pos:\t0\nmnt_id:\t44\n"
        return f"44 1 8:1 {root} /alias rw - ext4 /dev/source rw\n"

    monkeypatch.setattr(module, "_read_proc_metadata", metadata)
    with pytest.raises(TranscriptExportError, match="subtree/bind"):
        TranscriptExportDirectory(tmp_path / "not-created", protected_paths=roots)
    assert not (tmp_path / "not-created").exists()
    assert_protected(before)


@pytest.mark.parametrize("fdinfo,mountinfo", [
    ("pos: 0\n", "44 1 8:1 / / rw - ext4 /dev/root rw\n"),
    ("mnt_id: nope\n", "44 1 8:1 / / rw - ext4 /dev/root rw\n"),
    ("mnt_id: 44\nmnt_id: 44\n", "44 1 8:1 / / rw - ext4 /dev/root rw\n"),
    ("mnt_id: 44\n", "45 1 8:1 / / rw - ext4 /dev/root rw\n"),
    ("mnt_id: 44\n", "44 malformed\n"),
    ("mnt_id: 44\n", "44 1 8:1 / / rw - ext4 /dev/root rw\n" * 2),
])
def test_unknown_or_malformed_mount_identity_fails_closed(tmp_path, monkeypatch, fdinfo, mountinfo):
    monkeypatch.setattr(module, "_read_proc_metadata",
                        lambda path, _limit: fdinfo if "/fdinfo/" in path else mountinfo)
    with pytest.raises(TranscriptExportError):
        TranscriptExportDirectory(tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_unreadable_mount_metadata_does_not_enable_an_unguarded_fallback(tmp_path, monkeypatch):
    def denied(*_args):
        raise PermissionError("mount metadata denied")

    monkeypatch.setattr(module, "_read_proc_metadata", denied)
    with pytest.raises(TranscriptExportError, match="mount metadata denied"):
        TranscriptExportDirectory(tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_mount_guard_checks_opened_child_before_creating_any_descendants(tmp_path, protected, monkeypatch):
    roots, _files, before = protected
    alias = tmp_path / "mount-alias"
    alias.mkdir()
    identity = (alias.stat().st_dev, alias.stat().st_ino)
    original = module._mount_root

    def subtree(descriptor):
        info = os.fstat(descriptor)
        return "/protected/subtree" if (info.st_dev, info.st_ino) == identity else original(descriptor)

    monkeypatch.setattr(module, "_mount_root", subtree)
    with pytest.raises(TranscriptExportError, match="subtree/bind"):
        TranscriptExportDirectory(alias / "not-created", protected_paths=roots)
    assert list(alias.iterdir()) == []
    assert_protected(before)


def test_current_pinned_mount_is_rechecked_before_writing_but_protected_reads_still_work(
        tmp_path, protected, monkeypatch):
    roots, _files, before = protected
    with TranscriptExportDirectory(tmp_path / "exports", protected_paths=roots) as exporter:
        monkeypatch.setattr(module, "_mount_root", lambda _descriptor: "/protected/subtree")
        descriptor = open_directory_fd(roots[2])  # Read-only source pinning is allowed.
        os.close(descriptor)
        with pytest.raises(SessionError, match="subtree/bind"):
            store_with().save("not-created.json", exporter=exporter)
        assert list(exporter.path.iterdir()) == []
    assert_protected(before)


def test_real_subtree_bind_alias_is_rejected_in_an_unprivileged_private_namespace(tmp_path):
    import shutil
    import subprocess
    import sys

    unshare, mount = shutil.which("unshare"), shutil.which("mount")
    if unshare is None or mount is None:
        pytest.skip("Unprivileged private mount-namespace tools are unavailable")
    source, alias = tmp_path / "protected-subtree", tmp_path / "alias"
    source.mkdir()
    alias.mkdir()
    script = r'''
import os, subprocess, sys
from pathlib import Path
from alice_runtime.export import TranscriptExportDirectory, TranscriptExportError
source, alias, mount = sys.argv[1:]
bound = subprocess.run([mount, "--bind", source, alias], capture_output=True, text=True)
if bound.returncode:
    print(bound.stderr, file=sys.stderr)
    raise SystemExit(77)
try:
    TranscriptExportDirectory(Path(alias) / "not-created")
except TranscriptExportError as exc:
    assert "subtree/bind" in str(exc), str(exc)
else:
    raise AssertionError("A real protected subtree bind alias was accepted")
assert not (Path(source) / "not-created").exists()
'''
    result = subprocess.run(
        [unshare, "--user", "--map-root-user", "--mount", "--propagation", "private",
         sys.executable, "-B", "-c", script, str(source), str(alias), mount],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    if (result.returncode == 77 or (result.returncode and result.stderr.startswith("unshare:")
                                  and "Operation not permitted" in result.stderr)):
        pytest.skip("Unprivileged private mount namespace/bind mount is denied by this host")
    assert result.returncode == 0, result.stderr
    assert list(source.iterdir()) == []
