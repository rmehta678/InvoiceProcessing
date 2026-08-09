"""Raw SQLite source snapshots and migration locking without opening source databases."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

SQLITE_SIGNATURE = b"SQLite format 3\x00"
SQLITE_HEADER_SIZE = 100
SQLITE_RESERVED_BYTE = 1_073_741_825
_COPY_CHUNK_BYTES = 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 5.0
_MAINTENANCE_PREFIX = ".invoice-db-maintenance-"


def lexical_absolute_path(path: Path) -> Path:
    """Make a path absolute without resolving any symlink component."""

    return Path(os.path.abspath(os.fspath(path)))


def _missing_error(path: Path) -> DatabaseVerificationError:
    return DatabaseVerificationError(
        ErrorCategory.DATABASE,
        f"required database does not exist: {path}",
        stop_reason="DATABASE_MISSING",
    )


def _symlink_error(path: Path) -> DatabaseVerificationError:
    return DatabaseVerificationError(
        ErrorCategory.DATABASE,
        f"database paths may not contain symlink components: {path}",
        stop_reason="DATABASE_SYMLINK_UNSUPPORTED",
    )


def validate_lexical_database_path(
    path: Path,
    *,
    missing_ok: bool = False,
) -> Path:
    """Return an absolute lexical path after rejecting every symlink component."""

    lexical = lexical_absolute_path(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            if missing_ok:
                return lexical
            raise _missing_error(lexical) from exc
        except OSError as exc:
            raise _changed_error(lexical) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise _symlink_error(lexical)
    return lexical


@dataclass(frozen=True, slots=True)
class SQLiteSourceRole:
    """Role-specific error contract for one authoritative database source."""

    key: str
    label: str
    wal_stop_reason: str


@dataclass(frozen=True, slots=True)
class SQLiteSidecarIdentity:
    name: str
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SQLiteSourceIdentity:
    resolved_path: Path
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    sha256: str
    header: bytes
    sidecars: tuple[SQLiteSidecarIdentity, ...]


@dataclass(frozen=True, slots=True)
class SQLiteSourceSnapshot:
    role: SQLiteSourceRole
    identity: SQLiteSourceIdentity
    copy_path: Path


@dataclass(frozen=True, slots=True)
class SQLiteMaintenanceBinding:
    """One caller pathname, locked descriptor, and same-inode SQLite hardlink."""

    caller_path: Path
    descriptor: int
    sqlite_path: Path


@dataclass(frozen=True, slots=True)
class SQLiteMaintenanceLocks:
    """Raw locks plus verified same-inode paths used only by migration SQLite opens."""

    bindings: tuple[SQLiteMaintenanceBinding, ...]

    def _binding(self, path: Path) -> SQLiteMaintenanceBinding:
        lexical = lexical_absolute_path(path)
        binding = next(
            (candidate for candidate in self.bindings if candidate.caller_path == lexical),
            None,
        )
        if binding is None:
            raise RuntimeError(f"database maintenance does not hold {lexical}")
        return binding

    def reacquire(self) -> None:
        for binding in self.bindings:
            _lock_reserved_byte(binding.descriptor, binding.caller_path)

    def sqlite_path(self, path: Path) -> Path:
        """Return the verified private pathname for opening the locked inode through SQLite."""

        binding = self._binding(path)
        _assert_maintenance_binding(binding)
        return binding.sqlite_path

    def assert_binding(self, path: Path) -> None:
        """Require the caller and private pathname to still name the locked inode."""

        _assert_maintenance_binding(self._binding(path))

    def locked_size(self, path: Path) -> int:
        """Return the held inode size only after both path bindings are verified."""

        binding = self._binding(path)
        _assert_maintenance_binding(binding)
        return os.fstat(binding.descriptor).st_size

    def validated_identity(
        self,
        path: Path,
        role: SQLiteSourceRole,
    ) -> SQLiteSourceIdentity:
        binding = self._binding(path)
        _assert_maintenance_binding(binding)
        identity, header = _read_identity_from_descriptor(
            binding.caller_path,
            binding.descriptor,
        )
        _validate_source_contract(identity, header, role)
        _assert_hardlink_content_identity(binding, identity)
        return identity


class _ProductionConnectionCoordinator:
    """Let normal connections coexist while maintenance excludes every production open.

    POSIX record locks are process-associated: closing an unrelated descriptor for the
    same inode can release them.  Every production SQLite open therefore participates
    in this gate, while ordinary application concurrency remains available outside a
    maintenance interval.  Direct third-party ``sqlite3.connect`` calls in this process
    cannot participate and are caught by the in-lock journal/header recheck instead.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._active_connections = 0
        self._active_by_thread: dict[int, int] = {}
        self._maintenance_owner: int | None = None
        self._maintenance_depth = 0

    @contextmanager
    def connection(self) -> Iterator[None]:
        owner = threading.get_ident()
        with self._condition:
            while self._maintenance_owner not in (None, owner):
                self._condition.wait()
            self._active_connections += 1
            self._active_by_thread[owner] = self._active_by_thread.get(owner, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                self._active_connections -= 1
                remaining = self._active_by_thread[owner] - 1
                if remaining:
                    self._active_by_thread[owner] = remaining
                else:
                    del self._active_by_thread[owner]
                self._condition.notify_all()

    @contextmanager
    def maintenance(self) -> Iterator[None]:
        owner = threading.get_ident()
        with self._condition:
            if self._maintenance_owner == owner:
                self._maintenance_depth += 1
            else:
                if self._active_by_thread.get(owner, 0):
                    raise RuntimeError(
                        "database maintenance cannot begin inside a production connection"
                    )
                while self._maintenance_owner is not None or self._active_connections:
                    self._condition.wait()
                self._maintenance_owner = owner
                self._maintenance_depth = 1
        try:
            yield
        finally:
            with self._condition:
                self._maintenance_depth -= 1
                if not self._maintenance_depth:
                    self._maintenance_owner = None
                    self._condition.notify_all()


_PRODUCTION_CONNECTIONS = _ProductionConnectionCoordinator()


@contextmanager
def coordinated_production_connection() -> Iterator[None]:
    """Register one production ``connect_database`` lifetime."""

    with _PRODUCTION_CONNECTIONS.connection():
        yield


def _identity_stat(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _changed_error(path: Path) -> DatabaseVerificationError:
    return DatabaseVerificationError(
        ErrorCategory.DATABASE,
        f"database source changed during authoritative verification: {path}",
        stop_reason="DATABASE_CHANGED_DURING_VERIFICATION",
    )


def _signature_error(path: Path) -> DatabaseVerificationError:
    return DatabaseVerificationError(
        ErrorCategory.DATABASE,
        f"file is not a SQLite database with a complete valid header: {path}",
        stop_reason="DATABASE_SIGNATURE_INVALID",
    )


def _page_size(header: bytes) -> int | None:
    encoded = int.from_bytes(header[16:18], "big")
    if encoded == 1:
        return 65_536
    if encoded < 512 or encoded > 32_768 or encoded & (encoded - 1):
        return None
    return encoded


def validate_complete_sqlite_header(path: Path, header: bytes, file_size: int) -> None:
    """Validate fixed SQLite header invariants before interpreting journal bytes."""

    page_size = _page_size(header) if len(header) >= SQLITE_HEADER_SIZE else None
    schema_format = int.from_bytes(header[44:48], "big") if len(header) >= 48 else -1
    text_encoding = int.from_bytes(header[56:60], "big") if len(header) >= 60 else -1
    schema_encoding_pair_is_valid = (schema_format, text_encoding) == (0, 0) or (
        schema_format in range(1, 5) and text_encoding in range(1, 4)
    )
    if (
        len(header) < SQLITE_HEADER_SIZE
        or header[: len(SQLITE_SIGNATURE)] != SQLITE_SIGNATURE
        or page_size is None
        or header[18] not in (1, 2)
        or header[19] not in (1, 2)
        or header[21:24] != b"\x40\x20\x20"
        or not schema_encoding_pair_is_valid
        or any(header[72:92])
    ):
        raise _signature_error(path)
    assert page_size is not None
    reserved_bytes = header[20]
    if reserved_bytes >= page_size or page_size - reserved_bytes < 480:
        raise _signature_error(path)
    if file_size < page_size or file_size % page_size:
        raise _signature_error(path)
    actual_page_count = file_size // page_size
    change_counter = int.from_bytes(header[24:28], "big")
    declared_page_count = int.from_bytes(header[28:32], "big")
    version_valid_for = int.from_bytes(header[92:96], "big")
    if change_counter == version_valid_for and (
        declared_page_count == 0 or declared_page_count != actual_page_count
    ):
        raise _signature_error(path)
    first_freelist_page = int.from_bytes(header[32:36], "big")
    freelist_page_count = int.from_bytes(header[36:40], "big")
    largest_root_page = int.from_bytes(header[52:56], "big")
    if (
        first_freelist_page > actual_page_count
        or freelist_page_count > actual_page_count
        or largest_root_page > actual_page_count
    ):
        raise _signature_error(path)


def _sidecar_name(database_name: str, candidate: str) -> bool:
    return candidate in {
        f"{database_name}-journal",
        f"{database_name}-wal",
        f"{database_name}-shm",
    } or candidate.startswith(f"{database_name}-mj ")


def _hash_regular_file(path: Path) -> tuple[os.stat_result, str, bytes]:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _changed_error(path) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _changed_error(path)
        digest = hashlib.sha256()
        header = bytearray()
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            if len(header) < SQLITE_HEADER_SIZE:
                header.extend(chunk[: SQLITE_HEADER_SIZE - len(header)])
        after = os.fstat(descriptor)
        if _identity_stat(before) != _identity_stat(after):
            raise _changed_error(path)
        return after, digest.hexdigest(), bytes(header)
    finally:
        os.close(descriptor)


def _read_identity_from_descriptor(
    path: Path,
    descriptor: int,
) -> tuple[SQLiteSourceIdentity, bytes]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise _changed_error(path)
    digest = hashlib.sha256()
    header = bytearray()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(_COPY_CHUNK_BYTES, before.st_size - offset), offset)
        if not chunk:
            raise _changed_error(path)
        digest.update(chunk)
        if len(header) < SQLITE_HEADER_SIZE:
            header.extend(chunk[: SQLITE_HEADER_SIZE - len(header)])
        offset += len(chunk)
    after = os.fstat(descriptor)
    if _identity_stat(before) != _identity_stat(after):
        raise _changed_error(path)
    identity = SQLiteSourceIdentity(
        resolved_path=path,
        device=after.st_dev,
        inode=after.st_ino,
        mode=after.st_mode,
        size=after.st_size,
        modified_ns=after.st_mtime_ns,
        sha256=digest.hexdigest(),
        header=bytes(header),
        sidecars=_sidecar_identities(path),
    )
    return identity, bytes(header)


def _sidecar_identities(path: Path) -> tuple[SQLiteSidecarIdentity, ...]:
    try:
        with os.scandir(path.parent) as entries:
            names = sorted(entry.name for entry in entries if _sidecar_name(path.name, entry.name))
    except OSError as exc:
        raise _changed_error(path) from exc
    identities: list[SQLiteSidecarIdentity] = []
    for name in names:
        sidecar = path.parent / name
        metadata, digest, _header = _hash_regular_file(sidecar)
        identities.append(
            SQLiteSidecarIdentity(
                name=name,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                mode=metadata.st_mode,
                size=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
                sha256=digest,
            )
        )
    return tuple(identities)


def _read_identity(path: Path) -> tuple[SQLiteSourceIdentity, bytes]:
    metadata, digest, header = _hash_regular_file(path)
    sidecars = _sidecar_identities(path)
    return (
        SQLiteSourceIdentity(
            resolved_path=path,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            sha256=digest,
            header=header,
            sidecars=sidecars,
        ),
        header,
    )


def _validate_source_contract(
    identity: SQLiteSourceIdentity,
    header: bytes,
    role: SQLiteSourceRole,
) -> None:
    validate_complete_sqlite_header(identity.resolved_path, header, identity.size)
    if header[18] == 2 or header[19] == 2:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"{role.label} database uses WAL file-format header bytes; "
            "DELETE journal mode is required",
            stop_reason=role.wal_stop_reason,
        )
    if identity.sidecars:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"{role.label} database has unsupported SQLite sidecars: "
            f"{[sidecar.name for sidecar in identity.sidecars]}",
            stop_reason="DATABASE_SIDECAR_UNSUPPORTED",
        )


def read_validated_source_identity(path: Path, role: SQLiteSourceRole) -> SQLiteSourceIdentity:
    """Hash and validate one existing source without opening it through SQLite."""

    lexical = validate_lexical_database_path(path)
    identity, header = _read_identity(lexical)
    _validate_source_contract(identity, header, role)
    return identity


def _copy_validated_source(
    path: Path,
    destination: Path,
    role: SQLiteSourceRole,
) -> SQLiteSourceIdentity:
    lexical = validate_lexical_database_path(path)
    sidecars_before = _sidecar_identities(lexical)
    source_flags = os.O_RDONLY | os.O_NOFOLLOW
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        source_descriptor = os.open(lexical, source_flags)
    except OSError as exc:
        raise _changed_error(lexical) from exc
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _changed_error(lexical)
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        try:
            digest = hashlib.sha256()
            header = bytearray()
            copied = 0
            while True:
                chunk = os.read(source_descriptor, _COPY_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                if len(header) < SQLITE_HEADER_SIZE:
                    header.extend(chunk[: SQLITE_HEADER_SIZE - len(header)])
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise OSError("database snapshot copy made no progress")
                    copied += written
                    view = view[written:]
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        after = os.fstat(source_descriptor)
        sidecars_after = _sidecar_identities(lexical)
        if (
            _identity_stat(before) != _identity_stat(after)
            or copied != after.st_size
            or sidecars_before != sidecars_after
        ):
            raise _changed_error(lexical)
        identity = SQLiteSourceIdentity(
            resolved_path=lexical,
            device=after.st_dev,
            inode=after.st_ino,
            mode=after.st_mode,
            size=after.st_size,
            modified_ns=after.st_mtime_ns,
            sha256=digest.hexdigest(),
            header=bytes(header),
            sidecars=sidecars_after,
        )
        _validate_source_contract(identity, bytes(header), role)
        return identity
    finally:
        os.close(source_descriptor)


def assert_source_identity_unchanged(identity: SQLiteSourceIdentity) -> None:
    """Re-stat and rehash an original source, translating every drift to one code."""

    try:
        validate_lexical_database_path(identity.resolved_path)
        current, _header = _read_identity(identity.resolved_path)
    except DatabaseVerificationError as exc:
        raise _changed_error(identity.resolved_path) from exc
    if current != identity:
        raise _changed_error(identity.resolved_path)


@contextmanager
def authoritative_database_snapshots(
    sources: Sequence[tuple[Path, SQLiteSourceRole]],
) -> Iterator[dict[str, SQLiteSourceSnapshot]]:
    """Pin a batch of raw rollback files, audit copies, then compare every original."""

    with tempfile.TemporaryDirectory(prefix="invoice-db-verify-") as temporary_directory:
        # This directory is internally created and owned; canonicalize macOS's /var ->
        # /private/var temporary-root alias so caller-path symlink rejection remains strict.
        root = Path(temporary_directory).resolve(strict=True)
        snapshots: dict[str, SQLiteSourceSnapshot] = {}
        for index, (path, role) in enumerate(sources):
            copy_path = root / f"snapshot-{index}.db"
            identity = _copy_validated_source(path, copy_path, role)
            snapshots[role.key] = SQLiteSourceSnapshot(role, identity, copy_path)
        for snapshot in snapshots.values():
            assert_source_identity_unchanged(snapshot.identity)
        try:
            yield snapshots
        finally:
            for snapshot in snapshots.values():
                assert_source_identity_unchanged(snapshot.identity)


def _lock_reserved_byte(descriptor: int, path: Path) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.lockf(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
                1,
                SQLITE_RESERVED_BYTE,
                os.SEEK_SET,
            )
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"could not lock database for migration: {path}",
                    stop_reason="DATABASE_LOCK_UNAVAILABLE",
                ) from exc
            if time.monotonic() >= deadline:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"database remained busy during migration lock acquisition: {path}",
                    stop_reason="DATABASE_LOCK_UNAVAILABLE",
                ) from exc
            time.sleep(0.01)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_maintenance_binding(binding: SQLiteMaintenanceBinding) -> None:
    """Require both names to remain regular hardlinks to the held descriptor."""

    try:
        locked = os.fstat(binding.descriptor)
        caller = os.lstat(binding.caller_path)
        private = os.lstat(binding.sqlite_path)
    except OSError as exc:
        raise _changed_error(binding.caller_path) from exc
    expected = (locked.st_dev, locked.st_ino)
    if (
        not stat.S_ISREG(locked.st_mode)
        or not stat.S_ISREG(caller.st_mode)
        or not stat.S_ISREG(private.st_mode)
        or (caller.st_dev, caller.st_ino) != expected
        or (private.st_dev, private.st_ino) != expected
    ):
        raise _changed_error(binding.caller_path)


def _assert_hardlink_content_identity(
    binding: SQLiteMaintenanceBinding,
    locked_identity: SQLiteSourceIdentity,
) -> None:
    """Compare hardlink metadata without opening a descriptor that would drop POSIX locks."""

    try:
        metadata = os.lstat(binding.sqlite_path)
    except OSError as exc:
        raise _changed_error(binding.caller_path) from exc
    private_identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    expected_identity = (
        locked_identity.device,
        locked_identity.inode,
        locked_identity.mode,
        locked_identity.size,
        locked_identity.modified_ns,
    )
    if private_identity != expected_identity:
        raise _changed_error(binding.caller_path)


def _cleanup_maintenance_directory(directory: Path) -> None:
    try:
        with os.scandir(directory) as iterator:
            entries = list(iterator)
    except FileNotFoundError:
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                raise OSError(f"unexpected directory in SQLite maintenance directory: {entry.name}")
            os.unlink(entry.name, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(directory)
    os.rmdir(directory)
    _fsync_directory(directory.parent)


@contextmanager
def exclusive_database_maintenance(
    paths: Sequence[Path],
    *,
    create_paths: Sequence[Path] = (),
) -> Iterator[SQLiteMaintenanceLocks]:
    """Exclude production connections and hold SQLite-compatible RESERVED locks.

    The raw descriptors stay open through the caller's final source check, SQLite
    ``BEGIN IMMEDIATE`` transaction, post-lock audit, and commit.  SQLite's final
    commit may release this process-associated byte lock; no protected work remains
    after that commit.
    """

    unique_paths = tuple(sorted({lexical_absolute_path(path) for path in paths}, key=str))
    allowed_creations = {lexical_absolute_path(path) for path in create_paths}
    with _PRODUCTION_CONNECTIONS.maintenance():
        descriptors: list[tuple[Path, int]] = []
        bindings: list[SQLiteMaintenanceBinding] = []
        maintenance_directories: list[Path] = []
        primary_error: BaseException | None = None
        try:
            for path in unique_paths:
                validate_lexical_database_path(
                    path,
                    missing_ok=path in allowed_creations,
                )
                try:
                    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                except OSError as exc:
                    if exc.errno == errno.ENOENT and path in allowed_creations:
                        try:
                            descriptor = os.open(
                                path,
                                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600,
                            )
                        except OSError as creation_exc:
                            raise DatabaseVerificationError(
                                ErrorCategory.DATABASE,
                                f"could not create database for locked migration: {path}",
                                stop_reason="DATABASE_LOCK_UNAVAILABLE",
                            ) from creation_exc
                    else:
                        raise DatabaseVerificationError(
                            ErrorCategory.DATABASE,
                            f"could not open database for locked migration: {path}",
                            stop_reason="DATABASE_LOCK_UNAVAILABLE",
                        ) from exc
                descriptors.append((path, descriptor))
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise _changed_error(path)
                _lock_reserved_byte(descriptor, path)
                validate_lexical_database_path(path)
                maintenance_directory = Path(
                    tempfile.mkdtemp(prefix=_MAINTENANCE_PREFIX, dir=path.parent)
                )
                maintenance_directories.append(maintenance_directory)
                os.chmod(maintenance_directory, 0o700)
                if stat.S_IMODE(os.lstat(maintenance_directory).st_mode) != 0o700:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"SQLite maintenance directory is not private: {maintenance_directory}",
                        stop_reason="DATABASE_MAINTENANCE_BINDING_FAILED",
                    )
                sqlite_path = maintenance_directory / f"source-{len(bindings)}.db"
                try:
                    os.link(path, sqlite_path, follow_symlinks=False)
                except OSError as exc:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"could not bind SQLite migration to the locked database inode: {path}",
                        stop_reason="DATABASE_MAINTENANCE_BINDING_FAILED",
                    ) from exc
                binding = SQLiteMaintenanceBinding(path, descriptor, sqlite_path)
                _assert_maintenance_binding(binding)
                _fsync_directory(maintenance_directory)
                _fsync_directory(path.parent)
                bindings.append(binding)
            yield SQLiteMaintenanceLocks(tuple(bindings))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            for directory in reversed(maintenance_directories):
                try:
                    _cleanup_maintenance_directory(directory)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            for _path, descriptor in reversed(descriptors):
                try:
                    fcntl.lockf(
                        descriptor,
                        fcntl.LOCK_UN,
                        1,
                        SQLITE_RESERVED_BYTE,
                        os.SEEK_SET,
                    )
                finally:
                    os.close(descriptor)
            if cleanup_error is not None:
                if primary_error is not None:
                    primary_error.add_note(
                        f"SQLite maintenance cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        "SQLite maintenance cleanup failed",
                        stop_reason="DATABASE_MAINTENANCE_CLEANUP_FAILED",
                    ) from cleanup_error
