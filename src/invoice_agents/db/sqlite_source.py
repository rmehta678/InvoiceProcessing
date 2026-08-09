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
    sidecars: tuple[SQLiteSidecarIdentity, ...]


@dataclass(frozen=True, slots=True)
class SQLiteSourceSnapshot:
    role: SQLiteSourceRole
    identity: SQLiteSourceIdentity
    copy_path: Path


@dataclass(frozen=True, slots=True)
class SQLiteMaintenanceLocks:
    """Open raw descriptors whose SQLite RESERVED locks can be restored after open/ATTACH."""

    descriptors: tuple[tuple[Path, int], ...]

    def reacquire(self) -> None:
        for path, descriptor in self.descriptors:
            _lock_reserved_byte(descriptor, path)

    def validated_identity(
        self,
        path: Path,
        role: SQLiteSourceRole,
    ) -> SQLiteSourceIdentity:
        resolved = path.resolve()
        descriptor = next(
            (candidate for locked_path, candidate in self.descriptors if locked_path == resolved),
            None,
        )
        if descriptor is None:
            raise RuntimeError(f"database maintenance does not hold {resolved}")
        identity, header = _read_identity_from_descriptor(resolved, descriptor)
        _validate_source_contract(identity, header, role)
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
    if (
        len(header) < SQLITE_HEADER_SIZE
        or header[: len(SQLITE_SIGNATURE)] != SQLITE_SIGNATURE
        or page_size is None
        or header[18] not in (1, 2)
        or header[19] not in (1, 2)
        or header[21:24] != b"\x40\x20\x20"
        or header[44:48] not in tuple(value.to_bytes(4, "big") for value in range(1, 5))
        or header[56:60] not in tuple(value.to_bytes(4, "big") for value in range(1, 4))
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
            "rollback-journal mode is required",
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

    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"required database does not exist: {path}",
            stop_reason="DATABASE_MISSING",
        ) from exc
    identity, header = _read_identity(resolved)
    _validate_source_contract(identity, header, role)
    return identity


def _copy_validated_source(
    path: Path,
    destination: Path,
    role: SQLiteSourceRole,
) -> SQLiteSourceIdentity:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"required database does not exist: {path}",
            stop_reason="DATABASE_MISSING",
        ) from exc
    sidecars_before = _sidecar_identities(resolved)
    source_flags = os.O_RDONLY | os.O_NOFOLLOW
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        source_descriptor = os.open(resolved, source_flags)
    except OSError as exc:
        raise _changed_error(resolved) from exc
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _changed_error(resolved)
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
        sidecars_after = _sidecar_identities(resolved)
        if (
            _identity_stat(before) != _identity_stat(after)
            or copied != after.st_size
            or sidecars_before != sidecars_after
        ):
            raise _changed_error(resolved)
        identity = SQLiteSourceIdentity(
            resolved_path=resolved,
            device=after.st_dev,
            inode=after.st_ino,
            mode=after.st_mode,
            size=after.st_size,
            modified_ns=after.st_mtime_ns,
            sha256=digest.hexdigest(),
            sidecars=sidecars_after,
        )
        _validate_source_contract(identity, bytes(header), role)
        return identity
    finally:
        os.close(source_descriptor)


def assert_source_identity_unchanged(identity: SQLiteSourceIdentity) -> None:
    """Re-stat and rehash an original source, translating every drift to one code."""

    try:
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
        root = Path(temporary_directory)
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

    unique_paths = tuple(sorted({path.resolve() for path in paths}, key=str))
    allowed_creations = {path.resolve() for path in create_paths}
    with _PRODUCTION_CONNECTIONS.maintenance():
        descriptors: list[tuple[Path, int]] = []
        try:
            for path in unique_paths:
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
                _lock_reserved_byte(descriptor, path)
            yield SQLiteMaintenanceLocks(tuple(descriptors))
        finally:
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
