"""Raw SQLite source snapshots and migration locking without opening source databases."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import secrets
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

SQLITE_SIGNATURE = b"SQLite format 3\x00"
SQLITE_HEADER_SIZE = 100
SQLITE_RESERVED_BYTE = 1_073_741_825
_COPY_CHUNK_BYTES = 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 5.0
_MAINTENANCE_PREFIX = ".invoice-db-maintenance-"
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_DARWIN_SYSTEM_ROOT_ALIASES = {
    "var": ("private", "var"),
    "tmp": ("private", "tmp"),
    "etc": ("private", "etc"),
}


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
    for index, part in enumerate(lexical.parts[1:]):
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
            if (
                sys.platform == "darwin"
                and index == 0
                and part in _DARWIN_SYSTEM_ROOT_ALIASES
                and os.readlink(current).lstrip("/") == f"private/{part}"
            ):
                continue
            raise _symlink_error(lexical)
    return lexical


@dataclass(frozen=True, slots=True)
class SQLiteDirectoryIdentity:
    """One retained no-follow directory component in lexical order."""

    name: str
    descriptor: int
    device: int
    inode: int
    mode: int


@dataclass(frozen=True, slots=True)
class SQLiteSystemRootAlias:
    """One explicitly allowed immutable macOS root alias such as /var."""

    lexical_path: Path
    device: int
    inode: int
    mode: int
    target: str
    target_component_index: int


@dataclass(slots=True)
class SQLiteRetainedPath:
    """A caller path bound to retained root-to-parent directory descriptors."""

    caller_path: Path
    root_descriptor: int
    root_device: int
    root_inode: int
    root_mode: int
    directories: tuple[SQLiteDirectoryIdentity, ...]
    leaf_name: str
    leaf_descriptor: int | None
    system_alias: SQLiteSystemRootAlias | None = None

    @property
    def parent_descriptor(self) -> int:
        return self.directories[-1].descriptor if self.directories else self.root_descriptor

    def assert_component_chain(self) -> None:
        """Require every lexical directory name to retain its captured inode."""

        root = os.fstat(self.root_descriptor)
        if (
            root.st_dev,
            root.st_ino,
            root.st_mode,
        ) != (self.root_device, self.root_inode, self.root_mode):
            raise _changed_error(self.caller_path)
        parent_descriptor = self.root_descriptor
        for component in self.directories:
            try:
                named = os.stat(
                    component.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                retained = os.fstat(component.descriptor)
            except OSError as exc:
                raise _changed_error(self.caller_path) from exc
            expected = (component.device, component.inode, component.mode)
            if (
                (named.st_dev, named.st_ino, named.st_mode) != expected
                or (retained.st_dev, retained.st_ino, retained.st_mode) != expected
                or not stat.S_ISDIR(named.st_mode)
            ):
                raise _changed_error(self.caller_path)
            parent_descriptor = component.descriptor
        if self.system_alias is not None:
            alias = self.system_alias
            try:
                lexical_alias = os.lstat(alias.lexical_path)
                alias_target = os.readlink(alias.lexical_path)
                followed = os.stat(alias.lexical_path)
                retained_target = os.fstat(
                    self.directories[alias.target_component_index].descriptor
                )
            except OSError as exc:
                raise _changed_error(self.caller_path) from exc
            if (
                (lexical_alias.st_dev, lexical_alias.st_ino, lexical_alias.st_mode)
                != (alias.device, alias.inode, alias.mode)
                or not stat.S_ISLNK(lexical_alias.st_mode)
                or alias_target != alias.target
                or (followed.st_dev, followed.st_ino)
                != (retained_target.st_dev, retained_target.st_ino)
            ):
                raise _changed_error(self.caller_path)

    def assert_leaf_binding(self, *, missing_ok: bool = False) -> None:
        """Require the leaf name to retain the held descriptor, or remain absent."""

        self.assert_component_chain()
        try:
            named = os.stat(
                self.leaf_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            if self.leaf_descriptor is None and missing_ok:
                return
            raise _changed_error(self.caller_path) from exc
        except OSError as exc:
            raise _changed_error(self.caller_path) from exc
        if self.leaf_descriptor is None:
            raise _changed_error(self.caller_path)
        retained = os.fstat(self.leaf_descriptor)
        if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino, named.st_mode) != (
            retained.st_dev,
            retained.st_ino,
            retained.st_mode,
        ):
            raise _changed_error(self.caller_path)

    def create_leaf(self, flags: int, mode: int = 0o600) -> int:
        """Create one absent regular leaf relative to its retained parent."""

        self.assert_leaf_binding(missing_ok=True)
        if self.leaf_descriptor is not None:
            return self.leaf_descriptor
        try:
            descriptor = os.open(
                self.leaf_name,
                flags | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=self.parent_descriptor,
            )
        except OSError as exc:
            raise _changed_error(self.caller_path) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise _changed_error(self.caller_path)
        self.leaf_descriptor = descriptor
        self.assert_leaf_binding()
        return descriptor


def _physical_parent_parts(
    lexical: Path,
) -> tuple[tuple[str, ...], SQLiteSystemRootAlias | None]:
    parent_parts = tuple(lexical.parts[1:-1])
    if sys.platform != "darwin" or not parent_parts:
        return parent_parts, None
    physical_alias = _DARWIN_SYSTEM_ROOT_ALIASES.get(parent_parts[0])
    if physical_alias is None:
        return parent_parts, None
    alias_path = Path(lexical.anchor) / parent_parts[0]
    alias_metadata = os.lstat(alias_path)
    if not stat.S_ISLNK(alias_metadata.st_mode):
        return parent_parts, None
    target = os.readlink(alias_path)
    expected_target = f"private/{parent_parts[0]}"
    if target.lstrip("/") != expected_target:
        raise _symlink_error(lexical)
    physical_parts = (*physical_alias, *parent_parts[1:])
    return physical_parts, SQLiteSystemRootAlias(
        lexical_path=alias_path,
        device=alias_metadata.st_dev,
        inode=alias_metadata.st_ino,
        mode=alias_metadata.st_mode,
        target=target,
        target_component_index=len(physical_alias) - 1,
    )


@contextmanager
def retained_lexical_database_path(
    path: Path,
    *,
    leaf_flags: int = os.O_RDONLY,
    missing_ok: bool = False,
) -> Iterator[SQLiteRetainedPath]:
    """Retain an ordered no-follow root-to-parent chain and open the leaf by dir_fd."""

    lexical = lexical_absolute_path(path)
    root_descriptor = os.open(lexical.anchor, _DIRECTORY_OPEN_FLAGS)
    root = os.fstat(root_descriptor)
    directory_descriptors: list[int] = []
    leaf_descriptor: int | None = None
    retained: SQLiteRetainedPath | None = None
    try:
        physical_parts, system_alias = _physical_parent_parts(lexical)
        identities: list[SQLiteDirectoryIdentity] = []
        parent_descriptor = root_descriptor
        for component in physical_parts:
            try:
                descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                try:
                    metadata = os.stat(
                        component,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    metadata = None
                if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                    raise _symlink_error(lexical) from exc
                if exc.errno == errno.ENOENT:
                    raise _missing_error(lexical) from exc
                raise _changed_error(lexical) from exc
            directory_descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise _changed_error(lexical)
            identities.append(
                SQLiteDirectoryIdentity(
                    name=component,
                    descriptor=descriptor,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    mode=metadata.st_mode,
                )
            )
            parent_descriptor = descriptor
        leaf_name = lexical.name
        try:
            leaf_descriptor = os.open(
                leaf_name,
                leaf_flags | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT and missing_ok:
                leaf_descriptor = None
            else:
                try:
                    metadata = os.stat(
                        leaf_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    metadata = None
                if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                    raise _symlink_error(lexical) from exc
                if exc.errno == errno.ENOENT:
                    raise _missing_error(lexical) from exc
                raise _changed_error(lexical) from exc
        retained = SQLiteRetainedPath(
            caller_path=lexical,
            root_descriptor=root_descriptor,
            root_device=root.st_dev,
            root_inode=root.st_ino,
            root_mode=root.st_mode,
            directories=tuple(identities),
            leaf_name=leaf_name,
            leaf_descriptor=leaf_descriptor,
            system_alias=system_alias,
        )
        retained.assert_leaf_binding(missing_ok=missing_ok)
        yield retained
    finally:
        final_leaf_descriptor = (
            retained.leaf_descriptor if retained is not None else leaf_descriptor
        )
        if final_leaf_descriptor is not None:
            os.close(final_leaf_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)
        os.close(root_descriptor)


@dataclass(frozen=True, slots=True)
class SQLiteSourceRole:
    """Role-specific error contract for one authoritative database source."""

    key: str
    label: str
    wal_stop_reason: str


@dataclass(frozen=True, slots=True)
class HeaderInfo:
    """Validated SQLite header facts that constrain the later schema audit."""

    page_size: int
    write_version: int
    read_version: int
    schema_format: int
    text_encoding: int

    @property
    def is_pre_schema(self) -> bool:
        return (self.schema_format, self.text_encoding) == (0, 0)


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
    retained_path: SQLiteRetainedPath
    maintenance_name: str
    maintenance_descriptor: int
    sqlite_name: str


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

    def assert_no_sidecars(self, path: Path) -> None:
        """Recheck retained directory entries immediately before SQLite writes."""

        binding = self._binding(path)
        _assert_maintenance_binding(binding)
        sidecars = _sidecar_identities(binding.retained_path)
        if sidecars:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                f"database has unsupported SQLite sidecars: "
                f"{[sidecar.name for sidecar in sidecars]}",
                stop_reason="DATABASE_SIDECAR_UNSUPPORTED",
            )

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
        identity, header = _read_identity_from_descriptor(binding.retained_path)
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


def validate_complete_sqlite_header(path: Path, header: bytes, file_size: int) -> HeaderInfo:
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
    return HeaderInfo(
        page_size=page_size,
        write_version=header[18],
        read_version=header[19],
        schema_format=schema_format,
        text_encoding=text_encoding,
    )


def _sidecar_name(database_name: str, candidate: str) -> bool:
    return candidate in {
        f"{database_name}-journal",
        f"{database_name}-wal",
        f"{database_name}-shm",
    } or candidate.startswith(f"{database_name}-mj")


def _hash_regular_file_at(
    retained: SQLiteRetainedPath,
    name: str,
) -> tuple[os.stat_result, str, bytes]:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=retained.parent_descriptor)
    except OSError as exc:
        raise _changed_error(retained.caller_path) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _changed_error(retained.caller_path)
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
            raise _changed_error(retained.caller_path)
        return after, digest.hexdigest(), bytes(header)
    finally:
        os.close(descriptor)


def _read_identity_from_descriptor(
    retained: SQLiteRetainedPath,
) -> tuple[SQLiteSourceIdentity, bytes]:
    path = retained.caller_path
    descriptor = retained.leaf_descriptor
    if descriptor is None:
        raise _missing_error(path)
    retained.assert_leaf_binding()
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
        sidecars=_sidecar_identities(retained),
    )
    return identity, bytes(header)


def _sidecar_identities(
    retained: SQLiteRetainedPath,
) -> tuple[SQLiteSidecarIdentity, ...]:
    retained.assert_component_chain()
    try:
        names = sorted(
            name
            for name in os.listdir(retained.parent_descriptor)
            if _sidecar_name(retained.leaf_name, name)
        )
    except OSError as exc:
        raise _changed_error(retained.caller_path) from exc
    identities: list[SQLiteSidecarIdentity] = []
    for name in names:
        metadata, digest, _header = _hash_regular_file_at(retained, name)
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

    with retained_lexical_database_path(path) as retained:
        identity, header = _read_identity_from_descriptor(retained)
        _validate_source_contract(identity, header, role)
        retained.assert_leaf_binding()
        return identity


def assert_no_sqlite_sidecars(path: Path, *, missing_ok: bool = False) -> None:
    """Enumerate retained parent entries without creating or opening a missing leaf."""

    with retained_lexical_database_path(path, missing_ok=missing_ok) as retained:
        sidecars = _sidecar_identities(retained)
        if sidecars:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                f"database has unsupported SQLite sidecars: "
                f"{[sidecar.name for sidecar in sidecars]}",
                stop_reason="DATABASE_SIDECAR_UNSUPPORTED",
            )


def _copy_validated_source(
    retained: SQLiteRetainedPath,
    destination: Path,
    role: SQLiteSourceRole,
) -> SQLiteSourceIdentity:
    lexical = retained.caller_path
    source_descriptor = retained.leaf_descriptor
    if source_descriptor is None:
        raise _missing_error(lexical)
    retained.assert_leaf_binding()
    sidecars_before = _sidecar_identities(retained)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    before = os.fstat(source_descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise _changed_error(lexical)
    destination_descriptor = os.open(destination, destination_flags, 0o600)
    try:
        digest = hashlib.sha256()
        header = bytearray()
        copied = 0
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                source_descriptor,
                min(_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise _changed_error(lexical)
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
            offset += len(chunk)
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
    after = os.fstat(source_descriptor)
    sidecars_after = _sidecar_identities(retained)
    retained.assert_leaf_binding()
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


def assert_source_identity_unchanged(
    identity: SQLiteSourceIdentity,
    retained: SQLiteRetainedPath | None = None,
) -> None:
    """Re-stat and rehash an original source, translating every drift to one code."""

    try:
        if retained is None:
            with retained_lexical_database_path(identity.resolved_path) as fresh:
                current, _header = _read_identity_from_descriptor(fresh)
                fresh.assert_leaf_binding()
        else:
            current, _header = _read_identity_from_descriptor(retained)
            retained.assert_leaf_binding()
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
        # The source chain handles the explicit macOS /var alias; no caller path
        # is canonicalized through resolve(), which would erase its lexical binding.
        root = lexical_absolute_path(Path(temporary_directory))
        with ExitStack() as retained_sources:
            snapshots: dict[str, SQLiteSourceSnapshot] = {}
            bindings: dict[str, SQLiteRetainedPath] = {}
            for index, (path, role) in enumerate(sources):
                retained = retained_sources.enter_context(retained_lexical_database_path(path))
                copy_path = root / f"snapshot-{index}.db"
                identity = _copy_validated_source(retained, copy_path, role)
                snapshots[role.key] = SQLiteSourceSnapshot(role, identity, copy_path)
                bindings[role.key] = retained
            for key, snapshot in snapshots.items():
                assert_source_identity_unchanged(snapshot.identity, bindings[key])
            try:
                yield snapshots
            finally:
                for key, snapshot in snapshots.items():
                    assert_source_identity_unchanged(snapshot.identity, bindings[key])


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


def _assert_maintenance_binding(binding: SQLiteMaintenanceBinding) -> None:
    """Require both names to remain regular hardlinks to the held descriptor."""

    try:
        binding.retained_path.assert_leaf_binding()
        locked = os.fstat(binding.descriptor)
        caller = os.stat(
            binding.retained_path.leaf_name,
            dir_fd=binding.retained_path.parent_descriptor,
            follow_symlinks=False,
        )
        maintenance = os.stat(
            binding.maintenance_name,
            dir_fd=binding.retained_path.parent_descriptor,
            follow_symlinks=False,
        )
        retained_maintenance = os.fstat(binding.maintenance_descriptor)
        private = os.stat(
            binding.sqlite_name,
            dir_fd=binding.maintenance_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise _changed_error(binding.caller_path) from exc
    expected = (locked.st_dev, locked.st_ino)
    if (
        not stat.S_ISREG(locked.st_mode)
        or not stat.S_ISREG(caller.st_mode)
        or not stat.S_ISREG(private.st_mode)
        or not stat.S_ISDIR(maintenance.st_mode)
        or (maintenance.st_dev, maintenance.st_ino, maintenance.st_mode)
        != (
            retained_maintenance.st_dev,
            retained_maintenance.st_ino,
            retained_maintenance.st_mode,
        )
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
        metadata = os.stat(
            binding.sqlite_name,
            dir_fd=binding.maintenance_descriptor,
            follow_symlinks=False,
        )
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


def _cleanup_maintenance_directory(binding: SQLiteMaintenanceBinding) -> None:
    try:
        entries = os.listdir(binding.maintenance_descriptor)
    except FileNotFoundError:
        return
    for entry in entries:
        metadata = os.stat(
            entry,
            dir_fd=binding.maintenance_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(metadata.st_mode):
            raise OSError(f"unexpected directory in SQLite maintenance directory: {entry}")
        os.unlink(entry, dir_fd=binding.maintenance_descriptor)
    os.fsync(binding.maintenance_descriptor)
    os.rmdir(
        binding.maintenance_name,
        dir_fd=binding.retained_path.parent_descriptor,
    )
    os.fsync(binding.retained_path.parent_descriptor)


def _create_maintenance_directory(
    retained: SQLiteRetainedPath,
) -> tuple[str, int, Path]:
    for _attempt in range(100):
        name = f"{_MAINTENANCE_PREFIX}{secrets.token_hex(8)}"
        try:
            os.mkdir(name, 0o700, dir_fd=retained.parent_descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                f"could not create private SQLite maintenance directory: {retained.caller_path}",
                stop_reason="DATABASE_MAINTENANCE_BINDING_FAILED",
            ) from exc
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=retained.parent_descriptor,
            )
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise OSError("private SQLite maintenance directory has invalid permissions")
            retained.assert_component_chain()
            return name, descriptor, retained.caller_path.parent / name
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                os.rmdir(name, dir_fd=retained.parent_descriptor)
            raise
    raise DatabaseVerificationError(
        ErrorCategory.DATABASE,
        f"could not allocate private SQLite maintenance directory: {retained.caller_path}",
        stop_reason="DATABASE_MAINTENANCE_BINDING_FAILED",
    )


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
    with _PRODUCTION_CONNECTIONS.maintenance(), ExitStack() as retained_paths:
        bindings: list[SQLiteMaintenanceBinding] = []
        primary_error: BaseException | None = None
        try:
            for path in unique_paths:
                retained = retained_paths.enter_context(
                    retained_lexical_database_path(
                        path,
                        leaf_flags=os.O_RDWR,
                        missing_ok=path in allowed_creations,
                    )
                )
                sidecars = _sidecar_identities(retained)
                if sidecars:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"database has unsupported SQLite sidecars: "
                        f"{[sidecar.name for sidecar in sidecars]}",
                        stop_reason="DATABASE_SIDECAR_UNSUPPORTED",
                    )
                if retained.leaf_descriptor is None:
                    if path not in allowed_creations:
                        raise _missing_error(path)
                    try:
                        descriptor = retained.create_leaf(os.O_RDWR)
                    except DatabaseVerificationError as exc:
                        raise DatabaseVerificationError(
                            ErrorCategory.DATABASE,
                            f"could not create database for locked migration: {path}",
                            stop_reason="DATABASE_LOCK_UNAVAILABLE",
                        ) from exc
                else:
                    descriptor = retained.leaf_descriptor
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise _changed_error(path)
                _lock_reserved_byte(descriptor, path)
                retained.assert_leaf_binding()
                if _sidecar_identities(retained):
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        "database gained unsupported SQLite sidecars during migration setup",
                        stop_reason="DATABASE_SIDECAR_UNSUPPORTED",
                    )
                maintenance_name, maintenance_descriptor, maintenance_directory = (
                    _create_maintenance_directory(retained)
                )
                sqlite_name = f"source-{len(bindings)}.db"
                sqlite_path = maintenance_directory / sqlite_name
                try:
                    os.link(
                        retained.leaf_name,
                        sqlite_name,
                        src_dir_fd=retained.parent_descriptor,
                        dst_dir_fd=maintenance_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    os.close(maintenance_descriptor)
                    with suppress(OSError):
                        os.rmdir(maintenance_name, dir_fd=retained.parent_descriptor)
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"could not bind SQLite migration to the locked database inode: {path}",
                        stop_reason="DATABASE_MAINTENANCE_BINDING_FAILED",
                    ) from exc
                binding = SQLiteMaintenanceBinding(
                    path,
                    descriptor,
                    sqlite_path,
                    retained,
                    maintenance_name,
                    maintenance_descriptor,
                    sqlite_name,
                )
                bindings.append(binding)
                _assert_maintenance_binding(binding)
                os.fsync(maintenance_descriptor)
                os.fsync(retained.parent_descriptor)
            yield SQLiteMaintenanceLocks(tuple(bindings))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            for binding in reversed(bindings):
                try:
                    _cleanup_maintenance_directory(binding)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                finally:
                    os.close(binding.maintenance_descriptor)
            for binding in reversed(bindings):
                with suppress(OSError):
                    fcntl.lockf(
                        binding.descriptor,
                        fcntl.LOCK_UN,
                        1,
                        SQLITE_RESERVED_BYTE,
                        os.SEEK_SET,
                    )
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
