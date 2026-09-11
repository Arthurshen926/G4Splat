#!/usr/bin/env python3
"""Run one training command under a fail-closed detached supervisor.

The launcher double-forks before starting the training process, so neither the
supervisor nor the trainer remains attached to the caller's terminal.  A run
directory lock prevents two supervisors from owning the same outputs.

Automatic restart is intentionally narrow.  A child is resumed only when all
of the following are true:

* its return code represents SIGHUP, SIGINT, or SIGTERM;
* both the interruption sidecar and checkpoint were atomically replaced by
  that child invocation;
* the sidecar describes the same signal and an advancing iteration; and
* the checkpoint is a structurally valid, non-symlink PyTorch ZIP archive.

All other failures, including ordinary exceptions, OOM exits, SIGKILL, and an
ambiguous/stale sidecar, are recorded in the atomic heartbeat without retry.
An explicitly configured external initial resume is consumed only when a new
supervisor contract is created.  An optional migration flag is injected into
that one child invocation; an exact same-implementation external resume needs
no migration flag.  Every validated restart and later supervisor launch
resumes only the rolling checkpoint inside the locked run directory without
the flag.
The script uses only the Python standard library and does not import the
trainer or torch.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import select
import shlex
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, BinaryIO
import uuid
import zipfile


SCHEMA_VERSION = 1
SAFE_SIGNALS = frozenset((signal.SIGHUP, signal.SIGINT, signal.SIGTERM))
CHILD_CONTRACT_ENVIRONMENT = "G4SPLAT_DETACHED_SUPERVISOR_CONTRACT"
V114_OPTICAL_OWNERSHIP_REPAIR_FLAG = (
    "--allow-v114-optical-ownership-repair-resume"
)
V115_PERSISTENT_OWNERSHIP_DEBT_FLAG = (
    "--allow-v115-persistent-ownership-debt-resume"
)
V116_SHARED_ENVELOPE_LOCALIZATION_FLAG = (
    "--allow-v116-shared-envelope-localization-resume"
)
SUPPORTED_ONE_SHOT_RESUME_FLAGS = frozenset(
    (
        V114_OPTICAL_OWNERSHIP_REPAIR_FLAG,
        V115_PERSISTENT_OWNERSHIP_DEBT_FLAG,
        V116_SHARED_ENVELOPE_LOCALIZATION_FLAG,
    )
)
CONTRACT_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PATH",
    "LD_LIBRARY_PATH",
)
MAX_SIDECAR_BYTES = 1024 * 1024
MAX_PICKLE_BYTES = 256 * 1024 * 1024
MAX_ZIP_MEMBERS = 2_000_000
CLOCK_SLOP_SECONDS = 5.0


class SupervisorError(RuntimeError):
    """A condition for which unattended training is unsafe."""


@dataclasses.dataclass(frozen=True)
class FileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileFingerprint":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
        )

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ValidatedInterruption:
    iteration: int
    checkpoint: Path
    sidecar_fingerprint: FileFingerprint
    checkpoint_fingerprint: FileFingerprint


@dataclasses.dataclass(frozen=True)
class SupervisorConfig:
    run_dir: Path
    working_directory: Path
    base_command: tuple[str, ...]
    sidecar: Path
    checkpoint: Path
    lock_file: Path
    heartbeat_file: Path
    contract_file: Path
    supervisor_log: Path
    training_log: Path
    resume_flag: str
    initial_resume_checkpoint: Path | None
    one_shot_resume_flags: tuple[str, ...]
    heartbeat_seconds: float
    restart_backoff_seconds: float
    max_safe_restarts: int
    minimum_checkpoint_bytes: int
    launch_timeout_seconds: float
    command_sha256: str


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Replace *path* atomically and durably with a mode-0600 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        data = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8") + b"\n"
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _regular_file_fingerprint(path: Path) -> FileFingerprint:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise SupervisorError(f"Required file is absent: {path}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise SupervisorError(f"Refusing symlink artifact: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise SupervisorError(f"Artifact is not a regular file: {path}")
    return FileFingerprint.from_stat(value)


def _optional_regular_file_fingerprint(
    path: Path,
) -> FileFingerprint | None:
    try:
        return _regular_file_fingerprint(path)
    except SupervisorError:
        if not path.exists() and not path.is_symlink():
            return None
        raise


def _read_small_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SupervisorError(f"Could not safely open {path}: {exc}") from exc
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise SupervisorError(f"Artifact is not a regular file: {path}")
        if value.st_size <= 0 or value.st_size > maximum_bytes:
            raise SupervisorError(
                f"Artifact size is unsafe for {path}: {value.st_size} bytes"
            )
        chunks: list[bytes] = []
        remaining = int(value.st_size)
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise SupervisorError(f"Artifact changed while reading: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if FileFingerprint.from_stat(os.fstat(descriptor)) != (
            FileFingerprint.from_stat(value)
        ):
            raise SupervisorError(f"Artifact changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_signal_from_returncode(returncode: int) -> int | None:
    """Return a safe signal number encoded by a POSIX/Python return code."""

    if returncode < 0:
        candidate = -int(returncode)
    elif returncode >= 128:
        candidate = int(returncode) - 128
    else:
        return None
    if candidate in SAFE_SIGNALS:
        return candidate
    return None


def _resume_positions(command: tuple[str, ...], resume_flag: str) -> list[int]:
    inline_prefix = resume_flag + "="
    if any(token.startswith(inline_prefix) for token in command):
        raise SupervisorError(
            f"Inline {resume_flag}=... syntax is unsupported; use two tokens"
        )
    return [index for index, token in enumerate(command) if token == resume_flag]


def _replace_resume_argument(
    command: tuple[str, ...], resume_flag: str, checkpoint: Path
) -> tuple[str, ...]:
    positions = _resume_positions(command, resume_flag)
    if len(positions) > 1:
        raise SupervisorError(f"Command contains duplicate {resume_flag} flags")
    result = list(command)
    if positions:
        position = positions[0]
        if position + 1 >= len(result):
            raise SupervisorError(f"Command has a dangling {resume_flag} flag")
        result[position + 1] = str(checkpoint)
    else:
        result.extend((resume_flag, str(checkpoint)))
    return tuple(result)


def _resume_argument_path(
    command: tuple[str, ...],
    resume_flag: str,
    *,
    working_directory: Path,
) -> Path | None:
    positions = _resume_positions(command, resume_flag)
    if len(positions) > 1:
        raise SupervisorError(f"Command contains duplicate {resume_flag} flags")
    if not positions:
        return None
    position = positions[0]
    if position + 1 >= len(command):
        raise SupervisorError(f"Command has a dangling {resume_flag} flag")
    candidate = Path(command[position + 1]).expanduser()
    if not candidate.is_absolute():
        candidate = working_directory / candidate
    return candidate.resolve()


def _initial_supervised_command(
    config: SupervisorConfig,
    *,
    initial_resume_available: bool,
) -> tuple[tuple[str, ...], bool]:
    """Select exactly one external first resume or an in-run continuation."""

    rolling_checkpoint = _optional_regular_file_fingerprint(config.checkpoint)
    initial_checkpoint = config.initial_resume_checkpoint
    if initial_checkpoint is None:
        return config.base_command, False
    if initial_resume_available:
        if rolling_checkpoint is not None:
            raise SupervisorError(
                "A fresh one-shot external resume requires an empty rolling "
                f"checkpoint path: {config.checkpoint}"
            )
        _validate_pytorch_zip(
            initial_checkpoint,
            minimum_checkpoint_bytes=config.minimum_checkpoint_bytes,
        )
        command = _replace_resume_argument(
            config.base_command,
            config.resume_flag,
            initial_checkpoint,
        )
        return command + config.one_shot_resume_flags, True
    if rolling_checkpoint is None:
        raise SupervisorError(
            "The one-shot external resume was already consumed, but no "
            f"rolling checkpoint exists for safe continuation: {config.checkpoint}"
        )
    _validate_pytorch_zip(
        config.checkpoint,
        minimum_checkpoint_bytes=config.minimum_checkpoint_bytes,
    )
    return (
        _replace_resume_argument(
            config.base_command,
            config.resume_flag,
            config.checkpoint,
        ),
        False,
    )


def _resolve_sidecar_checkpoint(
    raw_value: Any, *, working_directory: Path
) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise SupervisorError("Sidecar checkpoint must be a non-empty path")
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = working_directory / candidate
    return candidate.resolve()


def _stream_zip_member(handle: BinaryIO, expected_size: int) -> None:
    total = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise SupervisorError("ZIP member expanded beyond its declared size")
    if total != expected_size:
        raise SupervisorError(
            f"ZIP member size mismatch: read {total}, expected {expected_size}"
        )


def _validate_pytorch_zip(
    checkpoint: Path, *, minimum_checkpoint_bytes: int
) -> FileFingerprint:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(checkpoint, flags)
    except OSError as exc:
        raise SupervisorError(
            f"Could not safely open checkpoint {checkpoint}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SupervisorError(f"Checkpoint is not regular: {checkpoint}")
        if int(before.st_size) < int(minimum_checkpoint_bytes):
            raise SupervisorError(
                f"Checkpoint is only {before.st_size} bytes; expected at least "
                f"{minimum_checkpoint_bytes}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as checkpoint_handle:
            try:
                with zipfile.ZipFile(checkpoint_handle, "r") as archive:
                    members = archive.infolist()
                    if not members or len(members) > MAX_ZIP_MEMBERS:
                        raise SupervisorError(
                            f"Unsafe checkpoint ZIP member count: {len(members)}"
                        )
                    names: set[str] = set()
                    pickle_members: list[zipfile.ZipInfo] = []
                    version_members: list[zipfile.ZipInfo] = []
                    for member in members:
                        name = member.filename
                        pure = PurePosixPath(name)
                        if (
                            not name
                            or pure.is_absolute()
                            or ".." in pure.parts
                            or "" in pure.parts
                        ):
                            raise SupervisorError(
                                f"Unsafe checkpoint ZIP member: {name!r}"
                            )
                        if name in names:
                            raise SupervisorError(
                                f"Duplicate checkpoint ZIP member: {name!r}"
                            )
                        names.add(name)
                        if pure.name == "data.pkl":
                            pickle_members.append(member)
                        elif pure.name == "version":
                            version_members.append(member)
                    if len(pickle_members) != 1 or len(version_members) != 1:
                        raise SupervisorError(
                            "Checkpoint is not an unambiguous PyTorch ZIP archive"
                        )
                    pickle_member = pickle_members[0]
                    version_member = version_members[0]
                    if (
                        PurePosixPath(pickle_member.filename).parent
                        != PurePosixPath(version_member.filename).parent
                    ):
                        raise SupervisorError(
                            "PyTorch ZIP metadata members use different roots"
                        )
                    if (
                        pickle_member.file_size <= 0
                        or (
                            pickle_member.file_size > MAX_PICKLE_BYTES
                            and not (
                                pickle_member.compress_type == zipfile.ZIP_STORED
                                and pickle_member.file_size == pickle_member.compress_size
                                and pickle_member.file_size <= int(before.st_size)
                            )
                        )
                    ):
                        raise SupervisorError(
                            "Checkpoint data.pkl has an unsafe uncompressed size: "
                            f"{pickle_member.file_size}"
                        )
                    if version_member.file_size <= 0 or version_member.file_size > 64:
                        raise SupervisorError(
                            "Checkpoint version member has an unsafe size"
                        )
                    if pickle_member.flag_bits & 0x1 or version_member.flag_bits & 0x1:
                        raise SupervisorError("Encrypted checkpoint ZIP is unsupported")
                    # Plain-list candidate metadata can legitimately exceed
                    # 256 MiB. Uncompressed, physically file-bounded members
                    # are streamed in 1 MiB chunks, not loaded/unpickled. Keep
                    # the expansion cap for compressed metadata (ZIP bombs).
                    # Reading to EOF makes zipfile verify CRC for the two
                    # serialization metadata members without loading tensor
                    # storage (and therefore without risking GPU/host OOM).
                    with archive.open(pickle_member, "r") as pickle_handle:
                        _stream_zip_member(
                            pickle_handle, int(pickle_member.file_size)
                        )
                    with archive.open(version_member, "r") as version_handle:
                        _stream_zip_member(
                            version_handle, int(version_member.file_size)
                        )
            except (
                OSError,
                RuntimeError,
                zipfile.BadZipFile,
                zipfile.LargeZipFile,
            ) as exc:
                raise SupervisorError(
                    f"Checkpoint ZIP validation failed: {exc}"
                ) from exc
        after = os.fstat(descriptor)
        if FileFingerprint.from_stat(before) != FileFingerprint.from_stat(after):
            raise SupervisorError("Checkpoint changed during validation")
        return FileFingerprint.from_stat(after)
    finally:
        os.close(descriptor)


def _validate_interrupted_checkpoint(
    *,
    sidecar: Path,
    checkpoint: Path,
    working_directory: Path,
    expected_signal: int,
    child_started_at: float,
    baseline_sidecar: FileFingerprint | None,
    baseline_checkpoint: FileFingerprint | None,
    previous_iteration: int,
    minimum_checkpoint_bytes: int,
    now: float | None = None,
) -> ValidatedInterruption:
    """Validate that a child produced a new, resumable interruption pair."""

    if expected_signal not in SAFE_SIGNALS:
        raise SupervisorError(f"Signal is not safe to resume: {expected_signal}")
    checked_at = time.time() if now is None else float(now)
    sidecar_fingerprint = _regular_file_fingerprint(sidecar)
    if sidecar_fingerprint == baseline_sidecar:
        raise SupervisorError("Interruption sidecar was not replaced by this child")
    if (
        sidecar_fingerprint.mtime_ns
        < int((child_started_at - CLOCK_SLOP_SECONDS) * 1_000_000_000)
    ):
        raise SupervisorError("Interruption sidecar predates this child invocation")
    try:
        payload = json.loads(
            _read_small_regular_file(sidecar, MAX_SIDECAR_BYTES).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"Invalid interruption sidecar JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SupervisorError("Interruption sidecar must contain one JSON object")
    required_exact = {
        "status": "interrupted_checkpointed",
        "checkpoint_atomic": True,
        "resume_required": True,
        "signal_number": int(expected_signal),
        "signal": signal.Signals(expected_signal).name,
    }
    for key, expected in required_exact.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            raise SupervisorError(
                f"Interruption sidecar field {key!r} is not exactly {expected!r}"
            )
    iteration = payload.get("iteration")
    minimum_iteration = max(0, int(previous_iteration))
    if type(iteration) is not int or iteration <= minimum_iteration:
        raise SupervisorError(
            "Interrupted iteration must be an integer that advances beyond "
            f"{minimum_iteration}; got {iteration!r}"
        )
    requested_at = payload.get("requested_at_unix")
    if type(requested_at) not in (int, float):
        raise SupervisorError("Sidecar requested_at_unix must be numeric")
    requested_at = float(requested_at)
    if (
        requested_at < child_started_at - CLOCK_SLOP_SECONDS
        or requested_at > checked_at + CLOCK_SLOP_SECONDS
    ):
        raise SupervisorError(
            "Sidecar requested_at_unix is outside this child invocation"
        )
    sidecar_checkpoint = _resolve_sidecar_checkpoint(
        payload.get("checkpoint"), working_directory=working_directory
    )
    if sidecar_checkpoint != checkpoint:
        raise SupervisorError(
            "Sidecar checkpoint does not match the configured checkpoint: "
            f"{sidecar_checkpoint} != {checkpoint}"
        )
    checkpoint_fingerprint = _validate_pytorch_zip(
        checkpoint,
        minimum_checkpoint_bytes=minimum_checkpoint_bytes,
    )
    if checkpoint_fingerprint == baseline_checkpoint:
        raise SupervisorError("Checkpoint was not replaced by this child")
    if (
        checkpoint_fingerprint.mtime_ns
        < int((child_started_at - CLOCK_SLOP_SECONDS) * 1_000_000_000)
    ):
        raise SupervisorError("Checkpoint predates this child invocation")
    return ValidatedInterruption(
        iteration=int(iteration),
        checkpoint=checkpoint,
        sidecar_fingerprint=sidecar_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )


def _acquire_run_lock(path: Path, metadata: dict[str, Any]) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError(
                f"Another supervisor already owns run directory lock {path}"
            ) from exc
        data = _canonical_json_bytes(metadata) + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise SupervisorError(f"Could not write lock metadata: {path}")
            written += count
        os.fsync(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _release_run_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _contract(config: SupervisorConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(config.run_dir),
        "working_directory": str(config.working_directory),
        "base_command": list(config.base_command),
        "command_sha256": config.command_sha256,
        "sidecar": str(config.sidecar),
        "checkpoint": str(config.checkpoint),
        "lock_file": str(config.lock_file),
        "heartbeat_file": str(config.heartbeat_file),
        "contract_file": str(config.contract_file),
        "supervisor_log": str(config.supervisor_log),
        "training_log": str(config.training_log),
        "resume_flag": config.resume_flag,
        "initial_resume_checkpoint": (
            str(config.initial_resume_checkpoint)
            if config.initial_resume_checkpoint is not None
            else None
        ),
        "one_shot_resume_flags": list(config.one_shot_resume_flags),
        "heartbeat_seconds": config.heartbeat_seconds,
        "restart_backoff_seconds": config.restart_backoff_seconds,
        "max_safe_restarts": config.max_safe_restarts,
        "minimum_checkpoint_bytes": config.minimum_checkpoint_bytes,
        "environment": {
            name: os.environ.get(name) for name in CONTRACT_ENVIRONMENT_KEYS
        },
    }


def _read_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            _read_small_regular_file(path, MAX_SIDECAR_BYTES).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"Invalid supervisor contract JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SupervisorError("Supervisor contract must be a JSON object")
    return payload


def _establish_contract(config: SupervisorConfig) -> bool:
    """Establish the immutable contract and report first-launch ownership."""

    expected = _contract(config)
    if config.contract_file.exists() or config.contract_file.is_symlink():
        existing = _read_contract(config.contract_file)
        if existing != expected:
            raise SupervisorError(
                "Existing supervisor contract differs from this launch; use a "
                "fresh run directory instead of mutating a resumable run"
            )
        return False
    _atomic_write_json(config.contract_file, expected)
    return True


def _heartbeat(
    config: SupervisorConfig,
    *,
    state: str,
    training_pid: int | None,
    safe_restart_count: int,
    returncode: int | None = None,
    failure_kind: str | None = None,
    detail: str | None = None,
    validated_iteration: int | None = None,
) -> None:
    _atomic_write_json(
        config.heartbeat_file,
        {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at_unix": time.time(),
            "supervisor_pid": os.getpid(),
            "training_pid": training_pid,
            "run_dir": str(config.run_dir),
            "working_directory": str(config.working_directory),
            "command_sha256": config.command_sha256,
            "checkpoint": str(config.checkpoint),
            "sidecar": str(config.sidecar),
            "safe_restart_count": int(safe_restart_count),
            "max_safe_restarts": int(config.max_safe_restarts),
            "returncode": returncode,
            "failure_kind": failure_kind,
            "detail": detail,
            "validated_iteration": validated_iteration,
        },
    )


def _open_append_log(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode):
        os.close(descriptor)
        raise SupervisorError(f"Log path is not a regular file: {path}")
    return os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)


def _log(handle, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    handle.write(f"[{timestamp}] {message}\n")
    handle.flush()


def _send_handshake(descriptor: int, payload: dict[str, Any]) -> None:
    try:
        os.write(descriptor, _canonical_json_bytes(payload) + b"\n")
    except OSError:
        pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _forward_signal(process: subprocess.Popen[Any] | None, signum: int) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _failure_kind(returncode: int) -> str:
    if returncode in (137, -signal.SIGKILL):
        return "sigkill_or_oom"
    if returncode < 0:
        return "unsafe_signal"
    return "child_nonzero_exit"


def _wait_backoff(
    config: SupervisorConfig,
    *,
    stop_state: dict[str, Any],
    safe_restart_count: int,
    validated_iteration: int,
) -> bool:
    deadline = time.monotonic() + config.restart_backoff_seconds
    next_heartbeat = 0.0
    while time.monotonic() < deadline:
        if stop_state["signal"] is not None:
            return False
        if time.monotonic() >= next_heartbeat:
            _heartbeat(
                config,
                state="safe_restart_pending",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                validated_iteration=validated_iteration,
            )
            next_heartbeat = time.monotonic() + config.heartbeat_seconds
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return stop_state["signal"] is None


def _supervise(
    config: SupervisorConfig,
    *,
    initial_resume_available: bool,
    handshake_descriptor: int,
    handshake_state: dict[str, bool],
    supervisor_log_handle,
) -> int:
    stop_state: dict[str, Any] = {"signal": None, "child": None}

    def request_stop(signum, _frame) -> None:
        if stop_state["signal"] is None:
            stop_state["signal"] = int(signum)
        _forward_signal(stop_state["child"], int(signum))

    for signum in SAFE_SIGNALS:
        signal.signal(signum, request_stop)

    current_command, external_initial_resume = _initial_supervised_command(
        config,
        initial_resume_available=initial_resume_available,
    )
    safe_restart_count = 0
    previous_iteration = -1
    handshake_sent = False

    while True:
        if stop_state["signal"] is not None:
            _heartbeat(
                config,
                state=("stopped_checkpointed" if previous_iteration >= 0 else "stopped_uncheckpointed"),
                training_pid=None,
                safe_restart_count=safe_restart_count,
                detail="Supervisor stop requested while no child was running",
                validated_iteration=(
                    previous_iteration if previous_iteration >= 0 else None
                ),
            )
            return 0

        baseline_sidecar = _optional_regular_file_fingerprint(config.sidecar)
        baseline_checkpoint = _optional_regular_file_fingerprint(
            config.checkpoint
        )
        actual_resume = _resume_argument_path(
            current_command,
            config.resume_flag,
            working_directory=config.working_directory,
        )
        if external_initial_resume and baseline_checkpoint is None:
            expected_resume = config.initial_resume_checkpoint
        elif baseline_checkpoint is not None:
            expected_resume = config.checkpoint
        else:
            expected_resume = None
        if actual_resume != expected_resume:
            if baseline_checkpoint is not None and actual_resume is None:
                raise SupervisorError(
                    "Configured checkpoint already exists, but the launch "
                    f"command does not contain {config.resume_flag}; refusing "
                    "to overwrite a resumable run"
                )
            if baseline_checkpoint is None and actual_resume is not None:
                raise SupervisorError(
                    f"Launch command requests {config.resume_flag}, but the "
                    "supervisor-owned resume checkpoint is absent"
                )
            raise SupervisorError(
                "Launch resume target does not match the supervisor-owned "
                f"state: {actual_resume} != {expected_resume}"
            )
        child_started_at = time.time()
        _log(
            supervisor_log_handle,
            "Starting training child: " + shlex.join(current_command),
        )
        with _open_append_log(config.training_log) as training_log_handle:
            training_log_handle.write(
                f"\n$ {shlex.join(current_command)}\n"
            )
            training_log_handle.flush()
            try:
                child_environment = dict(os.environ)
                # The production trainer normally refuses every non-empty
                # clean output directory.  Give it a narrow, verifiable way
                # to recognize only this active supervisor's lock, contract,
                # heartbeat, and logs.  The trainer still rejects any model
                # artifact or unrelated file, and resumes remain governed by
                # the checkpoint contract instead.
                child_environment[CHILD_CONTRACT_ENVIRONMENT] = str(
                    config.contract_file
                )
                process = subprocess.Popen(
                    current_command,
                    cwd=config.working_directory,
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=training_log_handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                _heartbeat(
                    config,
                    state="failed",
                    training_pid=None,
                    safe_restart_count=safe_restart_count,
                    failure_kind="spawn_failed",
                    detail=str(exc),
                )
                if not handshake_sent and handshake_state["open"]:
                    handshake_state["open"] = False
                    _send_handshake(
                        handshake_descriptor,
                        {"status": "error", "error": str(exc)},
                    )
                return 1
        stop_state["child"] = process
        if stop_state["signal"] is not None:
            _forward_signal(process, int(stop_state["signal"]))
        try:
            _heartbeat(
                config,
                state="running",
                training_pid=process.pid,
                safe_restart_count=safe_restart_count,
                validated_iteration=(
                    previous_iteration if previous_iteration >= 0 else None
                ),
            )
        except BaseException as exc:
            # Never leave a successfully spawned trainer running without a
            # durable heartbeat and its run-directory-lock owner.
            if not handshake_sent and handshake_state["open"]:
                handshake_state["open"] = False
                _send_handshake(
                    handshake_descriptor,
                    {"status": "error", "error": str(exc)},
                )
            _forward_signal(process, int(signal.SIGTERM))
            process.wait()
            stop_state["child"] = None
            raise SupervisorError(
                f"Initial heartbeat failed after child spawn: {exc}"
            ) from exc
        if not handshake_sent:
            handshake_state["open"] = False
            _send_handshake(
                handshake_descriptor,
                {
                    "status": "started",
                    "supervisor_pid": os.getpid(),
                    "training_pid": process.pid,
                    "heartbeat": str(config.heartbeat_file),
                    "training_log": str(config.training_log),
                },
            )
            handshake_sent = True

        next_heartbeat = time.monotonic() + config.heartbeat_seconds
        heartbeat_failure: str | None = None
        while process.poll() is None:
            if time.monotonic() >= next_heartbeat:
                state = (
                    "stop_requested"
                    if stop_state["signal"] is not None
                    else "running"
                )
                try:
                    _heartbeat(
                        config,
                        state=state,
                        training_pid=process.pid,
                        safe_restart_count=safe_restart_count,
                        validated_iteration=(
                            previous_iteration if previous_iteration >= 0 else None
                        ),
                    )
                except BaseException as exc:
                    heartbeat_failure = repr(exc)
                    _log(
                        supervisor_log_handle,
                        "Heartbeat write failed; requesting safe child stop: "
                        + heartbeat_failure,
                    )
                    if stop_state["signal"] is None:
                        stop_state["signal"] = int(signal.SIGTERM)
                    _forward_signal(process, int(signal.SIGTERM))
                next_heartbeat = time.monotonic() + config.heartbeat_seconds
            time.sleep(min(0.2, config.heartbeat_seconds))

        returncode = int(process.returncode)
        stop_state["child"] = None
        _log(
            supervisor_log_handle,
            f"Training child {process.pid} exited with return code {returncode}",
        )
        if returncode == 0:
            _heartbeat(
                config,
                state="completed",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                returncode=returncode,
                validated_iteration=(
                    previous_iteration if previous_iteration >= 0 else None
                ),
            )
            return 0

        safe_signal = _safe_signal_from_returncode(returncode)
        if safe_signal is None:
            _heartbeat(
                config,
                state="failed",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                returncode=returncode,
                failure_kind=_failure_kind(returncode),
                detail=(
                    "Child failure is not a checkpoint-safe signal; no retry"
                ),
                validated_iteration=(
                    previous_iteration if previous_iteration >= 0 else None
                ),
            )
            return returncode if 0 < returncode < 256 else 1

        _heartbeat(
            config,
            state="validating_interruption",
            training_pid=None,
            safe_restart_count=safe_restart_count,
            returncode=returncode,
            validated_iteration=(
                previous_iteration if previous_iteration >= 0 else None
            ),
        )
        try:
            validated = _validate_interrupted_checkpoint(
                sidecar=config.sidecar,
                checkpoint=config.checkpoint,
                working_directory=config.working_directory,
                expected_signal=safe_signal,
                child_started_at=child_started_at,
                baseline_sidecar=baseline_sidecar,
                baseline_checkpoint=baseline_checkpoint,
                previous_iteration=previous_iteration,
                minimum_checkpoint_bytes=config.minimum_checkpoint_bytes,
            )
        except SupervisorError as exc:
            requested_stop = (
                stop_state["signal"] == safe_signal
                and heartbeat_failure is None
            )
            _heartbeat(
                config,
                state="stopped_uncheckpointed" if requested_stop else "failed",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                returncode=returncode,
                failure_kind=("requested_stop_without_valid_checkpoint" if requested_stop
                              else "invalid_interruption_pair"),
                detail=(("Supervisor stop was requested; no valid interruption checkpoint; "
                         "no restart and no claim of recoverable current progress. ") if requested_stop else "")+str(exc),
                validated_iteration=(
                    previous_iteration if previous_iteration >= 0 else None
                ),
            )
            return 0 if requested_stop else 1

        previous_iteration = validated.iteration
        if heartbeat_failure is not None:
            # A supervisor that cannot durably report liveness must not start
            # another expensive child, even though the child stopped safely.
            return 1
        if stop_state["signal"] is not None:
            _heartbeat(
                config,
                state="stopped_checkpointed",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                returncode=returncode,
                detail=(
                    "Supervisor stop was requested; checkpoint validated and "
                    "left for an explicit future launch"
                ),
                validated_iteration=previous_iteration,
            )
            return 0
        if safe_restart_count >= config.max_safe_restarts:
            _heartbeat(
                config,
                state="failed",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                returncode=returncode,
                failure_kind="safe_restart_limit",
                detail="Validated interruption, but safe restart limit reached",
                validated_iteration=previous_iteration,
            )
            return 1

        safe_restart_count += 1
        current_command = _replace_resume_argument(
            config.base_command,
            config.resume_flag,
            validated.checkpoint,
        )
        external_initial_resume = False
        _log(
            supervisor_log_handle,
            "Validated atomic interruption at iteration "
            f"{previous_iteration}; scheduling safe restart "
            f"{safe_restart_count}/{config.max_safe_restarts}",
        )
        if not _wait_backoff(
            config,
            stop_state=stop_state,
            safe_restart_count=safe_restart_count,
            validated_iteration=previous_iteration,
        ):
            _heartbeat(
                config,
                state="stopped_checkpointed",
                training_pid=None,
                safe_restart_count=safe_restart_count,
                validated_iteration=previous_iteration,
                detail="Supervisor stop requested during restart backoff",
            )
            return 0


def _redirect_supervisor_stdio(supervisor_log: Path) -> Any:
    log_handle = _open_append_log(supervisor_log)
    null_descriptor = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.dup2(null_descriptor, sys.stdin.fileno())
    finally:
        os.close(null_descriptor)
    os.dup2(log_handle.fileno(), sys.stdout.fileno())
    os.dup2(log_handle.fileno(), sys.stderr.fileno())
    return log_handle


def _daemon_entry(config: SupervisorConfig, handshake_descriptor: int) -> None:
    lock_descriptor = -1
    supervisor_log_handle = None
    handshake_state = {"open": True}
    try:
        lock_descriptor = _acquire_run_lock(
            config.lock_file,
            {
                "schema_version": SCHEMA_VERSION,
                "supervisor_pid": os.getpid(),
                "started_at_unix": time.time(),
                "command_sha256": config.command_sha256,
            },
        )
        initial_resume_available = _establish_contract(config)
        supervisor_log_handle = _redirect_supervisor_stdio(config.supervisor_log)
        _log(
            supervisor_log_handle,
            f"Detached supervisor {os.getpid()} acquired {config.lock_file}",
        )
        _heartbeat(
            config,
            state="starting",
            training_pid=None,
            safe_restart_count=0,
        )
        result = _supervise(
            config,
            initial_resume_available=initial_resume_available,
            handshake_descriptor=handshake_descriptor,
            handshake_state=handshake_state,
            supervisor_log_handle=supervisor_log_handle,
        )
        _log(supervisor_log_handle, f"Supervisor exiting with status {result}")
    except BaseException as exc:
        if supervisor_log_handle is not None:
            _log(supervisor_log_handle, f"Fatal supervisor error: {exc!r}")
            traceback.print_exc(file=supervisor_log_handle)
        try:
            _heartbeat(
                config,
                state="failed",
                training_pid=None,
                safe_restart_count=0,
                failure_kind="supervisor_failure",
                detail=str(exc),
            )
        except BaseException:
            pass
        if handshake_state["open"]:
            handshake_state["open"] = False
            _send_handshake(
                handshake_descriptor,
                {"status": "error", "error": str(exc)},
            )
    finally:
        if handshake_state["open"]:
            try:
                os.close(handshake_descriptor)
            except OSError:
                pass
        if supervisor_log_handle is not None:
            supervisor_log_handle.close()
        if lock_descriptor >= 0:
            _release_run_lock(lock_descriptor)


def _read_handshake(descriptor: int, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    reached_eof = False
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [descriptor], [], [], max(0.0, deadline - time.monotonic())
            )
            if not ready:
                break
            chunk = os.read(descriptor, 4096)
            if not chunk:
                reached_eof = True
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    finally:
        os.close(descriptor)
    if not chunks:
        if not reached_eof:
            # The daemon is detached already. A slow checkpoint/NFS read can
            # delay its acknowledgement without preventing a later launch.
            # Never report this as a definite failure that invites a retry.
            return {"status": "launch_unconfirmed", "training_pid": None,
                    "detail": "Acknowledgement timed out; detached supervisor may still launch. Check heartbeat; do not relaunch."}
        raise SupervisorError(
            "Detached supervisor closed its launch pipe without acknowledgement"
        )
    raw = b"".join(chunks).split(b"\n", 1)[0]
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"Invalid supervisor launch response: {exc}") from exc
    if not isinstance(payload, dict):
        raise SupervisorError("Supervisor launch response is not a JSON object")
    return payload


def _double_fork(config: SupervisorConfig) -> dict[str, Any]:
    read_descriptor, write_descriptor = os.pipe2(os.O_CLOEXEC)
    first_pid = os.fork()
    if first_pid:
        os.close(write_descriptor)
        _, status = os.waitpid(first_pid, 0)
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            os.close(read_descriptor)
            raise SupervisorError("First detach fork failed")
        response = _read_handshake(read_descriptor, config.launch_timeout_seconds)
        if response.get("status") == "launch_unconfirmed":
            response.update(heartbeat=str(config.heartbeat_file),
                            command_sha256=config.command_sha256,
                            training_log=str(config.training_log))
        return response

    os.close(read_descriptor)
    try:
        os.setsid()
        second_pid = os.fork()
        if second_pid:
            os.close(write_descriptor)
            os._exit(0)
        # Fork does not honor close-on-exec flags.  Explicitly discard every
        # inherited descriptor except stdio and the launch handshake so a
        # terminal/caller pipe cannot be kept alive by the detached daemon.
        try:
            inherited = [int(value) for value in os.listdir("/proc/self/fd")]
        except OSError:
            inherited = []
        for descriptor in inherited:
            if descriptor not in (0, 1, 2, write_descriptor):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        os.chdir("/")
        os.umask(0o077)
        _daemon_entry(config, write_descriptor)
    except BaseException as exc:
        _send_handshake(
            write_descriptor,
            {"status": "error", "error": str(exc)},
        )
    os._exit(0)


def _artifact_path(run_dir: Path, value: str, option: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.expanduser().resolve()
    else:
        resolved = (run_dir / candidate).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise SupervisorError(f"{option} must remain inside --run-dir") from exc
    return resolved


def _configured_path(run_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate.resolve()


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--working-directory", default=os.getcwd())
    parser.add_argument(
        "--state-sidecar", default="interrupted_training_state.json"
    )
    parser.add_argument(
        "--checkpoint", default="hybrid_teacher_checkpoint.pth"
    )
    parser.add_argument("--lock-file", default=".training_supervisor.lock")
    parser.add_argument(
        "--heartbeat-file", default="training_supervisor_heartbeat.json"
    )
    parser.add_argument(
        "--contract-file", default="training_supervisor_contract.json"
    )
    parser.add_argument("--supervisor-log", default="training_supervisor.log")
    parser.add_argument("--training-log", default="training.log")
    parser.add_argument("--resume-flag", default="--resume")
    parser.add_argument(
        "--initial-resume-checkpoint",
        help=(
            "One-shot checkpoint outside --run-dir. The supervisor injects "
            "it only into the first child; later children resume the rolling "
            "--checkpoint inside --run-dir."
        ),
    )
    parser.add_argument(
        "--one-shot-resume-flag",
        action="append",
        choices=sorted(SUPPORTED_ONE_SHOT_RESUME_FLAGS),
        default=[],
        help=(
            "Trainer boolean flag injected only with --initial-resume-"
            "checkpoint. Because the value begins with '--', pass this "
            "option as --one-shot-resume-flag=VALUE."
        ),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--restart-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--max-safe-restarts", type=int, default=8)
    parser.add_argument("--minimum-checkpoint-bytes", type=int, default=1024)
    parser.add_argument("--launch-timeout-seconds", type=float, default=15.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a training command is required after --")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.restart_backoff_seconds < 0:
        parser.error("--restart-backoff-seconds must be non-negative")
    if args.max_safe_restarts < 0:
        parser.error("--max-safe-restarts must be non-negative")
    if args.minimum_checkpoint_bytes <= 0:
        parser.error("--minimum-checkpoint-bytes must be positive")
    if args.launch_timeout_seconds <= 0:
        parser.error("--launch-timeout-seconds must be positive")
    if not args.resume_flag or "=" in args.resume_flag:
        parser.error("--resume-flag must be a non-empty exact token")
    if args.one_shot_resume_flag and not args.initial_resume_checkpoint:
        parser.error(
            "--one-shot-resume-flag requires --initial-resume-checkpoint"
        )
    if len(args.one_shot_resume_flag) != len(set(args.one_shot_resume_flag)):
        parser.error("--one-shot-resume-flag values must be unique")
    return args


def _build_config(args: argparse.Namespace) -> SupervisorConfig:
    run_dir = Path(args.run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()
    working_directory = Path(args.working_directory).expanduser().resolve()
    if not working_directory.is_dir():
        raise SupervisorError(
            f"Working directory does not exist: {working_directory}"
        )
    base_command = tuple(str(token) for token in args.command)
    one_shot_resume_flags = tuple(args.one_shot_resume_flag)
    for token in base_command:
        for one_shot_flag in SUPPORTED_ONE_SHOT_RESUME_FLAGS:
            if token == one_shot_flag or token.startswith(one_shot_flag + "="):
                raise SupervisorError(
                    f"Trainer flag {one_shot_flag} is one-shot and must be "
                    "supplied through --one-shot-resume-flag, never in the "
                    "base command"
                )
    initial_resume_checkpoint = None
    if args.initial_resume_checkpoint:
        candidate = Path(args.initial_resume_checkpoint).expanduser()
        if not candidate.is_absolute():
            candidate = working_directory / candidate
        initial_resume_checkpoint = candidate.resolve()
        try:
            initial_resume_checkpoint.relative_to(run_dir)
        except ValueError:
            pass
        else:
            raise SupervisorError(
                "--initial-resume-checkpoint must be outside --run-dir; use "
                "--checkpoint for an in-run rolling resume"
            )
    command_sha256 = hashlib.sha256(
        _canonical_json_bytes(list(base_command))
    ).hexdigest()
    config = SupervisorConfig(
        run_dir=run_dir,
        working_directory=working_directory,
        base_command=base_command,
        sidecar=_configured_path(run_dir, args.state_sidecar),
        checkpoint=_configured_path(run_dir, args.checkpoint),
        lock_file=_artifact_path(run_dir, args.lock_file, "--lock-file"),
        heartbeat_file=_artifact_path(
            run_dir, args.heartbeat_file, "--heartbeat-file"
        ),
        contract_file=_artifact_path(
            run_dir, args.contract_file, "--contract-file"
        ),
        supervisor_log=_artifact_path(
            run_dir, args.supervisor_log, "--supervisor-log"
        ),
        training_log=_artifact_path(
            run_dir, args.training_log, "--training-log"
        ),
        resume_flag=str(args.resume_flag),
        initial_resume_checkpoint=initial_resume_checkpoint,
        one_shot_resume_flags=one_shot_resume_flags,
        heartbeat_seconds=float(args.heartbeat_seconds),
        restart_backoff_seconds=float(args.restart_backoff_seconds),
        max_safe_restarts=int(args.max_safe_restarts),
        minimum_checkpoint_bytes=int(args.minimum_checkpoint_bytes),
        launch_timeout_seconds=float(args.launch_timeout_seconds),
        command_sha256=command_sha256,
    )
    artifact_paths = {
        config.lock_file,
        config.heartbeat_file,
        config.contract_file,
        config.supervisor_log,
        config.training_log,
    }
    if len(artifact_paths) != 5:
        raise SupervisorError("Supervisor artifact paths must be distinct")
    if config.sidecar == config.checkpoint:
        raise SupervisorError("Sidecar and checkpoint paths must be distinct")
    for path, option in (
        (config.sidecar, "--state-sidecar"),
        (config.checkpoint, "--checkpoint"),
    ):
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise SupervisorError(
                f"{option} must remain inside --run-dir so its flock owns it"
            ) from exc
    if config.sidecar in artifact_paths or config.checkpoint in artifact_paths:
        raise SupervisorError(
            "Sidecar/checkpoint paths must not overlap supervisor artifacts"
        )
    positions = _resume_positions(base_command, config.resume_flag)
    if len(positions) > 1:
        raise SupervisorError(
            f"Command contains duplicate {config.resume_flag} flags"
        )
    if config.initial_resume_checkpoint is not None and positions:
        raise SupervisorError(
            f"The base command must omit {config.resume_flag} when "
            "--initial-resume-checkpoint is configured; the supervisor owns "
            "both initial and rolling resume targets"
        )
    if positions:
        position = positions[0]
        if position + 1 >= len(base_command):
            raise SupervisorError(
                f"Command has a dangling {config.resume_flag} flag"
            )
        supplied = Path(base_command[position + 1]).expanduser()
        if not supplied.is_absolute():
            supplied = working_directory / supplied
        if supplied.resolve() != config.checkpoint:
            raise SupervisorError(
                f"Initial {config.resume_flag} path does not match configured "
                f"checkpoint {config.checkpoint}"
            )
    return config


def main(argv: list[str] | None = None) -> int:
    args = _parse_arguments(argv)
    try:
        config = _build_config(args)
        response = _double_fork(config)
    except (OSError, SupervisorError) as exc:
        print(f"supervisor launch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(response, sort_keys=True))
    if response.get("status") == "launch_unconfirmed":
        # Successful dispatch is not confirmation of running training. The
        # explicit JSON status and durable heartbeat remain authoritative.
        print(response["detail"], file=sys.stderr)
        return 0
    if response.get("status") != "started":
        print(
            "supervisor launch failed: " + str(response.get("error", response)),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
