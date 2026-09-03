#!/usr/bin/env python3
"""Build one immutable MoGe3 evidence cache with detached GPU shards.

The per-view producer is already content addressed and writes each archive
atomically.  This supervisor adds the missing run-level guarantees: disjoint
camera ranges, one frozen command contract, a run-directory lock, detached
workers, atomic heartbeat state, and a final all-camera index pass only after
every shard succeeds.  Re-running the same command is safe because completed
views are validated and reused by ``build_moge3_evidence.py``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "g4splat-moge3-detached-supervisor-v1"
DEFAULT_MODEL = "Ruicheng/moge-3-vitl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(
                json.dumps(
                    payload, sort_keys=True, indent=2, ensure_ascii=False
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _shard_ranges(total: int, count: int) -> tuple[tuple[int, int], ...]:
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("Shard count must be in [1, number of cameras]")
    base, remainder = divmod(int(total), int(count))
    start = 0
    result: list[tuple[int, int]] = []
    for index in range(count):
        limit = base + (1 if index < remainder else 0)
        result.append((start, limit))
        start += limit
    if start != total:
        raise AssertionError("Internal shard partition is incomplete")
    return tuple(result)


def _gpu_memory_used() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, used = line.split(",", 1)
        output[int(index.strip())] = int(used.strip())
    return output


def _worker_command(
    *,
    python: Path,
    dataset: Path,
    scene_contract: Path,
    output: Path,
    model: str,
    model_revision: str,
    refine_steps: int,
    resolution_level: int,
    start: int | None,
    limit: int | None,
    use_fp16: bool,
) -> tuple[str, ...]:
    command = [
        str(python),
        str(REPO_ROOT / "scripts/build_moge3_evidence.py"),
        "--dataset",
        str(dataset),
        "--scene-contract",
        str(scene_contract),
        "--output",
        str(output),
        "--model",
        str(model),
        "--model-revision",
        str(model_revision),
        "--device",
        "cuda:0",
        "--refine-steps",
        str(int(refine_steps)),
        "--resolution-level",
        str(int(resolution_level)),
    ]
    if use_fp16:
        command.append("--use-fp16")
    if start is not None:
        command.extend(("--start", str(int(start))))
    if limit is not None:
        command.extend(("--limit", str(int(limit))))
    return tuple(command)


def _detach(supervisor_log: Path) -> int | None:
    """Return ``None`` in the daemon and its pid in the original process."""

    read_fd, write_fd = os.pipe()
    first = os.fork()
    if first > 0:
        os.close(write_fd)
        with os.fdopen(read_fd, "r", encoding="ascii") as handle:
            payload = handle.read().strip()
        os.waitpid(first, 0)
        if not payload:
            raise RuntimeError("Detached MoGe3 supervisor failed to publish pid")
        return int(payload)

    os.close(read_fd)
    os.setsid()
    second = os.fork()
    if second > 0:
        os._exit(0)

    supervisor_log.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(
        supervisor_log,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
        0o600,
    )
    null_fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
    os.dup2(null_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    if null_fd > 2:
        os.close(null_fd)
    if log_fd > 2:
        os.close(log_fd)
    os.chdir(REPO_ROOT)
    os.umask(0o077)
    os.write(write_fd, f"{os.getpid()}\n".encode("ascii"))
    os.close(write_fd)
    return None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument(
        "--maximum-preexisting-gpu-memory-mib", type=int, default=1024
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--foreground", action="store_true")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    dataset = args.dataset.expanduser().resolve()
    scene_contract = args.scene_contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    python = args.python.expanduser().resolve()
    if not dataset.is_dir() or not scene_contract.is_file():
        raise FileNotFoundError("Dataset or scene contract is absent")
    if not python.is_file():
        raise FileNotFoundError(python)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU ids must be unique")
    scene = json.loads(scene_contract.read_text(encoding="utf-8"))
    if Path(scene["dataset"]).resolve() != dataset:
        raise RuntimeError("Scene contract belongs to a different dataset")
    total = len(scene.get("records", []))
    ranges = _shard_ranges(total, len(args.gpus))
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    memory = _gpu_memory_used()
    unavailable = {
        int(gpu): int(memory.get(int(gpu), -1))
        for gpu in args.gpus
        if memory.get(int(gpu), sys.maxsize)
        > int(args.maximum_preexisting_gpu_memory_mib)
    }
    if unavailable:
        raise RuntimeError(
            "Refusing to overlap MoGe3 with occupied GPUs: "
            + ", ".join(f"GPU {key}={value} MiB" for key, value in unavailable.items())
        )

    build_script = REPO_ROOT / "scripts/build_moge3_evidence.py"
    evidence_module = REPO_ROOT / "outdoor/moge3_evidence.py"
    commands = [
        _worker_command(
            python=python,
            dataset=dataset,
            scene_contract=scene_contract,
            output=output,
            model=args.model,
            model_revision=args.model_revision,
            refine_steps=args.refine_steps,
            resolution_level=args.resolution_level,
            start=start,
            limit=limit,
            use_fp16=args.use_fp16,
        )
        for start, limit in ranges
    ]
    final_command = _worker_command(
        python=python,
        dataset=dataset,
        scene_contract=scene_contract,
        output=output,
        model=args.model,
        model_revision=args.model_revision,
        refine_steps=args.refine_steps,
        resolution_level=args.resolution_level,
        start=None,
        limit=None,
        use_fp16=args.use_fp16,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset),
        "dataset_scene_contract_sha256": _sha256(scene_contract),
        "output": str(output),
        "python": str(python),
        "python_sha256": _sha256(python),
        "model": str(args.model),
        "model_revision": str(args.model_revision),
        "refine_steps": int(args.refine_steps),
        "resolution_level": int(args.resolution_level),
        "use_fp16": bool(args.use_fp16),
        "gpus": [int(value) for value in args.gpus],
        "ranges": [list(value) for value in ranges],
        "commands": [list(value) for value in commands],
        "final_command": list(final_command),
        "implementation_hashes": {
            "supervisor": _sha256(Path(__file__).resolve()),
            "producer": _sha256(build_script),
            "evidence_module": _sha256(evidence_module),
        },
    }
    contract["contract_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    contract_path = output / "moge3_supervisor_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(
                "Existing MoGe3 run has a different frozen supervisor contract"
            )
    else:
        _atomic_json(contract_path, contract)

    supervisor_log = logs / "moge3_supervisor.log"
    if not args.foreground:
        daemon_pid = _detach(supervisor_log)
        if daemon_pid is not None:
            print(
                json.dumps(
                    {
                        "status": "detached",
                        "pid": daemon_pid,
                        "contract_sha256": contract["contract_sha256"],
                        "state": str(output / "moge3_supervisor_state.json"),
                        "log": str(supervisor_log),
                    },
                    indent=2,
                )
            )
            return 0

    lock_handle = (output / ".moge3_supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another MoGe3 supervisor owns this output") from error

    state_path = output / "moge3_supervisor_state.json"
    stop_requested = False

    def request_stop(_number: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, request_stop)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(REPO_ROOT),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    if args.hf_home is not None:
        environment["HF_HOME"] = str(args.hf_home.expanduser().resolve())

    workers: list[dict[str, Any]] = []
    for shard, (gpu, command, selected_range) in enumerate(
        zip(args.gpus, commands, ranges)
    ):
        log_path = logs / f"moge3_shard_{shard}_gpu{gpu}.log"
        log_handle = log_path.open("ab", buffering=0)
        worker_environment = dict(environment)
        worker_environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=worker_environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        workers.append(
            {
                "shard": shard,
                "gpu": int(gpu),
                "range": list(selected_range),
                "pid": int(process.pid),
                "process": process,
                "log": str(log_path),
                "log_handle": log_handle,
                "started_unix": time.time(),
                "returncode": None,
            }
        )

    def publish(status: str, *, finalizer: dict[str, Any] | None = None) -> None:
        _atomic_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "supervisor_pid": os.getpid(),
                "contract_sha256": contract["contract_sha256"],
                "updated_unix": time.time(),
                "completed_view_archives": len(list((output / "views").glob("*.npz"))),
                "total_views": total,
                "workers": [
                    {
                        key: value
                        for key, value in worker.items()
                        if key not in {"process", "log_handle"}
                    }
                    for worker in workers
                ],
                "finalizer": finalizer,
            },
        )

    publish("running")
    while True:
        running = 0
        for worker in workers:
            process = worker["process"]
            returncode = process.poll()
            worker["returncode"] = returncode
            if returncode is None:
                running += 1
        if stop_requested:
            for worker in workers:
                process = worker["process"]
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for worker in workers:
                worker["returncode"] = worker["process"].wait()
            publish("stopped")
            return 130
        publish("running" if running else "shards_finished")
        if running == 0:
            break
        time.sleep(max(float(args.heartbeat_seconds), 1.0))

    for worker in workers:
        worker["log_handle"].close()
    failures = [
        worker for worker in workers if int(worker["returncode"] or 0) != 0
    ]
    if failures:
        publish("failed")
        return 1

    final_log_path = logs / "moge3_final_index.log"
    final_started = time.time()
    with final_log_path.open("ab", buffering=0) as final_log:
        final_env = dict(environment)
        final_env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpus[0]))
        final_returncode = subprocess.run(
            final_command,
            cwd=REPO_ROOT,
            env=final_env,
            stdin=subprocess.DEVNULL,
            stdout=final_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            check=False,
        ).returncode
    finalizer = {
        "returncode": int(final_returncode),
        "log": str(final_log_path),
        "started_unix": final_started,
        "finished_unix": time.time(),
    }
    index = output / "moge3_index.json"
    if final_returncode != 0 or not index.is_file():
        publish("failed_finalization", finalizer=finalizer)
        return 1
    finalizer["index"] = str(index)
    finalizer["index_sha256"] = _sha256(index)
    publish("complete", finalizer=finalizer)
    return 0


def main() -> None:
    try:
        raise SystemExit(_run(_args()))
    except Exception as error:
        print(f"MoGe3 supervisor failed: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
