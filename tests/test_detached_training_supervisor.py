from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import supervise_detached_training as supervisor


SCRIPT = REPO_ROOT / "scripts" / "supervise_detached_training.py"


def test_slow_detached_ack_is_unknown_not_failed_or_started():
    reader, writer = os.pipe()
    try:
        response = supervisor._read_handshake(reader, .01)
        assert response['status'] == 'launch_unconfirmed'
        assert response['training_pid'] is None
        assert 'do not relaunch' in response['detail']
    finally:
        os.close(writer)


def test_detached_pipe_eof_without_ack_remains_failure():
    reader, writer = os.pipe()
    os.close(writer)
    with pytest.raises(supervisor.SupervisorError, match='closed its launch pipe'):
        supervisor._read_handshake(reader, .1)


def test_large_stored_metadata_is_streamed_but_compressed_expansion_stays_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,"MAX_PICKLE_BYTES",64)
    for compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
        path=tmp_path/f"checkpoint_{compression}.pth"
        with zipfile.ZipFile(path,"w",compression=compression) as archive:
            archive.writestr("archive/data.pkl",b"metadata"*100)
            archive.writestr("archive/version",b"3\n")
        if compression==zipfile.ZIP_STORED:
            fingerprint=supervisor._validate_pytorch_zip(path,minimum_checkpoint_bytes=1)
            assert fingerprint.size==path.stat().st_size
        else:
            with pytest.raises(supervisor.SupervisorError,match="unsafe uncompressed size"):
                supervisor._validate_pytorch_zip(path,minimum_checkpoint_bytes=1)


def _write_checkpoint(path: Path, *, corrupt: bool = False) -> None:
    if corrupt:
        path.write_bytes(b"PK truncated")
        return
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", b"metadata pickle bytes")
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/data/0", b"tensor-storage")


def _write_sidecar(
    path: Path,
    checkpoint: Path,
    *,
    signum: int = signal.SIGINT,
    iteration: int = 17,
) -> None:
    supervisor._atomic_write_json(
        path,
        {
            "status": "interrupted_checkpointed",
            "iteration": iteration,
            "checkpoint": str(checkpoint),
            "checkpoint_atomic": True,
            "resume_required": True,
            "signal": signal.Signals(signum).name,
            "signal_number": signum,
            "requested_at_unix": time.time(),
        },
    )


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (129, signal.SIGHUP),
        (130, signal.SIGINT),
        (143, signal.SIGTERM),
        (-signal.SIGHUP, signal.SIGHUP),
        (-signal.SIGINT, signal.SIGINT),
        (-signal.SIGTERM, signal.SIGTERM),
        (0, None),
        (1, None),
        (137, None),
        (-signal.SIGKILL, None),
    ],
)
def test_safe_signal_returncode_is_narrow(returncode, expected):
    assert supervisor._safe_signal_from_returncode(returncode) == expected


def test_sigkill_or_oom_is_classified_but_never_safe_to_retry():
    assert supervisor._failure_kind(137) == "sigkill_or_oom"
    assert supervisor._failure_kind(-signal.SIGKILL) == "sigkill_or_oom"
    assert supervisor._safe_signal_from_returncode(137) is None


def test_resume_argument_is_replaced_or_appended(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    assert supervisor._replace_resume_argument(
        ("python", "train.py"), "--resume", checkpoint
    ) == ("python", "train.py", "--resume", str(checkpoint))
    assert supervisor._replace_resume_argument(
        ("python", "train.py", "--resume", "old.pth"),
        "--resume",
        checkpoint,
    ) == ("python", "train.py", "--resume", str(checkpoint))
    with pytest.raises(supervisor.SupervisorError, match="duplicate"):
        supervisor._replace_resume_argument(
            ("train", "--resume", "a", "--resume", "b"),
            "--resume",
            checkpoint,
        )
    with pytest.raises(supervisor.SupervisorError, match="dangling"):
        supervisor._replace_resume_argument(
            ("train", "--resume"), "--resume", checkpoint
        )


def test_valid_interruption_pair_is_accepted_without_loading_tensors(tmp_path):
    checkpoint = (tmp_path / "checkpoint.pth").resolve()
    sidecar = tmp_path / "interrupted_training_state.json"
    child_started_at = time.time() - 0.1
    _write_checkpoint(checkpoint)
    _write_sidecar(sidecar, checkpoint, iteration=23)

    validated = supervisor._validate_interrupted_checkpoint(
        sidecar=sidecar,
        checkpoint=checkpoint,
        working_directory=tmp_path,
        expected_signal=signal.SIGINT,
        child_started_at=child_started_at,
        baseline_sidecar=None,
        baseline_checkpoint=None,
        previous_iteration=11,
        minimum_checkpoint_bytes=1,
    )

    assert validated.iteration == 23
    assert validated.checkpoint == checkpoint
    assert validated.checkpoint_fingerprint.size == checkpoint.stat().st_size


def test_stale_or_corrupt_interruption_pair_is_rejected(tmp_path):
    checkpoint = (tmp_path / "checkpoint.pth").resolve()
    sidecar = tmp_path / "interrupted_training_state.json"
    _write_checkpoint(checkpoint)
    _write_sidecar(sidecar, checkpoint)
    sidecar_baseline = supervisor._regular_file_fingerprint(sidecar)
    checkpoint_baseline = supervisor._regular_file_fingerprint(checkpoint)

    with pytest.raises(supervisor.SupervisorError, match="not replaced"):
        supervisor._validate_interrupted_checkpoint(
            sidecar=sidecar,
            checkpoint=checkpoint,
            working_directory=tmp_path,
            expected_signal=signal.SIGINT,
            child_started_at=time.time() - 0.1,
            baseline_sidecar=sidecar_baseline,
            baseline_checkpoint=checkpoint_baseline,
            previous_iteration=0,
            minimum_checkpoint_bytes=1,
        )

    _write_checkpoint(checkpoint, corrupt=True)
    _write_sidecar(sidecar, checkpoint)
    with pytest.raises(supervisor.SupervisorError, match="ZIP"):
        supervisor._validate_interrupted_checkpoint(
            sidecar=sidecar,
            checkpoint=checkpoint,
            working_directory=tmp_path,
            expected_signal=signal.SIGINT,
            child_started_at=time.time() - 0.1,
            baseline_sidecar=None,
            baseline_checkpoint=None,
            previous_iteration=0,
            minimum_checkpoint_bytes=1,
        )


def test_run_directory_lock_is_exclusive(tmp_path):
    lock = tmp_path / ".lock"
    first = supervisor._acquire_run_lock(lock, {"owner": "first"})
    try:
        with pytest.raises(supervisor.SupervisorError, match="already owns"):
            supervisor._acquire_run_lock(lock, {"owner": "second"})
    finally:
        supervisor._release_run_lock(first)
    second = supervisor._acquire_run_lock(lock, {"owner": "second"})
    supervisor._release_run_lock(second)


def _launch(
    run_dir: Path,
    worker: Path,
    *,
    supervisor_options: tuple[str, ...] = (),
    worker_options: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--working-directory",
            str(run_dir),
            "--heartbeat-seconds",
            "0.05",
            "--restart-backoff-seconds",
            "0.05",
            "--max-safe-restarts",
            "2",
            "--minimum-checkpoint-bytes",
            "1",
            "--launch-timeout-seconds",
            "5",
            *supervisor_options,
            "--",
            sys.executable,
            str(worker),
            "--run-dir",
            str(run_dir),
            *worker_options,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _wait_for_terminal_heartbeat(run_dir: Path) -> dict:
    heartbeat = run_dir / "training_supervisor_heartbeat.json"
    deadline = time.monotonic() + 10
    last = None
    while time.monotonic() < deadline:
        try:
            last = json.loads(heartbeat.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        if last.get("state") in {"completed", "failed", "stopped_checkpointed"}:
            return last
        time.sleep(0.02)
    raise AssertionError(f"supervisor did not finish; last heartbeat={last!r}")


def test_detached_supervisor_resumes_only_valid_safe_interruption(tmp_path):
    run_dir = tmp_path / "safe-run"
    run_dir.mkdir()
    worker = run_dir / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import os
            from pathlib import Path
            import signal
            import sys
            import time
            import zipfile

            parser = argparse.ArgumentParser()
            parser.add_argument("--run-dir", required=True)
            parser.add_argument("--resume")
            args = parser.parse_args()
            run = Path(args.run_dir)
            attempts_path = run / "attempts.txt"
            attempts = int(attempts_path.read_text()) + 1 if attempts_path.exists() else 1
            attempts_path.write_text(str(attempts))
            checkpoint = (run / "hybrid_teacher_checkpoint.pth").resolve()
            if attempts == 1:
                checkpoint_tmp = run / ".checkpoint.tmp"
                with zipfile.ZipFile(checkpoint_tmp, "w", zipfile.ZIP_STORED) as archive:
                    archive.writestr("archive/data.pkl", b"metadata")
                    archive.writestr("archive/version", b"3\\n")
                    archive.writestr("archive/data/0", b"storage")
                os.replace(checkpoint_tmp, checkpoint)
                sidecar = run / "interrupted_training_state.json"
                sidecar_tmp = run / ".sidecar.tmp"
                sidecar_tmp.write_text(json.dumps({
                    "status": "interrupted_checkpointed",
                    "iteration": 7,
                    "checkpoint": str(checkpoint),
                    "checkpoint_atomic": True,
                    "resume_required": True,
                    "signal": "SIGINT",
                    "signal_number": signal.SIGINT,
                    "requested_at_unix": time.time(),
                }))
                os.replace(sidecar_tmp, sidecar)
                raise SystemExit(128 + signal.SIGINT)
            if Path(args.resume).resolve() != checkpoint:
                raise SystemExit(9)
            (run / "completed.txt").write_text("resumed")
            """
        )
    )

    launched = _launch(run_dir, worker)
    assert launched.returncode == 0, launched.stderr
    response = json.loads(launched.stdout)
    assert response["status"] == "started"
    heartbeat = _wait_for_terminal_heartbeat(run_dir)

    assert heartbeat["state"] == "completed"
    assert heartbeat["safe_restart_count"] == 1
    assert heartbeat["validated_iteration"] == 7
    assert (run_dir / "attempts.txt").read_text() == "2"
    assert (run_dir / "completed.txt").read_text() == "resumed"
    assert (run_dir / "training.log").is_file()
    assert (run_dir / "training_supervisor.log").is_file()


def test_external_initial_resume_and_migration_flag_are_one_shot(tmp_path):
    run_dir = tmp_path / "v114-run"
    run_dir.mkdir()
    predecessor = tmp_path / "v113-iteration-18000.pth"
    _write_checkpoint(predecessor)
    worker = tmp_path / "v114_worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            import os
            from pathlib import Path
            import signal
            import time
            import zipfile

            parser = argparse.ArgumentParser()
            parser.add_argument("--run-dir", required=True)
            parser.add_argument("--expected-initial", required=True)
            parser.add_argument("--resume", required=True)
            parser.add_argument(
                {supervisor.V114_OPTICAL_OWNERSHIP_REPAIR_FLAG!r},
                action="store_true",
            )
            args = parser.parse_args()
            run = Path(args.run_dir)
            attempts_path = run / "attempts.txt"
            attempt = (
                int(attempts_path.read_text()) + 1
                if attempts_path.exists()
                else 1
            )
            attempts_path.write_text(str(attempt))
            rolling = (run / "hybrid_teacher_checkpoint.pth").resolve()
            observed = run / f"observed-{{attempt}}.json"
            observed.write_text(json.dumps({{
                "resume": str(Path(args.resume).resolve()),
                "migration": bool(
                    args.allow_v114_optical_ownership_repair_resume
                ),
            }}))
            if attempt == 1:
                if Path(args.resume).resolve() != Path(args.expected_initial).resolve():
                    raise SystemExit(31)
                if not args.allow_v114_optical_ownership_repair_resume:
                    raise SystemExit(32)
                temporary = run / ".rolling.tmp"
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
                    archive.writestr("archive/data.pkl", b"metadata")
                    archive.writestr("archive/version", b"3\\n")
                    archive.writestr("archive/data/0", b"storage")
                os.replace(temporary, rolling)
                sidecar = run / "interrupted_training_state.json"
                temporary_sidecar = run / ".sidecar.tmp"
                temporary_sidecar.write_text(json.dumps({{
                    "status": "interrupted_checkpointed",
                    "iteration": 18001,
                    "checkpoint": str(rolling),
                    "checkpoint_atomic": True,
                    "resume_required": True,
                    "signal": "SIGINT",
                    "signal_number": signal.SIGINT,
                    "requested_at_unix": time.time(),
                }}))
                os.replace(temporary_sidecar, sidecar)
                raise SystemExit(128 + signal.SIGINT)
            if Path(args.resume).resolve() != rolling:
                raise SystemExit(33)
            if args.allow_v114_optical_ownership_repair_resume:
                raise SystemExit(34)
            (run / "completed.txt").write_text("rolling exact resume")
            """
        )
    )

    launched = _launch(
        run_dir,
        worker,
        supervisor_options=(
            "--initial-resume-checkpoint",
            str(predecessor),
            "--one-shot-resume-flag="
            + supervisor.V114_OPTICAL_OWNERSHIP_REPAIR_FLAG,
        ),
        worker_options=("--expected-initial", str(predecessor)),
    )
    assert launched.returncode == 0, launched.stderr
    heartbeat = _wait_for_terminal_heartbeat(run_dir)

    assert heartbeat["state"] == "completed"
    assert heartbeat["safe_restart_count"] == 1
    assert (run_dir / "attempts.txt").read_text() == "2"
    first = json.loads((run_dir / "observed-1.json").read_text())
    second = json.loads((run_dir / "observed-2.json").read_text())
    assert first == {"resume": str(predecessor.resolve()), "migration": True}
    assert second == {
        "resume": str(
            (run_dir / "hybrid_teacher_checkpoint.pth").resolve()
        ),
        "migration": False,
    }
    assert (run_dir / "completed.txt").read_text() == "rolling exact resume"


def test_external_exact_resume_does_not_require_a_migration_flag(tmp_path):
    run_dir = tmp_path / "exact-resume-run"
    run_dir.mkdir()
    predecessor = tmp_path / "same-implementation-checkpoint.pth"
    _write_checkpoint(predecessor)
    worker = tmp_path / "exact_resume_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--run-dir", required=True)
            parser.add_argument("--resume", required=True)
            args = parser.parse_args()
            run = Path(args.run_dir)
            (run / "observed_resume.txt").write_text(
                str(Path(args.resume).resolve())
            )
            """
        )
    )

    launched = _launch(
        run_dir,
        worker,
        supervisor_options=(
            "--initial-resume-checkpoint",
            str(predecessor),
        ),
    )
    assert launched.returncode == 0, launched.stderr
    heartbeat = _wait_for_terminal_heartbeat(run_dir)

    assert heartbeat["state"] == "completed"
    assert heartbeat["safe_restart_count"] == 0
    assert (run_dir / "observed_resume.txt").read_text() == str(
        predecessor.resolve()
    )
    contract = json.loads(
        (run_dir / "training_supervisor_contract.json").read_text()
    )
    assert contract["initial_resume_checkpoint"] == str(
        predecessor.resolve()
    )
    assert contract["one_shot_resume_flags"] == []


def test_v115_migration_flag_is_supported_as_one_shot_only():
    assert supervisor.V115_PERSISTENT_OWNERSHIP_DEBT_FLAG in (
        supervisor.SUPPORTED_ONE_SHOT_RESUME_FLAGS
    )


def test_v116_migration_flag_is_supported_as_one_shot_only():
    assert supervisor.V116_SHARED_ENVELOPE_LOCALIZATION_FLAG in (
        supervisor.SUPPORTED_ONE_SHOT_RESUME_FLAGS
    )


def test_one_shot_resume_cannot_be_reused_without_rolling_checkpoint(tmp_path):
    run_dir = tmp_path / "consumed-run"
    run_dir.mkdir()
    predecessor = tmp_path / "predecessor.pth"
    _write_checkpoint(predecessor)
    worker = tmp_path / "failing_initial_worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--run-dir", required=True)
            parser.add_argument("--resume", required=True)
            parser.add_argument(
                {supervisor.V114_OPTICAL_OWNERSHIP_REPAIR_FLAG!r},
                action="store_true",
            )
            args = parser.parse_args()
            attempts = Path(args.run_dir) / "attempts.txt"
            count = int(attempts.read_text()) + 1 if attempts.exists() else 1
            attempts.write_text(str(count))
            raise SystemExit(1)
            """
        )
    )
    options = (
        "--initial-resume-checkpoint",
        str(predecessor),
        "--one-shot-resume-flag="
        + supervisor.V114_OPTICAL_OWNERSHIP_REPAIR_FLAG,
    )

    first_launch = _launch(
        run_dir,
        worker,
        supervisor_options=options,
    )
    assert first_launch.returncode == 0, first_launch.stderr
    first_heartbeat = _wait_for_terminal_heartbeat(run_dir)
    assert first_heartbeat["state"] == "failed"
    assert (run_dir / "attempts.txt").read_text() == "1"

    second_launch = _launch(
        run_dir,
        worker,
        supervisor_options=options,
    )
    assert second_launch.returncode == 1
    assert "already consumed" in second_launch.stderr
    assert (run_dir / "attempts.txt").read_text() == "1"


def test_one_shot_trainer_flag_is_rejected_in_base_command(tmp_path):
    run_dir = tmp_path / "base-flag-run"
    worker = tmp_path / "worker.py"
    worker.write_text("raise SystemExit(0)\n")

    launched = _launch(
        run_dir,
        worker,
        worker_options=(supervisor.V114_OPTICAL_OWNERSHIP_REPAIR_FLAG,),
    )

    assert launched.returncode == 1
    assert "one-shot" in launched.stderr
    assert not (run_dir / "training_supervisor_contract.json").exists()


def test_detached_supervisor_does_not_retry_logic_failure(tmp_path):
    run_dir = tmp_path / "failed-run"
    run_dir.mkdir()
    worker = run_dir / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--run-dir", required=True)
            args = parser.parse_args()
            attempts = Path(args.run_dir) / "attempts.txt"
            count = int(attempts.read_text()) + 1 if attempts.exists() else 1
            attempts.write_text(str(count))
            raise SystemExit(1)
            """
        )
    )

    launched = _launch(run_dir, worker)
    assert launched.returncode == 0, launched.stderr
    heartbeat = _wait_for_terminal_heartbeat(run_dir)

    assert heartbeat["state"] == "failed"
    assert heartbeat["failure_kind"] == "child_nonzero_exit"
    assert heartbeat["safe_restart_count"] == 0
    time.sleep(0.1)
    assert (run_dir / "attempts.txt").read_text() == "1"


def test_training_child_receives_exact_supervisor_contract_path(tmp_path):
    run_dir = tmp_path / "contract-environment-run"
    run_dir.mkdir()
    worker = tmp_path / "contract_worker.py"
    worker.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from scripts.train_unified_outdoor_teacher import "
        "_assert_clean_training_output\n"
        "run = Path(os.environ['G4SPLAT_DETACHED_SUPERVISOR_CONTRACT']).parent\n"
        "_assert_clean_training_output(run, resume_requested=False)\n"
        "(run / 'child_contract_path.txt').write_text("
        "os.environ['G4SPLAT_DETACHED_SUPERVISOR_CONTRACT'])\n"
    )

    launched = _launch(run_dir, worker)
    assert launched.returncode == 0, launched.stderr
    heartbeat = _wait_for_terminal_heartbeat(run_dir)
    assert heartbeat["state"] == "completed"
    contract = run_dir / "training_supervisor_contract.json"
    assert (run_dir / "child_contract_path.txt").read_text() == str(
        contract.resolve()
    )


def test_existing_checkpoint_is_not_silently_overwritten(tmp_path):
    run_dir = tmp_path / "existing-checkpoint-run"
    run_dir.mkdir()
    _write_checkpoint(run_dir / "hybrid_teacher_checkpoint.pth")
    worker = run_dir / "worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        "Path('unexpected-start.txt').write_text('started')\n"
    )

    launched = _launch(run_dir, worker)

    assert launched.returncode == 1
    assert "does not contain --resume" in launched.stderr
    assert not (run_dir / "unexpected-start.txt").exists()
    heartbeat = json.loads(
        (run_dir / "training_supervisor_heartbeat.json").read_text()
    )
    assert heartbeat["state"] == "failed"
    assert heartbeat["failure_kind"] == "supervisor_failure"
