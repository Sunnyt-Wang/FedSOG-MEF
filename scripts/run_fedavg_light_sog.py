#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm4bin.federated import (
    build_fedavg_round,
    client_run_name,
    build_local_train_command,
    local_seed_for_round,
    partition_csv_by_client,
)


DEFAULT_TRAIN_CSV = str(Path(os.environ.get("FEDSOG_DATA_ROOT", "data")) / "Dataset-1/training_Dataset-1.csv")


def _model_label(config: str) -> str:
    return Path(config).stem


def _run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _parse_cuda_devices(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cuda_for_job(job_index: int, cuda_devices: list[str]) -> str | None:
    if not cuda_devices:
        return None
    return cuda_devices[job_index % len(cuda_devices)]


def _run_local_job(cmd: list[str], cuda_device: str | None, log_path: str | Path, dry_run: bool) -> None:
    env = os.environ.copy()
    log_path = Path(log_path)
    prefix = []
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        prefix = [f"CUDA_VISIBLE_DEVICES={cuda_device}"]
    print(" ".join(prefix + cmd), flush=True)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(prefix + cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def _run_local_jobs(
    jobs: list[dict[str, object]],
    *,
    dry_run: bool,
    parallel_clients: int,
) -> None:
    if parallel_clients <= 1:
        for job in jobs:
            _run_local_job(
                job["command"],  # type: ignore[arg-type]
                job["cuda_device"],  # type: ignore[arg-type]
                job["log_path"],  # type: ignore[arg-type]
                dry_run,
            )
        return

    with ThreadPoolExecutor(max_workers=parallel_clients) as executor:
        futures = [
            executor.submit(
                _run_local_job,
                job["command"],
                job["cuda_device"],
                job["log_path"],
                dry_run,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            future.result()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run true multi-round FedAvg training for LightSOG clients.")
    parser.add_argument("--config", default="configs/hermessim_sog_light.yaml")
    parser.add_argument("--train-csv", type=Path, default=Path(DEFAULT_TRAIN_CSV))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reports/fedavg_light_sog_true_fl"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument(
        "--initial-global-checkpoint",
        type=Path,
        default=None,
        help="optional checkpoint used to initialize every client before round 1",
    )
    parser.add_argument("--run-prefix", default=None, help="checkpoint run-name prefix; defaults to out-dir name")
    parser.add_argument("--group-by", default=None, help="optional CSV column for non-IID client partitioning")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--parallel-clients", type=int, default=1)
    parser.add_argument("--cuda-devices", default=None, help="comma-separated CUDA devices assigned round-robin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.local_steps <= 0:
        raise ValueError("--local-steps must be positive")
    if args.parallel_clients <= 0:
        raise ValueError("--parallel-clients must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = args.run_prefix or args.out_dir.name
    cuda_devices = _parse_cuda_devices(args.cuda_devices)
    partitions = partition_csv_by_client(
        args.train_csv,
        args.out_dir / "client_partitions",
        num_clients=args.num_clients,
        seed=args.seed,
        group_by=args.group_by,
    )

    all_commands: list[dict[str, object]] = []
    global_checkpoint: Path | None = args.initial_global_checkpoint
    rounds: list[dict[str, object]] = []
    for round_index in range(1, args.rounds + 1):
        round_name = f"round_{round_index:03d}"
        round_dir = args.out_dir / round_name
        round_dir.mkdir(parents=True, exist_ok=True)
        local_commands: list[list[str]] = []
        local_jobs: list[dict[str, object]] = []
        for client_offset, partition in enumerate(partitions):
            run_name = client_run_name(round_index, partition, run_prefix)
            cmd = build_local_train_command(
                config=args.config,
                run_name=run_name,
                partition=partition,
                max_steps=args.local_steps,
                seed=local_seed_for_round(args.seed, round_index, client_offset),
                save_every_steps=args.local_steps,
                init_model_from=global_checkpoint,
                python_bin=args.python_bin,
            )
            local_commands.append(cmd)
            cuda_device = _cuda_for_job(client_offset, cuda_devices)
            log_path = round_dir / "client_logs" / f"{partition.client_id}.log"
            local_job = {
                "round": round_index,
                "client_id": partition.client_id,
                "command": cmd,
                "cuda_device": cuda_device,
                "log_path": str(log_path),
            }
            local_jobs.append(local_job)
            all_commands.append(local_job)

        _run_local_jobs(local_jobs, dry_run=args.dry_run, parallel_clients=args.parallel_clients)

        fedavg_specs = build_fedavg_round(
            round_index=round_index,
            partitions=partitions,
            checkpoint_root=args.checkpoint_root,
            out_dir=round_dir,
            local_steps=args.local_steps,
            run_prefix=run_prefix,
        )
        global_checkpoint = args.out_dir / "global" / round_name
        fedavg_command = [
            args.python_bin,
            "scripts/fedavg_checkpoints.py",
            "--out-dir",
            str(global_checkpoint),
        ]
        for spec in fedavg_specs:
            fedavg_command.extend(["--client", spec])
        all_commands.append({"round": round_index, "client_id": "server", "command": fedavg_command})
        if args.dry_run:
            print(" ".join(fedavg_command), flush=True)
        else:
            subprocess.run(fedavg_command, check=True)

        rounds.append(
            {
                "round_index": round_index,
                "local_commands": local_commands,
                "local_jobs": local_jobs,
                "fedavg_specs": fedavg_specs,
                "global_checkpoint": str(global_checkpoint),
            }
        )

    manifest = {
        "mode": "true_federated_learning",
        "algorithm": "FedAvg",
        "model": _model_label(args.config),
        "config": args.config,
        "train_csv": str(args.train_csv),
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_steps": args.local_steps,
        "initial_global_checkpoint": str(args.initial_global_checkpoint) if args.initial_global_checkpoint else None,
        "group_by": args.group_by,
        "run_prefix": run_prefix,
        "parallel_clients": args.parallel_clients,
        "cuda_devices": cuda_devices,
        "dry_run": args.dry_run,
        "final_global_checkpoint": str(global_checkpoint) if global_checkpoint is not None else None,
        "round_records": rounds,
        "commands": all_commands,
    }
    (args.out_dir / "fedavg_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
