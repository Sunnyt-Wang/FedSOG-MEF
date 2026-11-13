#!/usr/bin/env python
"""Run reviewer-facing federated optimization and robustness experiments.

The runner keeps the model, optimizer, local-step budget, client weights, and
initial checkpoint fixed while varying only the declared federated condition.
It supports FedAvg, FedProx, SCAFFOLD, three deterministic partition regimes,
and a model-update poisoning stress test.  It intentionally writes model-only
checkpoints: the subsequent evaluator does not need optimizer state and this
keeps the experiment reproducible within the available disk budget.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from llm4bin.federated import ClientPartition, build_local_train_command


DEFAULT_TRAIN_CSV = Path(os.environ.get("FEDSOG_DATA_ROOT", "data")) / "Dataset-1/training_Dataset-1.csv"
DEFAULT_INIT = Path(
    "outputs/checkpoints/initializers/msoftv2/step_000030000"
)


def split_iid(df: pd.DataFrame, clients: int, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(len(df))
    assignment = np.empty(len(df), dtype=np.int16)
    assignment[order] = np.arange(len(df), dtype=np.int64) % clients
    return assignment


def split_dirichlet_project(
    df: pd.DataFrame, clients: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    assignment = np.empty(len(df), dtype=np.int16)
    for _project, indices in df.groupby("project", sort=True).groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        indices = rng.permutation(indices)
        proportions = rng.dirichlet(np.full(clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(indices), proportions)
        start = 0
        for client, count in enumerate(counts):
            assignment[indices[start : start + count]] = client
            start += count
    return assignment


def split_high_skew(df: pd.DataFrame, clients: int, rng: np.random.Generator) -> np.ndarray:
    mapping = {
        ("openssl", "x"): 0,
        ("openssl", "arm"): 4,
        ("openssl", "mips"): 4,
        ("clamav", "x"): 1,
        ("clamav", "arm"): 1,
        ("clamav", "mips"): 1,
        ("curl", "x"): 2,
        ("curl", "arm"): 2,
        ("curl", "mips"): 2,
        ("unrar", "x"): 3,
        ("unrar", "arm"): 3,
        ("unrar", "mips"): 3,
    }
    unknown = sorted(set(zip(df["project"], df["arch"])) - set(mapping))
    if unknown:
        raise ValueError(f"high-skew mapping has unknown project/arch pairs: {unknown}")
    macro_assignment = np.asarray(
        [mapping[(project, arch)] for project, arch in zip(df["project"], df["arch"])],
        dtype=np.int16,
    )
    if clients == 5:
        return macro_assignment
    if clients % 5 != 0:
        raise ValueError("high_skew requires five macro groups or a multiple of five clients")

    # Preserve the declared project/architecture skew while splitting each
    # macro-client into equally sized sub-clients for matched expert pools.
    subclients = clients // 5
    assignment = np.empty(len(df), dtype=np.int16)
    for macro_client in range(5):
        indices = np.flatnonzero(macro_assignment == macro_client)
        indices = rng.permutation(indices)
        for offset, index in enumerate(indices):
            assignment[index] = macro_client * subclients + (offset % subclients)
    return assignment


def build_partitions(
    train_csv: Path, out_dir: Path, clients: int, seed: int, condition: str
) -> list[ClientPartition]:
    df = pd.read_csv(train_csv)
    if df.empty:
        raise ValueError(f"empty training csv: {train_csv}")
    rng = np.random.default_rng(seed)
    if condition == "iid":
        assignment = split_iid(df, clients, rng)
    elif condition == "medium_skew":
        assignment = split_dirichlet_project(df, clients, 0.5, rng)
    elif condition == "high_skew":
        assignment = split_high_skew(df, clients, rng)
    else:
        raise ValueError(f"unsupported partition condition: {condition}")

    partitions: list[ClientPartition] = []
    for client in range(clients):
        frame = df.loc[assignment == client].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"partition produced an empty client: client_{client:02d}")
        client_id = f"client_{client:02d}"
        client_dir = out_dir / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        csv_path = client_dir / "train.csv"
        frame.to_csv(csv_path, index=False)
        partitions.append(ClientPartition(client_id, csv_path, len(frame), float(len(frame))))

    manifest = {
        "protocol_version": "reviewer_protocol_v2",
        "source_csv": str(train_csv),
        "condition": condition,
        "clients": clients,
        "seed": seed,
        "partition_seed": seed,
        "rows": len(df),
        "client_rows": {p.client_id: p.num_rows for p in partitions},
    }
    (out_dir / "partition_manifest.json").write_text(json.dumps(manifest, indent=2))
    return partitions


def parse_devices(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def run_local_job(job: dict[str, Any], dry_run: bool) -> None:
    env = os.environ.copy()
    device = job.get("cuda_device")
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in job["command"]]
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        if not dry_run:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_jobs(jobs: list[dict[str, Any]], parallel: int, dry_run: bool) -> None:
    if dry_run:
        for job in jobs:
            print("[dry-run]", " ".join(str(x) for x in job["command"]), flush=True)
        return
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(run_local_job, job, dry_run) for job in jobs]
        for future in as_completed(futures):
            future.result()


def zero_controls(init_checkpoint: Path, out_path: Path) -> None:
    state = load_file(str(init_checkpoint / "model.safetensors"), device="cpu")
    controls = {name: torch.zeros_like(value, dtype=torch.float32) for name, value in state.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"controls": controls}, out_path)


def load_controls(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    controls = payload.get("controls", payload) if isinstance(payload, dict) else payload
    if not isinstance(controls, dict):
        raise ValueError(f"invalid control file: {path}")
    return {name: value.float().cpu() for name, value in controls.items()}


def update_global_control(
    old_path: Path,
    client_old_paths: list[Path],
    client_new_paths: list[Path],
    weights: list[float],
    out_path: Path,
) -> None:
    old = load_controls(old_path)
    old_clients = [load_controls(path) for path in client_old_paths]
    new_clients = [load_controls(path) for path in client_new_paths]
    names = set(old)
    for controls in [*old_clients, *new_clients]:
        names.update(controls)
    output: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        reference = old.get(name)
        if reference is None:
            reference = next(controls[name] for controls in new_clients if name in controls)
        value = reference.clone()
        for weight, old_client, new_client in zip(weights, old_clients, new_clients, strict=True):
            value = value + float(weight) * (
                new_client.get(name, torch.zeros_like(value))
                - old_client.get(name, torch.zeros_like(value))
            )
        output[name] = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"controls": output}, out_path)


def checkpoint_for(partition: ClientPartition, checkpoint_root: Path, run_name: str, steps: int) -> Path:
    return checkpoint_root / f"{run_name}_round_{{round_index:03d}}_{partition.client_id}" / f"step_{steps:09d}"


def aggregate_states(
    global_checkpoint: Path,
    client_checkpoints: list[Path],
    weights: list[float],
    out_dir: Path,
    attack_client_ids: set[str] | None = None,
    attack_factor: float = 1.0,
    client_ids: list[str] | None = None,
) -> dict[str, Any]:
    global_state = load_file(str(global_checkpoint / "model.safetensors"), device="cpu")
    client_states = [load_file(str(path / "model.safetensors"), device="cpu") for path in client_checkpoints]
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum()
    attack_client_ids = attack_client_ids or set()
    client_ids = client_ids or [f"client_{i:02d}" for i in range(len(client_states))]
    output: dict[str, torch.Tensor] = {}
    attacked: list[str] = []
    for name, global_value in global_state.items():
        if not torch.is_floating_point(global_value):
            output[name] = global_value
            continue
        accumulator = torch.zeros_like(global_value, dtype=torch.float32)
        for weight, state, client_id in zip(normalized, client_states, client_ids, strict=True):
            value = state[name]
            if client_id in attack_client_ids:
                value = global_value + float(attack_factor) * (value - global_value)
                if client_id not in attacked:
                    attacked.append(client_id)
            accumulator.add_(value.float(), alpha=float(weight))
        output[name] = accumulator.to(dtype=global_value.dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(output, str(out_dir / "model.safetensors"))
    manifest = {
        "mode": "weighted_fedavg",
        "global_checkpoint": str(global_checkpoint),
        "clients": [
            {"client_id": client_id, "checkpoint": str(path), "weight": float(weight)}
            for client_id, path, weight in zip(client_ids, client_checkpoints, normalized, strict=True)
        ],
        "attack_client_ids": sorted(attacked),
        "attack_factor": float(attack_factor),
    }
    (out_dir / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hermessim_sog_msoftv2.yaml")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument(
        "--train-pkl",
        type=Path,
        default=None,
        help="optional local graph cache override; useful when the source disk is I/O constrained",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--initial-global-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox", "scaffold"], default="fedavg")
    parser.add_argument("--partition", choices=["iid", "medium_skew", "high_skew"], default="iid")
    parser.add_argument("--clients", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1400)
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="seed for the client split; defaults to --seed so methods can share paired partitions",
    )
    parser.add_argument("--fedprox-mu", type=float, default=1e-3)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--parallel-clients", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--attack-client", action="append", default=[])
    parser.add_argument("--attack-round", type=int, default=2)
    parser.add_argument("--attack-factor", type=float, default=-5.0)
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="additional dot-list training override appended to every local client command",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.clients <= 0 or args.rounds <= 0 or args.local_steps <= 0:
        raise ValueError("clients, rounds, and local_steps must be positive")
    if args.algorithm == "fedprox" and args.fedprox_mu <= 0:
        raise ValueError("FedProx requires --fedprox-mu > 0")
    if args.algorithm != "fedprox" and args.fedprox_mu != 1e-3:
        print("warning: --fedprox-mu is ignored unless --algorithm=fedprox", file=sys.stderr)

    root = Path.cwd()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    checkpoint_root = args.checkpoint_root if args.checkpoint_root.is_absolute() else root / args.checkpoint_root
    train_csv = args.train_csv if args.train_csv.is_absolute() else root / args.train_csv
    train_pkl = None
    if args.train_pkl is not None:
        train_pkl = args.train_pkl if args.train_pkl.is_absolute() else root / args.train_pkl
        if not train_pkl.exists():
            raise FileNotFoundError(train_pkl)
    initial = args.initial_global_checkpoint if args.initial_global_checkpoint.is_absolute() else root / args.initial_global_checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not (initial / "model.safetensors").exists():
        raise FileNotFoundError(initial / "model.safetensors")

    run_name = out_dir.name
    partition_seed = args.seed if args.partition_seed is None else args.partition_seed
    partitions = build_partitions(
        train_csv,
        out_dir / "client_partitions",
        args.clients,
        partition_seed,
        args.partition,
    )
    devices = parse_devices(args.cuda_devices)
    parallel = min(args.parallel_clients, len(partitions))
    attack_ids = set(args.attack_client)
    unknown_attack = attack_ids - {p.client_id for p in partitions}
    if unknown_attack:
        raise ValueError(f"unknown attack clients: {sorted(unknown_attack)}")
    weights = [p.weight for p in partitions]

    controls_dir = out_dir / "scaffold_controls"
    if args.algorithm == "scaffold":
        zero_controls(initial, controls_dir / "global_round_000.pt")
        for partition in partitions:
            zero_controls(initial, controls_dir / f"{partition.client_id}_round_000.pt")

    global_checkpoint = initial
    round_records: list[dict[str, Any]] = []
    all_jobs: list[dict[str, Any]] = []
    started = time.time()
    for round_index in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[dict[str, Any]] = []
        client_old_controls: list[Path] = []
        client_new_controls: list[Path] = []
        for offset, partition in enumerate(partitions):
            local_name = f"{run_name}_round_{round_index:03d}_{partition.client_id}"
            seed = args.seed + round_index * 10000 + offset
            command = build_local_train_command(
                config=str(args.config),
                run_name=local_name,
                partition=partition,
                max_steps=args.local_steps,
                seed=seed,
                save_every_steps=args.local_steps,
                init_model_from=global_checkpoint,
                python_bin=args.python_bin,
            )
            if args.algorithm == "fedprox":
                command.append(f"train.fedprox_mu={args.fedprox_mu}")
            if train_pkl is not None:
                command.append(f"data.train_pkl={train_pkl}")
            command.append(f"paths.checkpoint_dir={checkpoint_root}")
            command.extend(args.train_override)
            if args.algorithm == "scaffold":
                old_global = controls_dir / f"global_round_{round_index - 1:03d}.pt"
                old_client = controls_dir / f"{partition.client_id}_round_{round_index - 1:03d}.pt"
                new_client = controls_dir / f"{partition.client_id}_round_{round_index:03d}.pt"
                command.extend(
                    [
                        f"train.scaffold_c_global_path={old_global}",
                        f"train.scaffold_c_client_path={old_client}",
                        f"train.scaffold_c_out_path={new_client}",
                    ]
                )
                client_old_controls.append(old_client)
                client_new_controls.append(new_client)
            device = devices[offset % len(devices)] if devices else None
            job = {
                "round": round_index,
                "client_id": partition.client_id,
                "command": command,
                "cuda_device": device,
                "log_path": str(round_dir / "client_logs" / f"{partition.client_id}.log"),
            }
            jobs.append(job)
            all_jobs.append(job)
        runnable_jobs = []
        for job, partition in zip(jobs, partitions, strict=True):
            expected = (
                checkpoint_root
                / f"{run_name}_round_{round_index:03d}_{partition.client_id}"
                / f"step_{args.local_steps:09d}"
                / "model.safetensors"
            )
            scaffold_ready = True
            if args.algorithm == "scaffold":
                scaffold_ready = (
                    controls_dir / f"{partition.client_id}_round_{round_index:03d}.pt"
                ).is_file()
            if expected.is_file() and scaffold_ready:
                job["reused_complete_checkpoint"] = True
                print(f"reuse complete client checkpoint: {expected.parent}", flush=True)
            else:
                runnable_jobs.append(job)
        run_jobs(runnable_jobs, parallel, args.dry_run)

        client_checkpoints = [
            checkpoint_root / f"{run_name}_round_{round_index:03d}_{p.client_id}" / f"step_{args.local_steps:09d}"
            for p in partitions
        ]
        attack_this_round = attack_ids if round_index == args.attack_round else set()
        global_out = out_dir / "global" / f"round_{round_index:03d}"
        aggregation = aggregate_states(
            global_checkpoint,
            client_checkpoints,
            weights,
            global_out,
            attack_client_ids=attack_this_round,
            attack_factor=args.attack_factor,
            client_ids=[p.client_id for p in partitions],
        ) if not args.dry_run else {"dry_run": True}
        if args.algorithm == "scaffold" and not args.dry_run:
            update_global_control(
                controls_dir / f"global_round_{round_index - 1:03d}.pt",
                client_old_controls,
                client_new_controls,
                list(np.asarray(weights) / np.sum(weights)),
                controls_dir / f"global_round_{round_index:03d}.pt",
            )
        global_checkpoint = global_out
        round_records.append(
            {
                "round": round_index,
                "global_checkpoint": str(global_checkpoint),
                "client_checkpoints": [str(path) for path in client_checkpoints],
                "aggregation": aggregation,
                "elapsed_seconds": time.time() - started,
            }
        )

    manifest = {
        "mode": "reviewer_federated_experiment",
        "protocol_version": "reviewer_protocol_v2",
        "algorithm": args.algorithm,
        "config": str(args.config),
        "train_csv": str(train_csv),
        "train_pkl": str(train_pkl) if train_pkl is not None else None,
        "partition": args.partition,
        "seed": args.seed,
        "partition_seed": partition_seed,
        "training_seed": args.seed,
        "clients": args.clients,
        "rounds": args.rounds,
        "local_steps": args.local_steps,
        "fedprox_mu": args.fedprox_mu if args.algorithm == "fedprox" else None,
        "initial_global_checkpoint": str(initial),
        "client_weights": {p.client_id: p.weight for p in partitions},
        "cuda_devices": devices,
        "parallel_clients": parallel,
        "attack_client_ids": sorted(attack_ids),
        "attack_round": args.attack_round if attack_ids else None,
        "attack_factor": args.attack_factor if attack_ids else None,
        "train_overrides": args.train_override,
        "dry_run": args.dry_run,
        "final_global_checkpoint": str(global_checkpoint),
        "rounds_records": round_records,
        "jobs": all_jobs,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "fed_experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
