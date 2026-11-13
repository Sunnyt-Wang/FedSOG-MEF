#!/usr/bin/env python
"""Run a resumable five-seed sweep for the reviewer FL protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [1400, 1401, 1402, 1403, 1404]


def complete_run(report_dir: Path, checkpoint_root: Path, clients: int, rounds: int, steps: int) -> bool:
    manifest = report_dir / "fed_experiment_manifest.json"
    final_global = report_dir / "global" / f"round_{rounds:03d}" / "model.safetensors"
    if not manifest.exists() or not final_global.exists():
        return False
    for client in range(clients):
        path = checkpoint_root / f"{report_dir.name}_round_{rounds:03d}_client_{client:02d}" / f"step_{steps:09d}" / "model.safetensors"
        if not path.exists():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["iid", "medium_skew", "high_skew"], required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox", "scaffold"], default="fedavg")
    parser.add_argument("--config", default="configs/hermessim_sog_msoftv2.yaml")
    parser.add_argument("--initial-global-checkpoint", type=Path, default=Path("outputs/checkpoints/initializers/msoftv2/step_000030000"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs/reports/reviewer_protocol_v2"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints/reviewer_protocol_v2"))
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--parallel-clients", type=int, default=1)
    parser.add_argument("--fedprox-mu", type=float, default=1e-3)
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="additional dot-list override appended to every local training command",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_root = args.report_root if args.report_root.is_absolute() else ROOT / args.report_root
    checkpoint_root = args.checkpoint_root if args.checkpoint_root.is_absolute() else ROOT / args.checkpoint_root
    initial = args.initial_global_checkpoint if args.initial_global_checkpoint.is_absolute() else ROOT / args.initial_global_checkpoint
    report_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not (initial / "model.safetensors").exists():
        raise FileNotFoundError(initial / "model.safetensors")

    runner = ROOT / "scripts/run_reviewer_federated_experiment.py"
    for seed in args.seeds:
        report_dir = report_root / f"{args.algorithm}_{args.partition}_{args.clients}client_seed{seed}"
        if complete_run(report_dir, checkpoint_root, args.clients, args.rounds, args.local_steps):
            print(f"[skip complete] {report_dir}", flush=True)
            continue
        command = [
            sys.executable,
            str(runner),
            "--config", args.config,
            "--out-dir", str(report_dir),
            "--checkpoint-root", str(checkpoint_root),
            "--initial-global-checkpoint", str(initial),
            "--algorithm", args.algorithm,
            "--partition", args.partition,
            "--clients", str(args.clients),
            "--rounds", str(args.rounds),
            "--local-steps", str(args.local_steps),
            "--seed", str(seed),
            "--partition-seed", str(seed),
            "--cuda-devices", args.cuda_devices,
            "--parallel-clients", str(args.parallel_clients),
            "--fedprox-mu", str(args.fedprox_mu),
        ]
        for value in args.train_override:
            command.extend(["--train-override", value])
        log_path = report_dir / "sweep.log"
        report_dir.mkdir(parents=True, exist_ok=True)
        print("[run]", " ".join(command), flush=True)
        if args.dry_run:
            continue
        with log_path.open("w") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[complete] {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
