#!/usr/bin/env python
"""Run the Dataset-1 main protocol with explicit seeds, pools, and fusion rules."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "msoftv2": ("configs/hermessim_sog_msoftv2.yaml", 19),
    "lightsog": ("configs/hermessim_sog_light.yaml", 4),
}


def stage_commands(stage: str, seed: int, cuda_device: str) -> list[list[str]]:
    checkpoints = Path("outputs/checkpoints") / f"seed{seed}"
    reports = Path("outputs/reports") / f"seed{seed}"
    predictions = Path("outputs/predictions")
    initializers = Path("outputs/checkpoints/initializers")
    commands: list[list[str]] = []

    if stage == "train":
        for family, (config, _) in FAMILIES.items():
            stem = f"{family}_seed{seed}"
            commands.append([
                sys.executable, "scripts/run_reviewer_federated_experiment.py",
                "--config", config,
                "--algorithm", "fedavg", "--partition", "iid",
                "--clients", "25", "--rounds", "2", "--local-steps", "500",
                "--seed", str(seed), "--partition-seed", str(seed),
                "--initial-global-checkpoint", str(initializers / family / "step_000030000"),
                "--out-dir", str(reports / stem),
                "--checkpoint-root", str(checkpoints / family),
                "--cuda-devices", cuda_device, "--parallel-clients", "1",
                "--train-override", "train.batch_size=2",
            ])
        commands.append([
            sys.executable, "scripts/run_centralized_matched_ensemble.py",
            "--config", "configs/centralized_msoftv2.yaml",
            "--initial-checkpoint", str(initializers / "msoftv2/step_000030000"),
            "--models", "20", "--steps", "1000", "--base-seed", str(seed * 100),
            "--checkpoint-root", str(checkpoints / "central20"),
            "--report-dir", str(reports / "central20"),
            "--cuda-devices", cuda_device, "--parallel", "1",
            "--train-override", "train.batch_size=13",
        ])
    elif stage == "evaluate":
        for family, (config, retained_locals) in FAMILIES.items():
            stem = f"{family}_seed{seed}"
            commands.append([
                sys.executable, "scripts/run_fl_expert_pool_eval.py",
                "--stem", stem, "--report-dir", str(reports / stem),
                "--checkpoint-root", str(checkpoints / family),
                "--rounds", "2", "--local-steps", "500",
                "--num-local-experts", str(retained_locals),
                "--eval-config", config, "--cuda-devices", cuda_device,
                "--accelerate-config", "configs/accelerate/single_gpu.yaml",
            ])
        for index in range(20):
            member = f"central_msoftv2_model{index:02d}_seed{seed * 100 + index}"
            commands.append([
                sys.executable, "-m", "llm4bin.cli.evaluate",
                "--config", "configs/centralized_msoftv2.yaml",
                "--ckpt", str(checkpoints / "central20" / member / "step_000001000"),
                f"run_name=central20_seed{seed}_model{index:02d}_main_eval",
            ])
    elif stage == "fuse":
        pools = {}
        for family, (_, retained_locals) in FAMILIES.items():
            stem = f"{family}_seed{seed}"
            pools[family] = [predictions / f"{stem}_global_main_eval"] + [
                predictions / f"{stem}_client{index:02d}_main_eval"
                for index in range(retained_locals)
            ]
        configurations = {
            "fedavg_global": pools["msoftv2"][:1],
            "20M": pools["msoftv2"],
            "15M5L": pools["msoftv2"][:15] + pools["lightsog"][:5],
            "central20": [
                predictions / f"central20_seed{seed}_model{index:02d}_main_eval"
                for index in range(20)
            ],
        }
        for name, members in configurations.items():
            command = [
                sys.executable, "scripts/combine_hermessim_fullpool_scores.py",
                "--out-dir", str(reports / "fusion" / name),
                "--score-norm", "rank_percentile", "--method", "max",
            ]
            for member in members:
                command.extend(["--pred-dir", str(member), "--model-name", member.name])
            commands.append(command)
    else:
        raise ValueError(f"unknown stage: {stage}")
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train", "evaluate", "fuse", "all"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1400, 1405)))
    parser.add_argument("--cuda-device", default="0", help="one GPU for sequential execution")
    parser.add_argument("--dry-run", action="store_true", help="print commands without accessing data or starting jobs")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be distinct nonnegative integers")
    if not args.cuda_device.strip() or "," in args.cuda_device:
        parser.error("--cuda-device takes one device; use the lower-level runners for parallel jobs")

    stages = ("train", "evaluate", "fuse") if args.stage == "all" else (args.stage,)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    for seed in args.seeds:
        for stage in stages:
            for command in stage_commands(stage, seed, args.cuda_device):
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
