#!/usr/bin/env python
"""Train independent centralized MSoftV2 members from a shared initializer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/centralized_msoftv2.yaml")
    parser.add_argument("--train-csv", type=Path, default=Path(os.environ.get("FEDSOG_DATA_ROOT", "data")) / "Dataset-1/training_Dataset-1.csv")
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("outputs/checkpoints/initializers/msoftv2/step_000030000"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints/reviewer_protocol_v2/centralized_20_models"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/reports/reviewer_protocol_v2/centralized_20_models"))
    parser.add_argument("--models", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=1400)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--train-override", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = ROOT
    train_csv = args.train_csv if args.train_csv.is_absolute() else root / args.train_csv
    initial = args.initial_checkpoint if args.initial_checkpoint.is_absolute() else root / args.initial_checkpoint
    checkpoint_root = args.checkpoint_root if args.checkpoint_root.is_absolute() else root / args.checkpoint_root
    report_dir = args.report_dir if args.report_dir.is_absolute() else root / args.report_dir
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not train_csv.exists() or not (initial / "model.safetensors").exists():
        raise FileNotFoundError("training CSV or initial checkpoint is missing")

    def run(index: int) -> str:
        seed = args.base_seed + index
        run_name = f"central_msoftv2_model{index:02d}_seed{seed}"
        checkpoint = checkpoint_root / run_name / f"step_{args.steps:09d}" / "model.safetensors"
        log_path = report_dir / f"model{index:02d}.log"
        if checkpoint.exists() and not args.force:
            return f"[skip complete] {run_name}"
        device = devices[index % len(devices)] if devices else None
        cmd = [
            sys.executable,
            "-m",
            "llm4bin.cli.train",
            "--config",
            args.config,
            "--init_model_from",
            str(initial),
            f"run_name={run_name}",
            f"data.train_csv={train_csv}",
            f"paths.checkpoint_dir={checkpoint_root}",
            f"train.max_steps={args.steps}",
            f"train.seed={seed}",
            f"logging.save_every_steps={args.steps}",
            "logging.keep_last_n=1",
            "train.distributed_global_batch=false",
            "train.gather_distributed_loss=false",
            "train.continuous_sampler_rng=true",
        ]
        for value in args.train_override:
            cmd.append(value)
        env = __import__("os").environ.copy()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        with log_path.open("w") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return f"[complete] {run_name}"

    with ThreadPoolExecutor(max_workers=min(args.parallel, args.models)) as executor:
        futures = [executor.submit(run, index) for index in range(args.models)]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    manifest = {
        "protocol_version": "reviewer_protocol_v2",
        "control": f"{args.models} centralized MSoftV2 models",
        "models": args.models,
        "steps_per_model": args.steps,
        "total_model_steps": args.models * args.steps,
        "centralized_budget": f"{args.models} models x {args.steps} steps = {args.models * args.steps} model-steps",
        "initial_checkpoint": str(initial),
        "train_csv": str(train_csv),
        "seeds": [args.base_seed + i for i in range(args.models)],
        "checkpoint_root": str(checkpoint_root),
        "train_overrides": args.train_override,
    }
    (report_dir / "centralized_ensemble_manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
