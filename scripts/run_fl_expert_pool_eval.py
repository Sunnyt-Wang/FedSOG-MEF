#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
PREDICTIONS = ROOT / "outputs" / "predictions"


REQUIRED_PAIR_OUTPUTS = [
    "testing_embeddings.pt",
    "xa-1000-8192_Ds1_MRR_Recall_max.csv",
    "xc-1000-8192_Ds1_MRR_Recall_max.csv",
    "xm-1000-10000_Ds1_MRR_Recall_max.csv",
    "xo-1000-8192_Ds1_MRR_Recall_max.csv",
]


def _prediction_ready(run_name: str) -> bool:
    pair_dir = PREDICTIONS / run_name / "testing_pairs"
    return all((pair_dir / name).exists() for name in REQUIRED_PAIR_OUTPUTS)


def _eval_one(
    *,
    ckpt: Path,
    run_name: str,
    eval_config: str,
    accelerate_config: str,
    cuda_devices: str,
    master_port: str,
    force: bool,
    dry_run: bool,
) -> Path:
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    pred_dir = PREDICTIONS / run_name
    if _prediction_ready(run_name) and not force:
        print(f"skip existing prediction: {pred_dir}", flush=True)
        return pred_dir

    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "1")
    env.setdefault("NCCL_ALGO", "Ring")
    env.setdefault("NCCL_PROTO", "Simple")
    env.setdefault("NCCL_SOCKET_IFNAME", "lo")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = master_port
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config,
        "-m",
        "llm4bin.cli.evaluate",
        "--config",
        eval_config,
        "--ckpt",
        str(ckpt),
        f"run_name={run_name}",
    ]
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return pred_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate final FedAvg global/local checkpoints as expert pools.")
    parser.add_argument("--stem", required=True, help="FedAvg run prefix/stem.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Defaults to outputs/reports/<stem>.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=CHECKPOINTS,
        help="Root containing final-round local checkpoints; defaults to outputs/checkpoints.",
    )
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-steps", type=int, required=True)
    parser.add_argument("--num-local-experts", type=int, required=True, help="Evaluate client_00..client_N-1.")
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--pred-prefix", default=None, help="Defaults to <stem>.")
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--accelerate-config", default="configs/accelerate/single_gpu.yaml")
    parser.add_argument("--master-port", default="29652")
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_local_experts < 0:
        raise ValueError("--num-local-experts must be non-negative")

    report_dir = args.report_dir or (REPORTS / args.stem)
    pred_prefix = args.pred_prefix or args.stem
    checkpoint_root = args.checkpoint_root if args.checkpoint_root.is_absolute() else ROOT / args.checkpoint_root

    if not args.skip_global:
        global_ckpt = report_dir / "global" / f"round_{args.rounds:03d}"
        _eval_one(
            ckpt=global_ckpt,
            run_name=f"{pred_prefix}_global_main_eval",
            eval_config=args.eval_config,
            accelerate_config=args.accelerate_config,
            cuda_devices=args.cuda_devices,
            master_port=args.master_port,
            force=args.force,
            dry_run=args.dry_run,
        )

    for client_idx in range(args.num_local_experts):
        ckpt = (
            checkpoint_root
            / f"{args.stem}_round_{args.rounds:03d}_client_{client_idx:02d}"
            / f"step_{args.local_steps:09d}"
        )
        _eval_one(
            ckpt=ckpt,
            run_name=f"{pred_prefix}_client{client_idx:02d}_main_eval",
            eval_config=args.eval_config,
            accelerate_config=args.accelerate_config,
            cuda_devices=args.cuda_devices,
            master_port=args.master_port,
            force=args.force,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
