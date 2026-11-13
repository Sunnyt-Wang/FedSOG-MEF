from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import load_config, validate_paths
from ..eval.evaluator import VulnEvaluator
from ..eval.similarity_evaluator import SimilarityEvaluator
from ..utils.logger import get_logger


def _resolve_ckpt(run_dir: Path, target: str) -> Path:
    if target != "auto":
        return Path(target)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no manifest at {manifest}")
    data = json.loads(manifest.read_text())
    entries = data.get("entries", [])
    if not entries:
        raise RuntimeError(f"empty manifest at {manifest}")
    return Path(max(entries, key=lambda e: e["step"])["dir"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="llm4bin-eval")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="auto", help='checkpoint dir or "auto"')
    p.add_argument("overrides", nargs="*")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    validate_paths(cfg)

    run_dir = Path(cfg.paths.checkpoint_dir) / cfg.run_name
    ckpt_dir = _resolve_ckpt(run_dir, args.ckpt)

    logger = get_logger("cli.evaluate", log_dir=cfg.paths.log_dir)
    logger.info(f"evaluating with checkpoint: {ckpt_dir}")

    if cfg.train.task == "similarity":
        evaluator = SimilarityEvaluator(cfg, ckpt_dir=str(ckpt_dir))
        evaluator.run()
        evaluator.run_testing_pairs()
    else:
        evaluator = VulnEvaluator(cfg, ckpt_dir=str(ckpt_dir))
        evaluator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
