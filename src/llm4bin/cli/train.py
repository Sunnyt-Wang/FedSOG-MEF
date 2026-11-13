from __future__ import annotations

import argparse
import sys

from ..config import load_config, validate_paths, pretty
from ..trainer.similarity_trainer import SimilarityTrainer
from ..trainer.trainer import VulnTrainer
from ..utils.logger import get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="llm4bin-train")
    p.add_argument("--config", required=True, help="path to YAML config file")
    p.add_argument(
        "--resume_from",
        default=None,
        help='checkpoint dir or "auto" (latest in the run\'s checkpoint_dir)',
    )
    p.add_argument(
        "--init_model_from",
        default=None,
        help="checkpoint dir or model.safetensors used only to initialize model weights",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides (e.g. train.lr=1e-4)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_from is not None and args.init_model_from is not None:
        raise ValueError("--resume_from and --init_model_from are mutually exclusive")
    cfg = load_config(args.config, overrides=args.overrides)
    validate_paths(cfg)

    logger = get_logger("cli.train", log_dir=cfg.paths.log_dir)
    logger.info(f"loaded config:\n{pretty(cfg)}")

    if cfg.train.task == "similarity":
        trainer = SimilarityTrainer(cfg, resume_from=args.resume_from, init_model_from=args.init_model_from)
    else:
        trainer = VulnTrainer(cfg, resume_from=args.resume_from)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
