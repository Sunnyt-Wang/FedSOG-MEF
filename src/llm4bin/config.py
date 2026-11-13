from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ----- Dataclasses ----------------------------------------------------------

@dataclass
class PathsConfig:
    project_root: str
    datasets_root: str
    models_root: str
    output_dir: str
    checkpoint_dir: str
    log_dir: str
    predictions_dir: str


@dataclass
class ModelConfig:
    base_model_path: str
    similarity_arch: str = "qwen_fusion"
    image_token: str = "<image>"
    block_num: int = 5
    embed_dim: int = 32
    hidden_size: int = 2560
    similarity_dim: int = 128
    msoft_num_queries: int = 6
    msoft_hidden_channels: int = 64
    msoft_node_trans: int = 1
    msoft_aggr_trans: int = 1
    msoft_q_scale: float = 1.0
    graphormer_hidden_size: int = 128
    graphormer_num_layers: int = 4
    graphormer_num_heads: int = 4
    graphormer_ffn_dim: int = 256
    graphormer_dropout: float = 0.1
    graphormer_max_nodes: int = 128
    graphormer_max_degree: int = 128
    graphormer_max_distance: int = 16
    light_hidden_size: int = 96
    light_edge_hidden_size: int = 16
    light_num_layers: int = 3
    light_dropout: float = 0.1
    # Controlled ablation: retain SOG node attributes but remove edge-based
    # message passing/relation information from the graph encoder.
    disable_graph_relations: bool = False
    # Selectively retain SOG relations for controlled component ablations.
    graph_relation_mode: str = "full"
    freeze_llm: bool = True
    gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class GgnnEncoderConfig:
    n_node_feat_dim: int = 32
    n_edge_feat_dim: int = 8
    n_node_attr: int = 461
    n_edge_attr: int = 4
    n_pos_enc: int = 8


@dataclass
class GgnnArgsConfig:
    n_node_feat_dim: int = 32
    n_edge_feat_dim: int = 8
    layer_groups: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    n_message_net_layers: int = 3
    skip_mode: int = 0
    output_mode: int = 0
    num_query: int = 4
    n_atte_layers: int = 0
    layer_aggr: str = "add"
    layer_aggr_kwargs: dict = field(default_factory=dict)
    concat_skip: int = 0


@dataclass
class GgnnConfig:
    encoder: GgnnEncoderConfig = field(default_factory=GgnnEncoderConfig)
    args: GgnnArgsConfig = field(default_factory=GgnnArgsConfig)


@dataclass
class TrainConfig:
    task: str = "vuln"          # vuln | source | similarity
    code_type: str = "C"        # C | Java (vuln only)
    batch_size: int = 2
    num_workers: int = 0
    shuffle: bool = True
    epochs: int = 20
    max_steps: int = -1
    lr: float = 8e-5
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    mixed_precision: str = "bf16"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    grad_accum_steps: int = 3
    grad_clip: float = 1.0
    seed: int = 42
    subset_size: int = -1
    logical_batch_size: int = -1
    n_sim_funcs: int = 2
    loss: str = "cosine"
    loss_opt: str = "circle"
    loss_margin: float = 0.10
    loss_gama: float = 1e7
    norm_neg_sampling_s: float | None = 1.0
    gather_distributed_loss: bool = False
    distributed_global_batch: bool = False
    continuous_sampler_rng: bool = False
    graph_vec_regularizer_weight: float = 1e-8
    # Optional federated optimization terms.  The defaults preserve FedAvg.
    fedprox_mu: float = 0.0
    # Ditto keeps a client-specific model close to the current server model
    # while allowing the client model to remain personalized.
    ditto_mu: float = 0.0
    ditto_global_path: str | None = None
    scaffold_c_global_path: str | None = None
    scaffold_c_client_path: str | None = None
    scaffold_c_out_path: str | None = None
    max_num_nodes: int = 180000
    max_num_edges: int = 360000


@dataclass
class LoggingConfig:
    log_every: int = 10
    save_every_steps: int = 100
    eval_every_steps: int = -1
    keep_last_n: int = 3
    keep_best: bool = True


@dataclass
class EvalConfig:
    batch_size: int = 4
    max_new_tokens: int = 1
    subset_size: int = -1
    save_predictions: bool = True
    log_every_batches: int = 25
    pool_sizes: list[int] = field(default_factory=lambda: sorted([2**i for i in range(1, 14)] + [100, 10000]))
    rank_method: str = "max"
    pair_csvs: list[str] = field(default_factory=list)
    skip_pair_metrics: bool = False


@dataclass
class DataConfig:
    train_csv: str | None = None
    train_pkl: str | None = None
    validation_csv: str | None = None
    validation_pkl: str | None = None
    testing_csv: str | None = None
    testing_pkl: str | None = None
    eval_csv: str | None = None
    eval_pkl: str | None = None
    source_code_pool_pkl: str | None = None


@dataclass
class PromptConfig:
    system: str = ""
    user_prefix: str = ""
    num_classes: int = 0
    max_target_length: int = 8


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig
    ggnn: GgnnConfig
    train: TrainConfig
    logging: LoggingConfig
    eval: EvalConfig
    data: DataConfig
    prompt: PromptConfig
    run_name: str = "default"


# ----- Loader ---------------------------------------------------------------

def _merge_with_defaults(cfg: DictConfig) -> DictConfig:
    """Resolve the ``defaults`` list (a tiny subset of Hydra's behavior).

    Each entry in ``defaults`` is a name of a YAML in :data:`CONFIGS_DIR` (without
    extension). Files are merged left-to-right, then the current cfg is overlaid
    on top, so the current YAML always wins.
    """
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg
    merged = OmegaConf.create({})
    for name in defaults:
        path = CONFIGS_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"default config not found: {path}")
        sub = OmegaConf.load(path)
        sub = _merge_with_defaults(sub)
        merged = OmegaConf.merge(merged, sub)
    return OmegaConf.merge(merged, cfg)


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    raw = OmegaConf.load(path)
    raw = _merge_with_defaults(raw)
    if overrides:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(overrides))

    # Resolve ${...} interpolations.
    resolved: dict[str, Any] = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]

    # Build dataclass tree (manual to keep nested dataclasses well-typed).
    cfg = Config(
        paths=PathsConfig(**resolved["paths"]),
        model=ModelConfig(**resolved["model"]),
        ggnn=GgnnConfig(
            encoder=GgnnEncoderConfig(**resolved["ggnn"]["encoder"]),
            args=GgnnArgsConfig(**resolved["ggnn"]["args"]),
        ),
        train=TrainConfig(**resolved["train"]),
        logging=LoggingConfig(**resolved["logging"]),
        eval=EvalConfig(**resolved["eval"]),
        data=DataConfig(**resolved["data"]),
        prompt=PromptConfig(**resolved.get("prompt", {})),
        run_name=resolved.get("run_name", "default"),
    )
    return cfg


def validate_paths(cfg: Config, strict_data: bool = True) -> None:
    """Verify referenced paths exist. Output dirs are created if missing."""
    from .utils.paths import ensure_dir, require_exists

    require_exists(cfg.model.base_model_path, "model.base_model_path")
    require_exists(cfg.paths.datasets_root, "paths.datasets_root")

    ensure_dir(cfg.paths.output_dir)
    ensure_dir(cfg.paths.checkpoint_dir)
    ensure_dir(cfg.paths.log_dir)
    ensure_dir(cfg.paths.predictions_dir)

    if strict_data:
        items = [("data.train_csv", cfg.data.train_csv), ("data.train_pkl", cfg.data.train_pkl)]
        if cfg.train.task == "similarity":
            items.extend([
                ("data.validation_csv", cfg.data.validation_csv),
                ("data.validation_pkl", cfg.data.validation_pkl or cfg.data.train_pkl),
                ("data.testing_csv", cfg.data.testing_csv),
                ("data.testing_pkl", cfg.data.testing_pkl or cfg.data.validation_pkl or cfg.data.train_pkl),
            ])
            items.extend((f"eval.pair_csvs[{i}]", p) for i, p in enumerate(cfg.eval.pair_csvs))
        else:
            items.extend([
                ("data.eval_csv", cfg.data.eval_csv),
                ("data.eval_pkl", cfg.data.eval_pkl),
            ])
        if cfg.train.task == "source":
            items.append(("data.source_code_pool_pkl", cfg.data.source_code_pool_pkl))
        for label, p in items:
            if p is None:
                continue
            require_exists(p, label)


def pretty(cfg: Config) -> str:
    """Return a pretty-printed view of the config (for logging at startup)."""
    return OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True)
