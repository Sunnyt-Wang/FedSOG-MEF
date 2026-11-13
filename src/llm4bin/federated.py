from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ClientPartition:
    client_id: str
    csv_path: Path
    num_rows: int
    weight: float


def _client_id(index: int) -> str:
    return f"client_{index:02d}"


def _partition_iid(df: pd.DataFrame, num_clients: int, seed: int) -> list[pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [shuffled.iloc[i::num_clients].reset_index(drop=True) for i in range(num_clients)]


def _partition_by_group(df: pd.DataFrame, num_clients: int, group_by: str, seed: int) -> list[pd.DataFrame]:
    if group_by not in df.columns:
        raise ValueError(f"missing group_by column {group_by!r}")
    groups = list(df.groupby(group_by, sort=True))
    rng = pd.Series(range(len(groups))).sample(frac=1.0, random_state=seed).tolist()
    buckets: list[list[pd.DataFrame]] = [[] for _ in range(num_clients)]
    bucket_sizes = [0 for _ in range(num_clients)]
    for group_index in rng:
        _name, frame = groups[group_index]
        target = min(range(num_clients), key=lambda i: bucket_sizes[i])
        buckets[target].append(frame)
        bucket_sizes[target] += len(frame)
    return [
        pd.concat(bucket, ignore_index=True) if bucket else df.iloc[0:0].copy()
        for bucket in buckets
    ]


def partition_csv_by_client(
    train_csv: str | Path,
    out_dir: str | Path,
    num_clients: int,
    seed: int,
    group_by: str | None = None,
) -> list[ClientPartition]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    train_csv = Path(train_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(train_csv)
    if df.empty:
        raise ValueError(f"cannot partition empty training csv: {train_csv}")

    frames = (
        _partition_by_group(df, num_clients, group_by, seed)
        if group_by is not None
        else _partition_iid(df, num_clients, seed)
    )
    partitions: list[ClientPartition] = []
    for index, frame in enumerate(frames):
        cid = _client_id(index)
        client_dir = out_dir / cid
        client_dir.mkdir(parents=True, exist_ok=True)
        csv_path = client_dir / "train.csv"
        frame.to_csv(csv_path, index=False)
        partitions.append(ClientPartition(cid, csv_path, len(frame), float(len(frame))))
    manifest = {
        "source_csv": str(train_csv),
        "num_clients": num_clients,
        "seed": seed,
        "group_by": group_by,
        "clients": [_json_partition(p) for p in partitions],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "partition_manifest.json").write_text(json.dumps(manifest, indent=2))
    return partitions


def _json_partition(partition: ClientPartition) -> dict[str, object]:
    data = asdict(partition)
    data["csv_path"] = str(partition.csv_path)
    return data


def build_local_train_command(
    *,
    config: str,
    run_name: str,
    partition: ClientPartition,
    max_steps: int,
    seed: int,
    save_every_steps: int | None = None,
    init_model_from: str | Path | None = None,
    python_bin: str = "python",
) -> list[str]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    save_every_steps = max_steps if save_every_steps is None else save_every_steps
    cmd = [python_bin, "-m", "llm4bin.cli.train", "--config", config]
    if init_model_from is not None:
        cmd.extend(["--init_model_from", str(init_model_from)])
    cmd.extend(
        [
            f"run_name={run_name}",
            f"data.train_csv={partition.csv_path}",
            f"train.max_steps={max_steps}",
            f"train.seed={seed}",
            f"logging.save_every_steps={save_every_steps}",
            "logging.keep_last_n=1",
            "train.distributed_global_batch=false",
            "train.gather_distributed_loss=false",
            "train.continuous_sampler_rng=true",
        ]
    )
    return cmd


def local_seed_for_round(base_seed: int, round_index: int, client_offset: int) -> int:
    if round_index <= 0:
        raise ValueError("round_index must be positive")
    if client_offset < 0:
        raise ValueError("client_offset must be non-negative")
    if round_index == 1:
        return base_seed
    return base_seed + (round_index * 1000) + client_offset


def client_run_name(round_index: int, partition: ClientPartition, run_prefix: str | None = None) -> str:
    name = f"round_{round_index:03d}_{partition.client_id}"
    return f"{run_prefix}_{name}" if run_prefix else name


def checkpoint_for_client(
    checkpoint_root: str | Path,
    round_index: int,
    partition: ClientPartition,
    local_steps: int,
    run_prefix: str | None = None,
) -> Path:
    return Path(checkpoint_root) / client_run_name(round_index, partition, run_prefix) / f"step_{local_steps:09d}"


def build_fedavg_round(
    *,
    round_index: int,
    partitions: list[ClientPartition],
    checkpoint_root: str | Path,
    out_dir: str | Path,
    local_steps: int,
    run_prefix: str | None = None,
) -> list[str]:
    if not partitions:
        raise ValueError("at least one client partition is required")
    if local_steps <= 0:
        raise ValueError("local_steps must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        f"{checkpoint_for_client(checkpoint_root, round_index, partition, local_steps, run_prefix)}:{partition.weight}"
        for partition in partitions
    ]
    manifest = {
        "round_index": round_index,
        "aggregation": "FedAvg",
        "local_steps": local_steps,
        "clients": [
            {
                **_json_partition(partition),
                "checkpoint": str(checkpoint_for_client(checkpoint_root, round_index, partition, local_steps, run_prefix)),
            }
            for partition in partitions
        ],
    }
    (out_dir / "round_manifest.json").write_text(json.dumps(manifest, indent=2))
    return specs
