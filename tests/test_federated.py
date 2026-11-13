from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from llm4bin.federated import (
    ClientPartition,
    build_fedavg_round,
    build_local_train_command,
    local_seed_for_round,
    partition_csv_by_client,
)


def test_partition_csv_by_client_writes_disjoint_local_datasets(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    rows = pd.DataFrame(
        {
            "obj_file": [f"obj_{i}" for i in range(8)],
            "fva": [hex(i) for i in range(8)],
            "func_name": [f"func_{i}" for i in range(8)],
            "db_type": ["XA", "XC", "XM", "XO", "XA", "XC", "XM", "XO"],
        }
    )
    rows.to_csv(source, index=False)

    partitions = partition_csv_by_client(source, tmp_path / "clients", num_clients=3, seed=7)

    assert [p.client_id for p in partitions] == ["client_00", "client_01", "client_02"]
    assert sum(p.num_rows for p in partitions) == len(rows)
    assert all(p.csv_path.exists() for p in partitions)

    assigned = []
    for partition in partitions:
        frame = pd.read_csv(partition.csv_path)
        assigned.extend(frame["obj_file"].tolist())
    assert sorted(assigned) == sorted(rows["obj_file"].tolist())
    assert len(assigned) == len(set(assigned))


def test_partition_csv_by_client_can_group_by_column(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "obj_file": [f"obj_{i}" for i in range(6)],
            "fva": [hex(i) for i in range(6)],
            "site": ["arm", "arm", "x86", "x86", "mips", "mips"],
        }
    ).to_csv(source, index=False)

    partitions = partition_csv_by_client(
        source,
        tmp_path / "clients",
        num_clients=2,
        seed=3,
        group_by="site",
    )

    seen_sites: dict[str, str] = {}
    for partition in partitions:
        frame = pd.read_csv(partition.csv_path)
        for site in frame["site"].unique():
            assert site not in seen_sites
            seen_sites[site] = partition.client_id
    assert set(seen_sites) == {"arm", "x86", "mips"}


def test_build_local_train_command_uses_private_client_csv_and_global_resume(tmp_path: Path) -> None:
    partition = ClientPartition(
        client_id="client_00",
        csv_path=tmp_path / "clients" / "client_00" / "train.csv",
        num_rows=10,
        weight=10.0,
    )
    cmd = build_local_train_command(
        config="configs/hermessim_sog_light.yaml",
        run_name="fedavg_r02_client_00",
        partition=partition,
        max_steps=5,
        seed=123,
        save_every_steps=5,
        init_model_from=tmp_path / "global" / "round_001",
    )

    assert cmd[:4] == ["python", "-m", "llm4bin.cli.train", "--config"]
    assert "--init_model_from" in cmd
    assert str(tmp_path / "global" / "round_001") in cmd
    assert f"data.train_csv={partition.csv_path}" in cmd
    assert "train.max_steps=5" in cmd
    assert "logging.save_every_steps=5" in cmd
    assert "train.distributed_global_batch=false" in cmd
    assert "train.gather_distributed_loss=false" in cmd


def test_build_fedavg_round_records_weighted_clients(tmp_path: Path) -> None:
    partitions = [
        ClientPartition("client_00", tmp_path / "c0.csv", 10, 10.0),
        ClientPartition("client_01", tmp_path / "c1.csv", 30, 30.0),
    ]
    round_dir = tmp_path / "round_001"
    specs = build_fedavg_round(
        round_index=1,
        partitions=partitions,
        checkpoint_root=tmp_path / "checkpoints",
        out_dir=round_dir,
        local_steps=10,
    )

    assert specs == [
        str(tmp_path / "checkpoints" / "round_001_client_00" / "step_000000010") + ":10.0",
        str(tmp_path / "checkpoints" / "round_001_client_01" / "step_000000010") + ":30.0",
    ]
    manifest = json.loads((round_dir / "round_manifest.json").read_text())
    assert manifest["round_index"] == 1
    assert manifest["aggregation"] == "FedAvg"
    assert manifest["clients"][1]["weight"] == 30.0


def test_local_seed_for_round_keeps_initial_fedavg_model_identical() -> None:
    assert local_seed_for_round(base_seed=900, round_index=1, client_offset=0) == 900
    assert local_seed_for_round(base_seed=900, round_index=1, client_offset=1) == 900
    assert local_seed_for_round(base_seed=900, round_index=2, client_offset=0) == 2900
    assert local_seed_for_round(base_seed=900, round_index=2, client_offset=1) == 2901


def test_partition_rejects_missing_group_column(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame({"obj_file": ["a"], "fva": ["0x1"]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="missing group_by column"):
        partition_csv_by_client(source, tmp_path / "clients", num_clients=2, seed=0, group_by="compiler")


def test_fedavg_manifest_records_config_model_label(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "obj_file": [f"obj_{i}" for i in range(4)],
            "fva": [hex(i) for i in range(4)],
            "func_name": [f"func_{i}" for i in range(4)],
        }
    ).to_csv(source, index=False)

    out_dir = tmp_path / "fedavg"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_fedavg_light_sog.py",
            "--config",
            "configs/hermessim_sog_msoftv2.yaml",
            "--train-csv",
            str(source),
            "--out-dir",
            str(out_dir),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--num-clients",
            "2",
            "--rounds",
            "1",
            "--local-steps",
            "5",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out_dir / "fedavg_training_manifest.json").read_text())
    assert manifest["model"] == "hermessim_sog_msoftv2"


def test_fedavg_can_start_round_one_from_initial_global_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "obj_file": [f"obj_{i}" for i in range(4)],
            "fva": [hex(i) for i in range(4)],
            "func_name": [f"func_{i}" for i in range(4)],
        }
    ).to_csv(source, index=False)
    initial = tmp_path / "initial_global"
    initial.mkdir()

    out_dir = tmp_path / "fedavg"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_fedavg_light_sog.py",
            "--config",
            "configs/hermessim_sog_msoftv2.yaml",
            "--train-csv",
            str(source),
            "--out-dir",
            str(out_dir),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--num-clients",
            "2",
            "--rounds",
            "1",
            "--local-steps",
            "5",
            "--initial-global-checkpoint",
            str(initial),
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out_dir / "fedavg_training_manifest.json").read_text())
    assert manifest["initial_global_checkpoint"] == str(initial)
    round_one_commands = manifest["round_records"][0]["local_commands"]
    assert all("--init_model_from" in cmd for cmd in round_one_commands)
    assert all(str(initial) in cmd for cmd in round_one_commands)


def test_fedavg_dry_run_records_parallel_cuda_assignments(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "obj_file": [f"obj_{i}" for i in range(6)],
            "fva": [hex(i) for i in range(6)],
            "func_name": [f"func_{i}" for i in range(6)],
        }
    ).to_csv(source, index=False)

    out_dir = tmp_path / "fedavg"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_fedavg_light_sog.py",
            "--config",
            "configs/hermessim_sog_msoftv2.yaml",
            "--train-csv",
            str(source),
            "--out-dir",
            str(out_dir),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--num-clients",
            "3",
            "--rounds",
            "1",
            "--local-steps",
            "5",
            "--parallel-clients",
            "2",
            "--cuda-devices",
            "0,1",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out_dir / "fedavg_training_manifest.json").read_text())
    assert manifest["parallel_clients"] == 2
    assert manifest["cuda_devices"] == ["0", "1"]
    round_jobs = manifest["round_records"][0]["local_jobs"]
    assert [job["cuda_device"] for job in round_jobs] == ["0", "1", "0"]
    command_jobs = [entry for entry in manifest["commands"] if entry["client_id"] != "server"]
    assert [entry["cuda_device"] for entry in command_jobs] == ["0", "1", "0"]
