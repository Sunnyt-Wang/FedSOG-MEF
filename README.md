# FedSOG-MEF

Code for **A Federated Multi-Expert Fusion Framework for Cross-Architecture
Binary Code Similarity Detection Using Semantics-Oriented Graphs**.

The implementation trains MSoftV2 and LightSOG graph encoders with federated
averaging, retains global and local checkpoints, and combines per-expert rank
scores by maximum fusion. It also includes a centralized ensemble control.
The Python package keeps the historical `llm4bin` import name for compatibility.

## Installation

Use Python 3.10 or newer and install a PyTorch build suitable for your CUDA
environment, then install this checkout in editable mode:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The graph-only configurations do not download or load Qwen weights. Transformer
and PEFT dependencies are retained because the shared model module imports them.
See [the tested environment](docs/environment.md) for package versions.

## Data and Initializers

Obtain Dataset-1 and its SOG preprocessing resources from
[HermesSim](https://github.com/NSSL-SJTU/HermesSim). This repository contains code
and configurations; dataset files, graph caches, and trained weights are supplied
separately. Set these paths for your local data:

```bash
export FEDSOG_DATA_ROOT=/path/to/data
export FEDSOG_GRAPH_CACHE=/path/to/graph_func_dict.pkl
```

`FEDSOG_DATA_ROOT` must contain the `Dataset-1` directory, including the training
and testing CSVs and the `pairs/` tree. The graph cache must contain the matching
HermesSim SOGs. Keep the original CSV row identifiers because the evaluation pair
files index those rows. The expected layout and initializer commands are in
[the reproduction guide](docs/reproduction.md).

## Main Experiment

Preview the complete plan without loading data, allocating a GPU, or starting
training:

```bash
python scripts/run_paper.py --stage all --seeds 1400 --dry-run
```

After preparing data and both initializers, run each stage from the repository
root:

```bash
python scripts/run_paper.py --stage train --seeds 1400 --cuda-device 0
python scripts/run_paper.py --stage evaluate --seeds 1400 --cuda-device 0
python scripts/run_paper.py --stage fuse --seeds 1400
```

Omit `--seeds` to run base seeds 1400--1404. The additional replication seeds are
1405--1409 and can be passed explicitly. Each base seed specifies client
partitioning and training after initialization; the two initializer checkpoints
are shared across base seeds.

| Configuration | Candidate models trained | Updates per client/member | Function groups per batch | Retained models |
| --- | --- | --- | --- | --- |
| FedAvg global / 20M | 25 M clients | 2 rounds x 500 | 2 | 1 / 20 |
| 15M+5L | 25 M clients and 25 L clients | 2 rounds x 500 per family | 2 | 20 |
| Centralized ensemble | 20 M members | 1,000 | 13 | 20 |

Each function group contains two graph variants. M denotes MSoftV2; L denotes
LightSOG. The main comparisons share the per-client/member update schedule of
1,000 updates, while the full candidate-pool sizes and batch sizes are as listed.

The 20M pool contains the M global model and M clients 00--18. The 15M+5L pool
contains the M global model and M clients 00--13, plus the L global model and L
clients 00--03. Local checkpoints are taken from round 2, step 500. These identities
are fixed before scoring.

## Evaluation

Evaluation uses the Dataset-1 XA, XC, XM, and XO tasks, each with 1,000 queries.
Candidate pools contain 8,192 functions for XA/XC/XO and 10,000 for XM. The fusion
stage writes Recall@1, MRR, and its selected configuration under
`outputs/reports/seed<seed>/fusion/`.

The score combiner preserves the research implementation's ordinal rank
normalization and optimistic treatment of fused-score ties. The single-encoder
CLI also writes its original `rank_method=max` metrics. Use the same scoring
path when comparing methods; see the reproduction guide for the exact rule and
the scope of this release.

## Source Layout

| Path | Purpose |
| --- | --- |
| `src/llm4bin/models/` | SOG encoders and similarity objective |
| `src/llm4bin/federated.py` | Client partitions and local training commands |
| `src/llm4bin/eval/fed_moe.py` | Rank normalization and multi-expert fusion |
| `scripts/run_paper.py` | Main Dataset-1 training, evaluation, and fusion plan |
| `scripts/run_reviewer_federated_experiment.py` | FedAvg, FedProx, and SCAFFOLD runner |
| `scripts/run_centralized_matched_ensemble.py` | Centralized member training |
| `configs/` | Main-experiment and initializer configurations |
| `tests/` | Synthetic-data checks for partitions, fusion, and graph relations |

The word `matched` in the historical centralized runner filename is not a claim
that all methods have equal total computation. Runtime manifests record the
executed configuration.

## Attribution

Dataset-1, SOG conventions, and the reference similarity setup follow HermesSim.
Upstream datasets and dependencies retain their own licenses. No repository-wide
license has been assigned to this source release yet.
