# Reproduction Guide

## Release Scope

This checkout packages the Dataset-1 graph encoders, federated training, fixed
expert pools, centralized control, and reference fusion implementation. The
original research package name and reusable lower-level script names are kept.
The default paths and launch parameters have been adapted for this standalone
checkout. Experimental result tables and machine-specific job queues are not
distributed as measured results in this release.

The provided workflow materializes embeddings for transparency. Historical
experiments also used streaming CUDA scoring with different embedding batch
sizes. Floating-point dot products and ordinal ordering of equal scores can
change ties; this checkout has not been certified to reproduce every published
number bit for bit.

## Expected Files

```text
data/
  Dataset-1/
    training_Dataset-1.csv
    testing_Dataset-1.csv
    pairs/
      validation/validation_functions.csv
      experiments/full/
        pos-xa-1000-8192_Ds1.csv
        neg-xa-1000-8192_Ds1.csv
        pos-xc-1000-8192_Ds1.csv
        neg-xc-1000-8192_Ds1.csv
        pos-xm-1000-10000_Ds1.csv
        neg-xm-1000-10000_Ds1.csv
        pos-xo-1000-8192_Ds1.csv
        neg-xo-1000-8192_Ds1.csv
  graphs/graph_func_dict.pkl
outputs/checkpoints/initializers/
  msoftv2/step_000030000/model.safetensors
  lightsog/step_000030000/model.safetensors
```

`FEDSOG_DATA_ROOT` replaces `data/`; `FEDSOG_GRAPH_CACHE` replaces the graph-cache
path. Run commands from the checkout root. The repository does not download data
automatically. Consult HermesSim for binary processing and SOG extraction.

Training, validation, and testing use separate CSV selections over the graph
cache. Sharing a storage file does not add validation or test rows to training.
The Dataset-1 split uses ClamAV, Curl, OpenSSL, and UnRAR for training, zlib for
validation, and nmap and z3 for testing. Preserve the upstream split and pair
files when reproducing the protocol.

## Initializers

M and L use separate graph encoders. Each family starts from one shared
30,000-update initializer, with initialization seeds 14 and 41 respectively.
All centralized M members share the M initializer. Existing compatible checkpoints
can be placed at the paths above. To train with the recorded initializer schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file configs/accelerate/ddp_6gpu.yaml \
  -m llm4bin.cli.train --config configs/initializer_msoftv2.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file configs/accelerate/ddp_6gpu.yaml \
  -m llm4bin.cli.train --config configs/initializer_light.yaml
```

These initializers use 13 function groups per GPU on six GPUs, two graph
variants per group, and distributed loss gathering. Changing the GPU count
changes the global training batch. Initializer training is separate from the
1,000-update client/member schedules in the main experiment.

## Training and Selection

Use `scripts/run_paper.py` for the main IID protocol. The federated runner
randomly permutes training rows with the partition seed and assigns them across
25 clients. Each of two rounds performs 500 local updates, followed by FedAvg
weighted by client row counts. The local training seed is
`base_seed + 10000 * round + client_index`, with rounds starting at one.
Centralized member seeds are `100 * base_seed + member_index`.

Both encoder families use Adam, learning rate 0.001, a constant schedule, zero
weight decay, gradient clipping at 10, and no mixed precision. The local batch
has two function groups and the centralized batch has thirteen; each group
provides two graph variants. The objective is the existing cosine/circle batch
pair loss with margin 0.1, scale 1e7, and graph-vector regularizer weight 1e-8.

MSoftV2 produces 384-dimensional embeddings using six attention queries.
LightSOG produces 256-dimensional embeddings with hidden size 96, three layers,
and dropout 0.1. See the YAML files for the complete architecture parameters.

Both 25-client families are trained in full before constructing 15M+5L. The
released runner uses fixed client identities for retention. Changing the number
of retained experts only changes evaluation; it does not erase the cost of
training the candidate pools.

The lower-level runner also accepts `--algorithm fedprox` or `scaffold` and
`--partition medium_skew` or `high_skew`. Those are separate conditions from the
main IID configuration. The older `run_fedavg_light_sog.py` helper is retained
for compatibility tests and has a different seed schedule; use the paper runner
for the main experiment.

## Evaluation and Fusion

The `evaluate` stage creates validation and testing embeddings for the retained
global/local models and all twenty centralized members. This representation can
consume substantial disk space and memory. Generated artifacts are ignored by
Git.

The `fuse` stage evaluates four declared pools: FedAvg global, 20M, 15M+5L, and
centralized 20. It explicitly fixes `--score-norm rank_percentile --method max`.
No search across fusion rules is performed by this workflow.

For each query and expert, the scorer constructs cosine similarities to the
positive and negative candidates. Ordinal ranks are obtained with NumPy's
descending `argsort`; normalized scores are `1 - rank / (pool_size - 1)` for
zero-based ranks. Maximum fusion is applied across experts. The final positive
rank is `1 + count(negative_score > positive_score)`, so a fused-score tie does
not outrank the positive. This explicit tie rule is part of the reference
implementation. The diagnostic per-model CSVs from the embedding evaluator use
the separately configured `rank_method=max` convention.

Pair files must match the embedding row order exactly. The combiner reads each
positive CSV and its matching `neg-` CSV, retains the declared number of
negatives, and records model identities and input paths in a manifest.

## Seeds and Summaries

The original main experiment uses base seeds 1400--1404. The additional training
replications use 1405--1409; pass all ten explicitly to execute the expanded plan.
The earlier 100 sampled expert subsets are selection-sensitivity configurations,
not 100 independent training seeds.

For seed-level summaries, first average XA, XC, XM, and XO within each base seed,
then summarize across base seeds. Pair methods by the same base seed for paired
comparisons. Do not treat individual tasks or expert subsets as independent
training repetitions.

## Local Checks

```bash
python -m pytest -q
python scripts/run_paper.py --stage all --seeds 1400 --dry-run
```

The tests use synthetic data and do not launch full training. Passing them checks
the packaged implementation and its declared command plan; it does not replace
an experiment using the original data and initializers.
