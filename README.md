# GotenNet for Open Catalyst OC20 S2EF

This repository is now **OC20-first**: the primary/default path trains **GotenNet** on the
Open Catalyst 2020 Structure-to-Energy-and-Forces (S2EF) task with periodic boundary
conditions.

## What changed

- Replaced the QM9-first training workflow with a repo-local OC20 launcher.
- Added an OC20 S2EF model wrapper that uses GotenNet representation + periodic graph building.
- Added energy + force prediction (default: force from energy gradients).
- Added support for **standard OC20/FairChem LMDB datasets** in addition to the older repo-local
  `.pt` sample-list format.
- Added clearer dataset-path errors and OC20 smoke tests.

## Dependencies

Install the Python package metadata and the non-native training dependencies with:

```bash
pip install -e .[full]
pip install -r requirements.txt
```

For the PyG native extensions (`torch_scatter`, `torch_sparse`, `torch_cluster`, and `torch_spline_conv`), use the repair script below so the wheels match your existing PyTorch/CUDA build instead of pulling a stale binary.

## Environment repair

If you already have PyTorch `2.10.0+cu128` installed in your environment, repair the PyG stack with:

```bash
bash scripts/fix_pyg_env.sh
```

The script:

- prints the active Python / torch / CUDA versions
- removes stale PyG native-extension wheels
- reinstalls matching wheels from `https://data.pyg.org/whl/torch-2.10.0+cu128.html`
- runs an import smoke test for `torch_scatter`, `torch_cluster`, `torch_sparse`, and `torch_spline_conv`
- skips `pyg_lib` automatically when no matching wheel is published for this torch/CUDA combo

## Dataset formats

The OC20 runner now supports two dataset formats:

### 1. `dataset.format: lmdb` (default, recommended)

This matches the standard OC20/FairChem S2EF workflow. Point `dataset.train_path` and
`dataset.val_path` at either:

- a single LMDB file such as `train/data.lmdb`
- a single LMDB environment directory containing `data.mdb` / `lock.mdb`
- a directory containing multiple LMDB shards such as `train/0000.lmdb`, `train/0001.lmdb`, ...

The runner accepts common OC20/FairChem samples that store atom types under `z`, energy under `y`,
and forces under `force`.

### 2. `dataset.format: pt` (backward-compatible repo format)

This is the older repo-specific format. It expects `train.pt` / `val.pt` files, each containing a
Python list of dicts with at least:

- `atomic_numbers` (`[N]`, `long`)
- `pos` (`[N, 3]`, `float`)
- `cell` (`[3, 3]`, `float`)
- `pbc` (`[3]`, `bool`)
- `energy` (`[]` or `[1]`, optional for train/val)
- `forces` (`[N, 3]`, optional)
- `fixed` (`[N]`, optional bool mask)

If `.pt` files are missing, the runner now explicitly tells you that `.pt` mode is a **custom repo
format**, not the official OC20 storage format.

## Prepare / point to OC20 data

### Download standard OC20 S2EF data

Use the official Open Catalyst / FairChem dataset workflow to obtain the S2EF LMDB splits. The
Open Catalyst challenge page documents the dataset and split structure, and FairChem uses LMDB as
its standard training format.

Authoritative references:

- OC20 challenge page: <https://opencatalystproject.org/challenge.html>
- FairChem preprocessing docs: <https://fair-chem.github.io/autoapi/core/scripts/preprocess_relaxed/index.html>

### No extra preprocessing required for standard LMDB input

If you already have the official OC20 S2EF LMDB data, you can point this repo at those paths
**directly**. You do **not** need to convert LMDB into custom `train.pt` / `val.pt` files anymore.

### Example config

The default config already targets LMDB:

- `configs/oc20/s2ef/gotennet.yaml`

Example:

```yaml
dataset:
  format: lmdb
  train_path: /data/oc20/s2ef/train
  val_path: /data/oc20/s2ef/val_id
```

### Path overrides

You can choose any of these override mechanisms:

1. Edit `configs/oc20/s2ef/gotennet.yaml` directly.
2. Pass explicit CLI overrides:

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train --dataset-format lmdb --train-path /abs/path/to/train --val-path /abs/path/to/val_id --device auto
```

3. Export environment variables before launching:

```bash
export OC20_TRAIN_LMDB=/abs/path/to/train
export OC20_VAL_LMDB=/abs/path/to/val_id
export OC20_DEVICE=auto
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train
```

If you need the legacy `.pt` path for older experiments:

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train --dataset-format pt --train-path /abs/path/to/train.pt --val-path /abs/path/to/val.pt
```

## Default config

Primary config:

- `configs/oc20/s2ef/gotennet.yaml`

The default `device: auto` setting selects CUDA when available and falls back to CPU otherwise. You
can also override the device explicitly with `--device cuda`, `--device cuda:0`, `--device cpu`, or
`OC20_DEVICE`.

## Train

### Train with standard OC20 LMDB data

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train --dataset-format lmdb --train-path /abs/path/to/train --val-path /abs/path/to/val_id --device auto
```

### Train with config/environment overrides only

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train
```

### Validate with standard OC20 LMDB data

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode validate --dataset-format lmdb --val-path /abs/path/to/val_id --device auto
```

## Error messages you should now expect

- Missing `.pt` files explain that `.pt` mode is a repo-specific convenience format.
- Missing LMDB paths explain that the runner expects an LMDB file, an LMDB directory, or a
  directory containing LMDB shards.
- If LMDB support is selected but the `lmdb` Python package is not installed, the runner tells you
  exactly which package to install.

## Notes / caveats

- This repo still uses a lightweight OC20-compatible local runner rather than the full FairChem CLI.
- Periodic graph construction uses per-structure cell/pbc and enforces a neighbor cap.
- Official OC20/FairChem LMDB paths now work directly without a custom conversion step.
