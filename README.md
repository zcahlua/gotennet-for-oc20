# GotenNet for Open Catalyst OC20 S2EF

This repository is now **OC20-first**: the primary/default path trains **GotenNet** on the
Open Catalyst 2020 Structure-to-Energy-and-Forces (S2EF) task with periodic boundary
conditions.

## What changed

- Replaced the QM9-first training workflow with a repo-local OC20 launcher.
- Added an OC20 S2EF model wrapper that uses GotenNet representation + periodic graph building.
- Added energy + force prediction (default: force from energy gradients).
- Added an OC20 default config and smoke tests.

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

## Expected OC20 sample format

The launcher expects `train.pt` / `val.pt` files each containing a Python list of dicts, where
entries provide at least:

- `atomic_numbers` (`[N]`, `long`)
- `pos` (`[N, 3]`, `float`)
- `cell` (`[3, 3]`, `float`)
- `pbc` (`[3]`, `bool`)
- `energy` (`[]` or `[1]`, optional for train/val)
- `forces` (`[N, 3]`, optional)
- `fixed` (`[N]`, optional bool mask)


## Prepare / point to OC20 data

This fork currently expects preconverted `train.pt` and `val.pt` files. The repository does **not** include an OC20 LMDB conversion script, so you need to generate these tensors with your existing OC20 preprocessing pipeline and then point the runner at them.

You can choose any of these path override mechanisms:

1. Edit `configs/oc20/s2ef/gotennet.yaml` directly.
2. Pass explicit CLI overrides:

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train --train-path /abs/path/to/train.pt --val-path /abs/path/to/val.pt
```

3. Export environment variables before launching:

```bash
export OC20_TRAIN_PT=/abs/path/to/train.pt
export OC20_VAL_PT=/abs/path/to/val.pt
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train
```

If a dataset file is missing, the runner now raises a targeted error that tells you which file is missing, which config key points to it, and how to override it.

## Default config

Primary config:

- `configs/oc20/s2ef/gotennet.yaml`

Update `dataset.train_path` and `dataset.val_path` to your local OC20-derived tensors, or override them at runtime with CLI flags or environment variables.

## Train (default command)

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode train
```

## Validate (default command)

```bash
python main.py --config configs/oc20/s2ef/gotennet.yaml --mode validate
```

## Notes / caveats

- This repo now defaults to a lightweight OC20-compatible local runner (not a FairChem CLI plugin).
- Periodic graph construction uses per-structure cell/pbc and enforces a neighbor cap.
- If you want official OC20 LMDB ingestion, add a converter step to `.pt` sample lists.
