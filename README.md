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

Install core requirements from the project (PyTorch, PyG stack, e3nn, torch_scatter, ase, yaml).

```bash
pip install -e .[full]
```

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

## Default config

Primary config:

- `configs/oc20/s2ef/gotennet.yaml`

Update `dataset.train_path` and `dataset.val_path` to your local OC20-derived tensors.

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
