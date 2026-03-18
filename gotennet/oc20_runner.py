from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch_geometric.data import Batch, Data

from gotennet.models.oc20_model import OC20GotenNetS2EF

DEFAULT_CONFIG_PATH = "configs/oc20/s2ef/gotennet.yaml"
REQUIRED_SAMPLE_KEYS = ("atomic_numbers", "pos", "cell", "pbc")


class OC20DatasetConfigurationError(RuntimeError):
    """Raised when the OC20 dataset configuration is invalid or incomplete."""


class OC20DatasetFileNotFoundError(FileNotFoundError):
    """Raised when an OC20 dataset tensor file is missing."""


def _torch_load_data(path: str) -> Any:
    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def _config_path_hint(cfg: Dict[str, Any]) -> str:
    config_path = cfg.get("_meta", {}).get("config_path")
    return str(config_path) if config_path else DEFAULT_CONFIG_PATH


def _dataset_override_hint(config_path: str) -> str:
    return textwrap.dedent(
        f"""
        Override the dataset paths with either:
          - CLI flags: python main.py --config {config_path} --mode train --train-path /abs/path/to/train.pt --val-path /abs/path/to/val.pt
          - Environment variables: export OC20_TRAIN_PT=/abs/path/to/train.pt && export OC20_VAL_PT=/abs/path/to/val.pt
          - Direct YAML edits in {config_path}
        """
    ).strip()


def _build_missing_path_message(
    *,
    missing_path: str,
    split_name: str,
    config_key: str,
    cfg: Dict[str, Any],
) -> str:
    config_path = _config_path_hint(cfg)
    return textwrap.dedent(
        f"""
        Missing OC20 {split_name} dataset file: {missing_path}
        Config key: dataset.{config_key}
        Config file: {config_path}

        This repository does not ship OC20 `.pt` dataset tensors. The OC20 runner expects
        preconverted `train.pt` / `val.pt` files containing a Python list of sample dicts.
        Each sample must provide at least: atomic_numbers, pos, cell, and pbc.

        {_dataset_override_hint(config_path)}

        This fork currently does not include an OC20 LMDB-to-`.pt` conversion script, so if
        you only have the original OC20 data you need to generate these `.pt` files with your
        existing preprocessing pipeline before launching training.
        """
    ).strip()


def _resolve_device(device_name: str | None) -> torch.device:
    requested = (device_name or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _coerce_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _resolve_dataset_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    dataset_cfg = cfg.setdefault("dataset", {})
    resolved_paths = {
        "train_path": os.environ.get("OC20_TRAIN_PT") or dataset_cfg.get("train_path"),
        "val_path": os.environ.get("OC20_VAL_PT") or dataset_cfg.get("val_path"),
    }
    dataset_cfg.update(resolved_paths)
    return resolved_paths


def _validate_dataset_path(
    *,
    cfg: Dict[str, Any],
    config_key: str,
    split_name: str,
    path_value: str | None,
) -> str:
    if not path_value:
        raise OC20DatasetConfigurationError(
            _build_missing_path_message(
                missing_path="<unset>",
                split_name=split_name,
                config_key=config_key,
                cfg=cfg,
            )
        )

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise OC20DatasetFileNotFoundError(
            _build_missing_path_message(
                missing_path=str(path),
                split_name=split_name,
                config_key=config_key,
                cfg=cfg,
            )
        )
    return str(path)


def _read_samples(path: str, *, split_name: str, config_key: str, cfg: Dict[str, Any]) -> List[Data]:
    dataset_path = _validate_dataset_path(
        cfg=cfg,
        config_key=config_key,
        split_name=split_name,
        path_value=path,
    )
    raw = _torch_load_data(dataset_path)
    if not isinstance(raw, list):
        raise OC20DatasetConfigurationError(
            f"Expected {dataset_path} to contain a Python list of sample dictionaries, got {type(raw).__name__}."
        )
    if not raw:
        raise OC20DatasetConfigurationError(f"Dataset file {dataset_path} is empty; expected at least one sample.")

    samples: List[Data] = []
    for sample_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise OC20DatasetConfigurationError(
                f"Sample {sample_index} in {dataset_path} is a {type(item).__name__}, expected a dict."
            )
        missing_keys = [key for key in REQUIRED_SAMPLE_KEYS if key not in item]
        if missing_keys:
            raise OC20DatasetConfigurationError(
                f"Sample {sample_index} in {dataset_path} is missing required keys: {', '.join(missing_keys)}."
            )

        pos = _coerce_tensor(item["pos"], dtype=torch.float)
        samples.append(
            Data(
                atomic_numbers=_coerce_tensor(item["atomic_numbers"], dtype=torch.long),
                pos=pos,
                energy=_coerce_tensor(item.get("energy", torch.tensor(0.0)), dtype=torch.float).view(1),
                forces=_coerce_tensor(item.get("forces", torch.zeros_like(pos)), dtype=torch.float),
                cell=_coerce_tensor(item["cell"], dtype=torch.float),
                pbc=_coerce_tensor(item["pbc"], dtype=torch.bool),
                fixed=_coerce_tensor(
                    item.get("fixed", torch.zeros(pos.size(0), dtype=torch.bool)),
                    dtype=torch.bool,
                ),
                natoms=torch.tensor([pos.size(0)], dtype=torch.long),
            )
        )
    return samples


def _iter_batches(samples: List[Data], batch_size: int):
    for i in range(0, len(samples), batch_size):
        yield Batch.from_data_list(samples[i : i + batch_size])


def _masked_force_mae(pred: torch.Tensor, target: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    free_mask = ~fixed
    if free_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (pred[free_mask] - target[free_mask]).abs().mean()


def run(mode: str, cfg: Dict) -> None:
    dataset_paths = _resolve_dataset_paths(cfg)
    device = _resolve_device(cfg.get("device", "auto"))
    model = OC20GotenNetS2EF(
        representation=cfg["model"]["representation"],
        graph=cfg["model"]["graph"],
        direct_forces=cfg["model"].get("direct_forces", False),
    ).to(device)

    train_samples = (
        _read_samples(
            dataset_paths["train_path"],
            split_name="train",
            config_key="train_path",
            cfg=cfg,
        )
        if mode == "train"
        else []
    )
    val_samples = _read_samples(
        dataset_paths["val_path"],
        split_name="validation",
        config_key="val_path",
        cfg=cfg,
    )

    if mode == "train":
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
        )

        epochs = cfg["trainer"]["max_epochs"]
        for epoch in range(epochs):
            model.train()
            for batch in _iter_batches(train_samples, cfg["dataset"]["batch_size"]):
                batch = batch.to(device)
                out = model(batch)
                e_loss = (out["energy"] - batch.energy.view(-1)).abs().mean()
                f_loss = _masked_force_mae(out["forces"], batch.forces, batch.fixed)
                loss = cfg["loss"]["energy_weight"] * e_loss + cfg["loss"]["force_weight"] * f_loss
                optim.zero_grad()
                loss.backward()
                optim.step()
            print(f"epoch={epoch+1}/{epochs} train_loss={float(loss):.6f}")

    model.eval()
    with torch.set_grad_enabled(True):
        mae_energy = []
        mae_force = []
        for batch in _iter_batches(val_samples, cfg["dataset"]["batch_size"]):
            batch = batch.to(device)
            out = model(batch)
            mae_energy.append((out["energy"] - batch.energy.view(-1)).abs().mean().detach())
            mae_force.append(_masked_force_mae(out["forces"], batch.forces, batch.fixed).detach())

    print(
        f"validation_energy_mae={torch.stack(mae_energy).mean().item():.6f} "
        f"validation_force_mae={torch.stack(mae_force).mean().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OC20 S2EF runner for GotenNet")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["train", "validate"], default="train")
    parser.add_argument("--train-path", help="Override dataset.train_path for the OC20 train split.")
    parser.add_argument("--val-path", help="Override dataset.val_path for the OC20 validation split.")
    parser.add_argument("--device", help="Override config device. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("_meta", {})["config_path"] = str(cfg_path)
    if args.train_path:
        cfg.setdefault("dataset", {})["train_path"] = args.train_path
    if args.val_path:
        cfg.setdefault("dataset", {})["val_path"] = args.val_path
    if args.device:
        cfg["device"] = args.device

    run(mode=args.mode, cfg=cfg)


if __name__ == "__main__":
    main()
