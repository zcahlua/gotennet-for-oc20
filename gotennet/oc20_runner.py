from __future__ import annotations

import argparse
import io
import os
import pickle
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from gotennet.models.oc20_model import OC20GotenNetS2EF

DEFAULT_CONFIG_PATH = "configs/oc20/s2ef/gotennet.yaml"
PT_REQUIRED_SAMPLE_KEYS = ("atomic_numbers", "pos", "cell", "pbc")
LMDB_ENV_CANDIDATE_NAMES = ("data.lmdb",)
DUMMY_LMDB_TRAIN_SAMPLES = 20
DUMMY_LMDB_VAL_SAMPLES = 8


class OC20DatasetConfigurationError(RuntimeError):
    """Raised when the OC20 dataset configuration is invalid or incomplete."""


class OC20DatasetFileNotFoundError(FileNotFoundError):
    """Raised when an OC20 dataset tensor file is missing."""


class OC20DatasetDependencyError(ImportError):
    """Raised when an optional OC20 dataset dependency is unavailable."""


class OC20DatasetFormatError(RuntimeError):
    """Raised when dataset contents cannot be parsed into OC20 samples."""


def create_dummy_oc20_lmdb(path: str, num_samples: int = 20) -> None:
    """Create a tiny OC20-style LMDB dataset for smoke runs and local debugging."""
    try:
        import lmdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise OC20DatasetDependencyError(
            "Cannot auto-generate dummy OC20 LMDB dataset because 'lmdb' is not installed. "
            "Install it with `pip install lmdb`."
        ) from exc

    dataset_path = Path(path).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    dataset_path = dataset_path.resolve()

    if dataset_path.suffix == ".lmdb":
        env_path = dataset_path
        env_path.parent.mkdir(parents=True, exist_ok=True)
        subdir = False
    else:
        env_path = dataset_path
        if env_path.exists() and env_path.is_file():
            env_path.unlink()
        env_path.mkdir(parents=True, exist_ok=True)
        subdir = True

    if env_path.exists() and env_path.is_dir() and (env_path / "data.mdb").exists():
        shutil.rmtree(env_path)
        env_path.mkdir(parents=True, exist_ok=True)
    if env_path.exists() and env_path.is_file():
        env_path.unlink()

    map_size = 1 << 24
    env = lmdb.open(str(env_path), map_size=map_size, subdir=subdir)
    with env.begin(write=True) as txn:
        txn.put(b"length", str(num_samples).encode("utf-8"))
        base_cell = torch.eye(3, dtype=torch.float) * 6.0
        base_z = torch.tensor([8, 1, 1], dtype=torch.long)
        base_force = torch.zeros(3, 3, dtype=torch.float)
        base_fixed = torch.tensor([False, False, False], dtype=torch.bool)
        base_pbc = torch.tensor([True, True, False], dtype=torch.bool)
        for idx in range(num_samples):
            offset = float(idx) * 0.01
            pos = torch.tensor(
                [[0.0 + offset, 0.0, 0.0], [1.4 + offset, 0.0, 0.0], [0.0 + offset, 1.4, 0.0]],
                dtype=torch.float,
            )
            item = {
                "z": base_z.clone(),
                "pos": pos,
                "cell": base_cell.clone(),
                "pbc": base_pbc.clone(),
                "y": torch.tensor([0.0], dtype=torch.float),
                "force": base_force.clone(),
                "fixed": base_fixed.clone(),
                "natoms": torch.tensor([3], dtype=torch.long),
            }
            txn.put(str(idx).encode("ascii"), pickle.dumps(item))
    env.sync()
    env.close()


class PTListDataset(Dataset[Data]):
    def __init__(self, samples: Sequence[Data]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Data:
        return self.samples[index]


class LMDBDataset(Dataset[Data]):
    def __init__(self, env_paths: Sequence[Path]):
        try:
            import lmdb  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in integration environments
            raise OC20DatasetDependencyError(
                "LMDB dataset support requires the optional 'lmdb' Python package. "
                "Install it with `pip install lmdb` or `pip install -e .[full]` after updating dependencies."
            ) from exc

        self._lmdb = lmdb
        self._env_paths = list(env_paths)
        self._envs = []
        self._keys: List[tuple[int, bytes]] = []

        for env_index, env_path in enumerate(self._env_paths):
            env = lmdb.open(
                str(env_path),
                subdir=env_path.is_dir(),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
            self._envs.append(env)
            for key in self._enumerate_env_keys(env, env_path):
                self._keys.append((env_index, key))

        if not self._keys:
            raise OC20DatasetConfigurationError(
                "Resolved LMDB dataset path(s), but no sample entries were found. "
                "Expected numbered keys such as '0', '1', ... inside the LMDB environment(s)."
            )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int) -> Data:
        env_index, key = self._keys[index]
        env = self._envs[env_index]
        with env.begin(write=False) as txn:
            value = txn.get(key)
        if value is None:
            raise OC20DatasetFormatError(
                f"LMDB key {key!r} disappeared while reading from {self._env_paths[env_index]}."
            )
        raw = _deserialize_lmdb_value(value, env_path=self._env_paths[env_index], key=key)
        return _sample_to_data(raw, source=f"{self._env_paths[env_index]}::{key.decode(errors='replace')}")

    @staticmethod
    def _enumerate_env_keys(env: Any, env_path: Path) -> List[bytes]:
        keys: List[bytes] = []
        with env.begin(write=False) as txn:
            length = txn.get(b"length")
            if length is not None:
                try:
                    num_entries = int(length.decode("utf-8"))
                except ValueError as exc:
                    raise OC20DatasetFormatError(
                        f"LMDB metadata key 'length' in {env_path} is not an integer: {length!r}."
                    ) from exc
                keys.extend(str(idx).encode("ascii") for idx in range(num_entries))
                return keys

            cursor = txn.cursor()
            for key, _ in cursor:
                if key in {b"length", b"metadata", b"__keys__"}:
                    continue
                if key.isdigit():
                    keys.append(bytes(key))

        if keys:
            keys.sort(key=lambda item: int(item.decode("ascii")))
        return keys


def _torch_load_data(path: str) -> Any:
    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def _torch_load_bytes(blob: bytes) -> Any:
    buffer = io.BytesIO(blob)
    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(buffer, weights_only=False, **load_kwargs)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, **load_kwargs)


def _config_path_hint(cfg: Dict[str, Any]) -> str:
    config_path = cfg.get("_meta", {}).get("config_path")
    return str(config_path) if config_path else DEFAULT_CONFIG_PATH


def _dataset_override_hint(config_path: str, dataset_format: str) -> str:
    format_help = (
        "Use train.pt / val.pt Python-list tensors."
        if dataset_format == "pt"
        else "Use an LMDB file or a directory containing one or more LMDB environments (*.lmdb or data.mdb/lock.mdb)."
    )
    train_env = "OC20_TRAIN_PT" if dataset_format == "pt" else "OC20_TRAIN_LMDB"
    val_env = "OC20_VAL_PT" if dataset_format == "pt" else "OC20_VAL_LMDB"
    return textwrap.dedent(
        f"""
        Expected dataset.format: {dataset_format}
        Format notes: {format_help}

        Override the dataset paths with either:
          - CLI flags: python main.py --config {config_path} --mode train --dataset-format {dataset_format} --train-path /abs/path/to/train --val-path /abs/path/to/val
          - Environment variables: export {train_env}=/abs/path/to/train && export {val_env}=/abs/path/to/val
          - Direct YAML edits in {config_path}
        """
    ).strip()


def _build_missing_path_message(
    *,
    missing_path: str,
    split_name: str,
    config_key: str,
    cfg: Dict[str, Any],
    dataset_format: str,
) -> str:
    config_path = _config_path_hint(cfg)
    if dataset_format == "pt":
        format_message = textwrap.dedent(
            """
            This repository does not ship OC20 `.pt` dataset tensors. The `.pt` mode is a
            repo-specific convenience format that expects a Python list of sample dicts.
            Each sample must provide at least: atomic_numbers, pos, cell, and pbc.
            """
        ).strip()
    else:
        format_message = textwrap.dedent(
            """
            Standard OC20 / FairChem S2EF training data is stored as LMDB. Point this path at
            either a single LMDB file, an LMDB directory, or a directory containing one or more
            LMDB shards. Typical layouts include paths like `train/data.lmdb`, `val_id/data.lmdb`,
            or a split directory that contains multiple `*.lmdb` shards.
            """
        ).strip()

    return textwrap.dedent(
        f"""
        Missing OC20 {split_name} dataset path: {missing_path}
        Config key: dataset.{config_key}
        Config file: {config_path}

        {format_message}

        {_dataset_override_hint(config_path, dataset_format)}
        """
    ).strip()


def _build_invalid_path_message(
    *,
    invalid_path: str,
    split_name: str,
    config_key: str,
    cfg: Dict[str, Any],
    dataset_format: str,
) -> str:
    config_path = _config_path_hint(cfg)
    expectation = (
        "Expected a `.pt` file containing a Python list of sample dictionaries."
        if dataset_format == "pt"
        else "Expected an LMDB file, an LMDB directory, or a directory that contains one or more LMDB environments."
    )
    return textwrap.dedent(
        f"""
        Invalid OC20 {split_name} dataset path: {invalid_path}
        Config key: dataset.{config_key}
        Config file: {config_path}

        {expectation}

        {_dataset_override_hint(config_path, dataset_format)}
        """
    ).strip()


def _resolve_device(device_name: str | None) -> torch.device:
    requested = (device_name or os.environ.get("OC20_DEVICE") or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _coerce_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _resolve_dataset_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_format = str(dataset_cfg.get("format", os.environ.get("OC20_DATASET_FORMAT") or "pt")).lower()
    if dataset_format not in {"pt", "lmdb"}:
        raise OC20DatasetConfigurationError(
            f"Unsupported dataset.format={dataset_format!r}. Expected 'pt' or 'lmdb'."
        )

    env_prefix = "PT" if dataset_format == "pt" else "LMDB"
    resolved_paths = {
        "format": dataset_format,
        "train_path": os.environ.get(f"OC20_TRAIN_{env_prefix}") or dataset_cfg.get("train_path"),
        "val_path": os.environ.get(f"OC20_VAL_{env_prefix}") or dataset_cfg.get("val_path"),
    }
    dataset_cfg.update(resolved_paths)
    return resolved_paths


def _validate_dataset_path(
    *,
    cfg: Dict[str, Any],
    config_key: str,
    split_name: str,
    path_value: str | None,
    dataset_format: str,
) -> Path:
    if not path_value:
        if dataset_format != "lmdb":
            raise OC20DatasetConfigurationError(
                _build_missing_path_message(
                    missing_path="<unset>",
                    split_name=split_name,
                    config_key=config_key,
                    cfg=cfg,
                    dataset_format=dataset_format,
                )
            )
        path_value = f"data/oc20/dummy/{config_key}"

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        if dataset_format != "lmdb":
            raise OC20DatasetFileNotFoundError(
                _build_missing_path_message(
                    missing_path=str(path),
                    split_name=split_name,
                    config_key=config_key,
                    cfg=cfg,
                    dataset_format=dataset_format,
                )
            )

        sample_count = DUMMY_LMDB_TRAIN_SAMPLES if split_name == "train" else DUMMY_LMDB_VAL_SAMPLES
        create_dummy_oc20_lmdb(str(path), num_samples=sample_count)
        print(f"[oc20_runner] Created dummy LMDB dataset for {split_name} split at: {path}")
    return path


def _discover_lmdb_env_paths(path: Path, *, cfg: Dict[str, Any], split_name: str, config_key: str) -> List[Path]:
    if path.is_file():
        if path.suffix != ".lmdb":
            raise OC20DatasetConfigurationError(
                _build_invalid_path_message(
                    invalid_path=str(path),
                    split_name=split_name,
                    config_key=config_key,
                    cfg=cfg,
                    dataset_format="lmdb",
                )
            )
        return [path]

    if path.is_dir() and (path / "data.mdb").exists():
        return [path]

    matches: List[Path] = []
    if path.is_dir():
        for candidate in sorted(path.rglob("*.lmdb")):
            matches.append(candidate)
        for candidate_name in LMDB_ENV_CANDIDATE_NAMES:
            for candidate in sorted(path.rglob(candidate_name)):
                if candidate not in matches:
                    matches.append(candidate)
        for candidate in sorted(path.rglob("data.mdb")):
            env_dir = candidate.parent
            if env_dir not in matches:
                matches.append(env_dir)

    if matches:
        return matches

    raise OC20DatasetConfigurationError(
        _build_invalid_path_message(
            invalid_path=str(path),
            split_name=split_name,
            config_key=config_key,
            cfg=cfg,
            dataset_format="lmdb",
        )
    )


def _deserialize_lmdb_value(blob: bytes, *, env_path: Path, key: bytes) -> Any:
    errors: List[str] = []
    for loader_name, loader in (("pickle", pickle.loads), ("torch", _torch_load_bytes)):
        try:
            return loader(blob)
        except Exception as exc:  # pragma: no cover - defensive branch
            errors.append(f"{loader_name}: {exc}")
    raise OC20DatasetFormatError(
        f"Could not decode LMDB sample {key!r} from {env_path}. Tried pickle and torch deserialization. "
        f"Errors: {'; '.join(errors)}"
    )


def _sample_to_data(sample: Any, *, source: str) -> Data:
    if isinstance(sample, Data):
        item = sample.to_dict()
    elif isinstance(sample, dict):
        item = sample
    else:
        raise OC20DatasetConfigurationError(
            f"Sample from {source} is a {type(sample).__name__}, expected a dict or torch_geometric.data.Data."
        )

    if "pos" not in item:
        raise OC20DatasetConfigurationError(f"Sample from {source} is missing required key: pos.")

    pos = _coerce_tensor(item["pos"], dtype=torch.float)

    atomic_numbers = item.get("atomic_numbers")
    if atomic_numbers is None and "atomic_numbers" not in item:
        atomic_numbers = item.get("z")
    if atomic_numbers is None:
        raise OC20DatasetConfigurationError(
            f"Sample from {source} is missing required atomic numbers ('atomic_numbers' or 'z')."
        )

    cell = item.get("cell")
    if cell is None:
        raise OC20DatasetConfigurationError(f"Sample from {source} is missing required key: cell.")
    cell_tensor = _coerce_tensor(cell, dtype=torch.float)
    if cell_tensor.ndim == 3 and cell_tensor.size(0) == 1:
        cell_tensor = cell_tensor[0]
    # Ensure cell is [3, 3] - per-graph property
    cell_tensor = cell_tensor.view(3, 3)

    pbc = item.get("pbc")
    if pbc is None:
        raise OC20DatasetConfigurationError(f"Sample from {source} is missing required key: pbc.")
    pbc_tensor = _coerce_tensor(pbc, dtype=torch.bool)
    # Ensure pbc is always [3] shape (per-graph property, not per-node)
    if pbc_tensor.ndim == 2 and pbc_tensor.size(0) == 1:
        pbc_tensor = pbc_tensor[0]
    elif pbc_tensor.ndim == 2 and pbc_tensor.size(1) == 3:
        # If somehow we got [N, 3], take the first row as all should be same per graph
        pbc_tensor = pbc_tensor[0]

    energy = item.get("energy")
    if energy is None:
        energy = item.get("y", torch.tensor(0.0))
    energy_tensor = _coerce_tensor(energy, dtype=torch.float).view(1)

    forces = item.get("forces")
    if forces is None:
        forces = item.get("force", torch.zeros_like(pos))
    forces_tensor = _coerce_tensor(forces, dtype=torch.float)

    fixed = item.get("fixed")
    if fixed is None:
        tags = item.get("tags")
        if tags is not None:
            fixed = _coerce_tensor(tags, dtype=torch.long) == 0
        else:
            fixed = torch.zeros(pos.size(0), dtype=torch.bool)
    fixed_tensor = _coerce_tensor(fixed, dtype=torch.bool).view(-1)

    natoms = item.get("natoms")
    if natoms is None:
        natoms = torch.tensor([pos.size(0)], dtype=torch.long)
    natoms_tensor = _coerce_tensor(natoms, dtype=torch.long).view(1)

    data = Data(
        atomic_numbers=_coerce_tensor(atomic_numbers, dtype=torch.long).view(-1),
        pos=pos,
        energy=energy_tensor,
        forces=forces_tensor,
        cell=cell_tensor.view(1, 3, 3),  # [1, 3, 3] to prevent PyG from flattening during batching
        pbc=pbc_tensor.view(1, 3),  # [1, 3] to prevent PyG from flattening during batching
        fixed=fixed_tensor,
        natoms=natoms_tensor,
    )
    return data


def _read_pt_dataset(path: Path, *, split_name: str, config_key: str, cfg: Dict[str, Any]) -> PTListDataset:
    raw = _torch_load_data(str(path))
    if not isinstance(raw, list):
        raise OC20DatasetConfigurationError(
            f"Expected {path} to contain a Python list of sample dictionaries, got {type(raw).__name__}."
        )
    if not raw:
        raise OC20DatasetConfigurationError(f"Dataset file {path} is empty; expected at least one sample.")

    samples: List[Data] = []
    for sample_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise OC20DatasetConfigurationError(
                f"Sample {sample_index} in {path} is a {type(item).__name__}, expected a dict."
            )
        missing_keys = [key for key in PT_REQUIRED_SAMPLE_KEYS if key not in item]
        if missing_keys:
            raise OC20DatasetConfigurationError(
                f"Sample {sample_index} in {path} is missing required keys: {', '.join(missing_keys)}."
            )
        samples.append(_sample_to_data(item, source=f"{path}::{sample_index}"))
    return PTListDataset(samples)


def _read_lmdb_dataset(path: Path, *, split_name: str, config_key: str, cfg: Dict[str, Any]) -> LMDBDataset:
    env_paths = _discover_lmdb_env_paths(path, cfg=cfg, split_name=split_name, config_key=config_key)
    return LMDBDataset(env_paths)


def _read_dataset(
    path: str,
    *,
    split_name: str,
    config_key: str,
    cfg: Dict[str, Any],
    dataset_format: str,
) -> Dataset[Data]:
    dataset_path = _validate_dataset_path(
        cfg=cfg,
        config_key=config_key,
        split_name=split_name,
        path_value=path,
        dataset_format=dataset_format,
    )

    if dataset_format == "pt":
        return _read_pt_dataset(dataset_path, split_name=split_name, config_key=config_key, cfg=cfg)
    return _read_lmdb_dataset(dataset_path, split_name=split_name, config_key=config_key, cfg=cfg)


def _iter_batches(dataset: Dataset[Data], batch_size: int, *, shuffle: bool) -> Iterable[Batch]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Batch.from_data_list,
    )
    yield from dataloader


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

    train_dataset = (
        _read_dataset(
            dataset_paths["train_path"],
            split_name="train",
            config_key="train_path",
            cfg=cfg,
            dataset_format=dataset_paths["format"],
        )
        if mode == "train"
        else None
    )
    val_dataset = _read_dataset(
        dataset_paths["val_path"],
        split_name="validation",
        config_key="val_path",
        cfg=cfg,
        dataset_format=dataset_paths["format"],
    )

    if mode == "train" and train_dataset is not None:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
        )

        epochs = cfg["trainer"]["max_epochs"]
        for epoch in range(epochs):
            model.train()
            loss = None
            for batch in _iter_batches(train_dataset, cfg["dataset"]["batch_size"], shuffle=True):
                batch = batch.to(device)
                out = model(batch)
                e_loss = (out["energy"] - batch.energy.view(-1)).abs().mean()
                f_loss = _masked_force_mae(out["forces"], batch.forces, batch.fixed)
                loss = cfg["loss"]["energy_weight"] * e_loss + cfg["loss"]["force_weight"] * f_loss
                optim.zero_grad()
                loss.backward()
                optim.step()
            if loss is None:
                raise OC20DatasetConfigurationError("Training dataset is empty; expected at least one batch.")
            print(f"epoch={epoch+1}/{epochs} train_loss={float(loss):.6f}")

    model.eval()
    with torch.set_grad_enabled(True):
        mae_energy = []
        mae_force = []
        for batch in _iter_batches(val_dataset, cfg["dataset"]["batch_size"], shuffle=False):
            batch = batch.to(device)
            out = model(batch)
            mae_energy.append((out["energy"] - batch.energy.view(-1)).abs().mean().detach())
            mae_force.append(_masked_force_mae(out["forces"], batch.forces, batch.fixed).detach())

    if not mae_energy or not mae_force:
        raise OC20DatasetConfigurationError("Validation dataset is empty; expected at least one batch.")

    print(
        f"validation_energy_mae={torch.stack(mae_energy).mean().item():.6f} "
        f"validation_force_mae={torch.stack(mae_force).mean().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OC20 S2EF runner for GotenNet")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["train", "validate"], default="train")
    parser.add_argument("--dataset-format", choices=["pt", "lmdb"], help="Override dataset.format.")
    parser.add_argument("--train-path", help="Override dataset.train_path for the OC20 train split.")
    parser.add_argument("--val-path", help="Override dataset.val_path for the OC20 validation split.")
    parser.add_argument("--device", help="Override config device. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("_meta", {})["config_path"] = str(cfg_path)
    if args.dataset_format:
        cfg.setdefault("dataset", {})["format"] = args.dataset_format
    if args.train_path:
        cfg.setdefault("dataset", {})["train_path"] = args.train_path
    if args.val_path:
        cfg.setdefault("dataset", {})["val_path"] = args.val_path
    if args.device:
        cfg["device"] = args.device

    run(mode=args.mode, cfg=cfg)


if __name__ == "__main__":
    main()
