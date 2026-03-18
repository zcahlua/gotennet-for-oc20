import pickle
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from torch_geometric.data import Batch, Data

from gotennet.models.oc20_model import OC20GotenNetS2EF
from gotennet.models.pbc import build_pbc_graph
from gotennet.oc20_runner import OC20DatasetFileNotFoundError, run

try:
    import lmdb
except ImportError:  # pragma: no cover - depends on local environment
    lmdb = None


def _sample_data(shift=None):
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]])
    if shift is not None:
        pos = pos + shift
    return Data(
        atomic_numbers=torch.tensor([8, 1, 1]),
        pos=pos,
        batch=torch.zeros(3, dtype=torch.long),
        cell=torch.eye(3).unsqueeze(0) * 6.0,
        pbc=torch.tensor([[True, True, False]]),
        natoms=torch.tensor([3]),
        fixed=torch.tensor([False, False, False]),
        energy=torch.tensor([0.0]),
        forces=torch.zeros(3, 3),
    )


def _cfg(tmpdir, dataset_format="pt"):
    train_path = f"{tmpdir}/train.pt" if dataset_format == "pt" else f"{tmpdir}/train"
    val_path = f"{tmpdir}/val.pt" if dataset_format == "pt" else f"{tmpdir}/val"
    base = {
        "device": "cpu",
        "dataset": {
            "format": dataset_format,
            "train_path": train_path,
            "val_path": val_path,
            "batch_size": 1,
        },
        "model": {
            "direct_forces": False,
            "graph": {"cutoff": 3.0, "max_neighbors": 20},
            "representation": {
                "n_atom_basis": 32,
                "n_interactions": 2,
                "n_rbf": 16,
                "radial_basis": "expnorm",
                "activation": "swish",
                "max_z": 100,
                "num_heads": 4,
                "attn_dropout": 0.0,
                "edge_updates": False,
                "lmax": 2,
                "aggr": "add",
                "scale_edge": False,
            },
        },
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "loss": {"energy_weight": 0.1, "force_weight": 1.0},
        "trainer": {"max_epochs": 1},
    }
    return base


def _write_lmdb_dataset(path: str, items: list[dict]):
    if lmdb is None:
        pytest.skip("lmdb package is not installed in this environment")
    env = lmdb.open(path, map_size=1 << 20, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"length", str(len(items)).encode("utf-8"))
        for idx, item in enumerate(items):
            lmdb_item = {
                "z": item["atomic_numbers"],
                "pos": item["pos"],
                "cell": item["cell"],
                "pbc": item["pbc"],
                "y": item.get("energy", torch.tensor([0.0])).view(1),
                "force": item.get("forces", torch.zeros_like(item["pos"])),
                "fixed": item.get("fixed", torch.zeros(item["pos"].size(0), dtype=torch.bool)),
                "natoms": item.get("natoms", torch.tensor([item["pos"].size(0)])),
            }
            txn.put(str(idx).encode("ascii"), pickle.dumps(lmdb_item))
    env.close()


def test_config_and_model_construction():
    cfg = _cfg("/tmp")
    model = OC20GotenNetS2EF(
        representation=cfg["model"]["representation"],
        graph=cfg["model"]["graph"],
        direct_forces=False,
    )
    assert isinstance(model, OC20GotenNetS2EF)


def test_synthetic_periodic_batch_forward_and_output_shapes():
    cfg = _cfg("/tmp")
    model = OC20GotenNetS2EF(
        representation=cfg["model"]["representation"],
        graph=cfg["model"]["graph"],
        direct_forces=False,
    )
    batch = Batch.from_data_list([_sample_data()])
    out = model(batch)
    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (3, 3)
    assert out["edge_index"].shape[0] == 2


def test_geometry_translation_invariance_for_energy():
    cfg = _cfg("/tmp")
    model = OC20GotenNetS2EF(
        representation=cfg["model"]["representation"],
        graph=cfg["model"]["graph"],
        direct_forces=False,
    )
    b1 = Batch.from_data_list([_sample_data()])
    b2 = Batch.from_data_list([_sample_data(shift=torch.tensor([3.0, -2.0, 0.5]))])
    e1 = model(b1)["energy"]
    e2 = model(b2)["energy"]
    assert torch.allclose(e1, e2, atol=1e-4)


def test_pbc_graph_builder_smoke():
    d = _sample_data()
    edge_index, edge_dist, edge_vec = build_pbc_graph(
        pos=d.pos,
        batch=d.batch,
        cell=d.cell,
        pbc=d.pbc,
        cutoff=3.0,
        max_neighbors=20,
    )
    assert edge_index.shape[1] > 0
    assert edge_dist.shape[0] == edge_vec.shape[0]


def test_launcher_config_load_smoke_with_pt_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        train = [_sample_data().to_dict()]
        val = [_sample_data().to_dict()]
        torch.save(train, f"{tmpdir}/train.pt")
        torch.save(val, f"{tmpdir}/val.pt")
        cfg = _cfg(tmpdir, dataset_format="pt")
        with open(f"{tmpdir}/cfg.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        run("train", cfg)
        run("validate", cfg)


def test_launcher_config_load_smoke_with_lmdb_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        train = [_sample_data().to_dict()]
        val = [_sample_data().to_dict()]
        _write_lmdb_dataset(f"{tmpdir}/train", train)
        _write_lmdb_dataset(f"{tmpdir}/val", val)
        cfg = _cfg(tmpdir, dataset_format="lmdb")

        run("train", cfg)
        run("validate", cfg)


def test_missing_pt_dataset_error_is_actionable():
    cfg = _cfg("/tmp/does-not-exist", dataset_format="pt")
    cfg["_meta"] = {"config_path": "configs/oc20/s2ef/gotennet.yaml"}

    with pytest.raises(OC20DatasetFileNotFoundError) as excinfo:
        run("train", cfg)

    message = str(excinfo.value)
    assert "dataset.train_path" in message
    assert "--train-path" in message
    assert "OC20_TRAIN_PT" in message
    assert "repo-specific convenience format" in message


def test_missing_lmdb_dataset_error_is_actionable():
    cfg = _cfg("/tmp/does-not-exist", dataset_format="lmdb")
    cfg["_meta"] = {"config_path": "configs/oc20/s2ef/gotennet.yaml"}

    with pytest.raises(OC20DatasetFileNotFoundError) as excinfo:
        run("train", cfg)

    message = str(excinfo.value)
    assert "dataset.train_path" in message
    assert "OC20_TRAIN_LMDB" in message
    assert "LMDB" in message
    assert "directory containing one or more" in message


def test_lmdb_shard_directory_is_discovered():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        shard_dir = root / "train_shards"
        shard_dir.mkdir()
        _write_lmdb_dataset(str(shard_dir / "0000.lmdb"), [_sample_data().to_dict()])
        _write_lmdb_dataset(str(root / "val"), [_sample_data().to_dict()])

        cfg = _cfg(tmpdir, dataset_format="lmdb")
        cfg["dataset"]["train_path"] = str(shard_dir)
        cfg["dataset"]["val_path"] = str(root / "val")

        run("train", cfg)
