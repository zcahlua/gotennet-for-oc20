from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch_geometric.data import Batch, Data

from gotennet.models.oc20_model import OC20GotenNetS2EF


def _read_samples(path: str) -> List[Data]:
    raw = torch.load(path)
    samples = []
    for item in raw:
        samples.append(
            Data(
                atomic_numbers=item["atomic_numbers"].long(),
                pos=item["pos"].float(),
                energy=item.get("energy", torch.tensor(0.0)).float().view(1),
                forces=item.get("forces", torch.zeros_like(item["pos"]).float()),
                cell=item["cell"].float(),
                pbc=item["pbc"].bool(),
                fixed=item.get("fixed", torch.zeros(item["pos"].size(0), dtype=torch.bool)),
                natoms=torch.tensor([item["pos"].size(0)], dtype=torch.long),
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
    device = torch.device(cfg.get("device", "cpu"))
    model = OC20GotenNetS2EF(
        representation=cfg["model"]["representation"],
        graph=cfg["model"]["graph"],
        direct_forces=cfg["model"].get("direct_forces", False),
    ).to(device)

    train_samples = _read_samples(cfg["dataset"]["train_path"]) if mode == "train" else []
    val_samples = _read_samples(cfg["dataset"]["val_path"])

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
    parser.add_argument("--config", default="configs/oc20/s2ef/gotennet.yaml")
    parser.add_argument("--mode", choices=["train", "validate"], default="train")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run(mode=args.mode, cfg=cfg)


if __name__ == "__main__":
    main()
