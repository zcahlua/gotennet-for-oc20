from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch_scatter import scatter

from gotennet.models.components.layers import CosineCutoff
from gotennet.models.pbc import build_pbc_graph
from gotennet.models.representation.gotennet import GotenNet


@dataclass
class GraphConfig:
    cutoff: float = 6.0
    max_neighbors: int = 50


class OC20GotenNetS2EF(nn.Module):
    """OC20 S2EF model wrapper around core GotenNet representation."""

    def __init__(
        self,
        representation: Dict,
        graph: Dict,
        direct_forces: bool = False,
    ) -> None:
        super().__init__()
        self.graph_cfg = GraphConfig(**graph)
        rep_cfg = dict(representation)
        rep_cfg.setdefault("cutoff_fn", CosineCutoff(self.graph_cfg.cutoff))
        self.representation = GotenNet(**rep_cfg)
        hidden = representation["n_atom_basis"]
        self.energy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.direct_forces = direct_forces
        if direct_forces:
            self.force_head = nn.Linear(hidden, 3)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        pos = batch.pos
        if not self.direct_forces:
            pos = pos.requires_grad_(True)

        edge_index, edge_dist, edge_vec = build_pbc_graph(
            pos=pos,
            batch=batch.batch,
            cell=batch.cell,
            pbc=batch.pbc,
            cutoff=self.graph_cfg.cutoff,
            max_neighbors=self.graph_cfg.max_neighbors,
            include_self=True,
        )

        h, _ = self.representation(
            atomic_numbers=batch.atomic_numbers,
            edge_index=edge_index,
            edge_diff=edge_dist,
            edge_vec=edge_vec,
        )

        per_atom_energy = self.energy_head(h).squeeze(-1)
        energy = scatter(per_atom_energy, batch.batch, dim=0, reduce="sum")

        if self.direct_forces:
            forces = self.force_head(h)
        else:
            grad_outputs = torch.ones_like(energy)
            forces = -torch.autograd.grad(
                energy,
                pos,
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        return {"energy": energy, "forces": forces, "edge_index": edge_index}
