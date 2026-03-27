from __future__ import annotations

import itertools
from typing import Tuple

import torch


def _as_batch_cell(cell: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Convert cell to [num_graphs, 3, 3] shape.
    
    cell can be:
    - [3, 3]: single graph, expand to [num_graphs, 3, 3]
    - [1, 3, 3]: single graph with batch dim, expand to [num_graphs, 3, 3]
    - [num_graphs, 3, 3]: already correct shape
    - [num_graphs * 1, 3, 3]: batched [1, 3, 3] from PyG batching
    """
    if cell.dim() == 2:
        # [3, 3] -> [num_graphs, 3, 3]
        return cell.unsqueeze(0).expand(num_graphs, -1, -1)
    if cell.dim() == 3:
        if cell.size(0) == num_graphs:
            # [num_graphs, 3, 3] - already correct
            return cell
        elif cell.size(0) == 1:
            # [1, 3, 3] - single graph, expand
            return cell.expand(num_graphs, -1, -1)
        elif cell.size(0) == num_graphs * 1:
            # [num_graphs, 3, 3] from batched [1, 3, 3] - reshape
            return cell.view(num_graphs, 3, 3)
        else:
            raise ValueError(f"Cannot reshape cell from {tuple(cell.shape)} to [{num_graphs}, 3, 3]")
    raise ValueError(f"Expected cell shape [3, 3], [1, 3, 3], or [B, 3, 3], got {tuple(cell.shape)}")


def _as_batch_pbc(pbc: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Convert pbc to [num_graphs, 3] shape.
    
    pbc can be:
    - [3]: single graph, expand to [num_graphs, 3]
    - [1, 3]: single graph with batch dim, expand to [num_graphs, 3]
    - [num_graphs, 3]: already correct shape
    - [num_graphs * 1, 3]: batched [1, 3] from PyG batching
    """
    if pbc.dim() == 1:
        # [3] -> [num_graphs, 3]
        return pbc.unsqueeze(0).expand(num_graphs, -1)
    if pbc.dim() == 2:
        if pbc.size(0) == num_graphs:
            # [num_graphs, 3] - already correct
            return pbc
        elif pbc.size(0) == 1:
            # [1, 3] - single graph, expand
            return pbc.expand(num_graphs, -1)
        elif pbc.size(0) == num_graphs * 1:
            # [num_graphs, 3] from batched [1, 3] - reshape
            return pbc.view(num_graphs, 3)
        else:
            raise ValueError(f"Cannot reshape pbc from {tuple(pbc.shape)} to [{num_graphs}, 3]")
    raise ValueError(f"Expected pbc shape [3], [1, 3], or [B, 3], got {tuple(pbc.shape)}")


def build_pbc_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
    include_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct periodic radius graph and edge geometry for OC20-style batches."""
    device = pos.device
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    batch_cell = _as_batch_cell(cell, num_graphs).to(device)
    batch_pbc = _as_batch_pbc(pbc.bool(), num_graphs).to(device)

    edge_i, edge_j, edge_vec, edge_dist = [], [], [], []

    for g in range(num_graphs):
        atom_idx = torch.where(batch == g)[0]
        if atom_idx.numel() == 0:
            continue
        pos_g = pos[atom_idx]
        n = pos_g.size(0)
        cell_g = batch_cell[g]
        pbc_g = batch_pbc[g]

        lengths = torch.linalg.norm(cell_g, dim=1).clamp(min=1e-6)
        repeats = torch.where(pbc_g, torch.ceil(cutoff / lengths).long(), torch.zeros(3, dtype=torch.long, device=device))
        ranges = [range(-int(r.item()), int(r.item()) + 1) for r in repeats]

        per_atom_neighbors = torch.zeros(n, dtype=torch.long, device=device)
        for sx, sy, sz in itertools.product(*ranges):
            shift_int = torch.tensor([sx, sy, sz], device=device, dtype=pos.dtype)
            shift_cart = shift_int @ cell_g
            shifted = pos_g + shift_cart
            diff = pos_g[:, None, :] - shifted[None, :, :]
            dist = torch.linalg.norm(diff, dim=-1)
            valid = dist <= cutoff
            if not include_self and sx == sy == sz == 0:
                valid.fill_diagonal_(False)
            if include_self:
                if sx == sy == sz == 0:
                    pass
                else:
                    # prevent periodic self-edges across images
                    diag = torch.arange(n, device=device)
                    valid[diag, diag] = False

            row, col = torch.where(valid)
            if row.numel() == 0:
                continue

            if max_neighbors > 0:
                # enforce cap per source atom
                keep = []
                for a in range(n):
                    local = torch.where(row == a)[0]
                    if local.numel() == 0:
                        continue
                    remaining = max(0, max_neighbors - int(per_atom_neighbors[a].item()))
                    if remaining == 0:
                        continue
                    if local.numel() > remaining:
                        local_dist = dist[a, col[local]]
                        _, order = torch.topk(local_dist, k=remaining, largest=False)
                        local = local[order]
                    per_atom_neighbors[a] += local.numel()
                    keep.append(local)
                if not keep:
                    continue
                keep_idx = torch.cat(keep)
                row, col = row[keep_idx], col[keep_idx]

            edge_i.append(atom_idx[row])
            edge_j.append(atom_idx[col])
            vec = diff[row, col]
            edge_vec.append(vec)
            edge_dist.append(torch.linalg.norm(vec, dim=-1))

    if not edge_i:
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        return empty_edges, torch.empty(0, device=device), torch.empty((0, 3), device=device)

    edge_index = torch.stack([torch.cat(edge_i), torch.cat(edge_j)], dim=0)
    edge_weight = torch.cat(edge_dist)
    edge_vec = torch.cat(edge_vec)
    return edge_index, edge_weight, edge_vec
