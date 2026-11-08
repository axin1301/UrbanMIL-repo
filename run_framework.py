"""Staged UrbanMIL training with visual retrieval and semantic refinement."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GROUPS = {
    "urban_form": ("building_density", "vertical_density", "road_density",
                   "impervious_surface", "block_compactness", "urban_sprawl"),
    "economic_function": ("commercial_intensity", "industrial_intensity",
                          "business_activity", "night_economy_potential",
                          "logistics_activity", "land_use_mix"),
    "livability": ("residential_quality", "building_quality", "green_amenity",
                   "walkability", "street_cleanliness", "public_service_access"),
    "mobility": ("transport_accessibility", "road_hierarchy", "parking_intensity",
                 "traffic_activity", "connectivity", "transit_oriented_development"),
    "socioeconomic_prior": ("income_level_prior", "urban_development_level",
                            "informality_probability", "energy_consumption_intensity",
                            "construction_activity", "urban_maturity"),
}
ATTRIBUTE_NAMES = [f"{group}:{name}" for group, names in GROUPS.items()
                   for name in names]


def score(truth, prediction):
    y, p = truth.double().cpu().numpy(), prediction.double().cpu().numpy()
    error = y - p
    variance = ((y - y.mean()) ** 2).sum()
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "r2": float(1 - (error ** 2).sum() / variance) if variance > 0 else 0.0,
            "pcc": float(np.corrcoef(y, p)[0, 1]) if y.std() > 1e-12
            and p.std() > 1e-12 else 0.0}


def load_inputs(args):
    rows = json.loads(args.task_json.read_text(encoding="utf-8"))
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    parts = {key: [str(x) for x in split[key + "_ids"]]
             for key in ("train", "val", "test")}
    ids = parts["train"] + parts["val"] + parts["test"]
    task = {str(row.get("region_id", f"sample_{i:06d}")): row
            for i, row in enumerate(rows)}
    if len(ids) != len(set(ids)) or len(task) != len(rows) or set(ids) != set(task):
        raise ValueError("Split must partition task regions exactly")
    visual = torch.load(args.visual_features, map_location="cpu", weights_only=False)
    sat, stv, mask, labels = [], [], [], []
    for rid in ids:
        entry = visual[rid]
        s = torch.as_tensor(entry["sat"]).float().reshape(-1)
        v = torch.as_tensor(entry["street"]).float()
        m = torch.as_tensor(entry["street_mask"]).bool().reshape(-1)
        y = float(task[rid]["reference_normalized"])
        if (v.shape != (10, s.numel()) or m.shape != (10,) or
                not math.isfinite(y) or not torch.isfinite(s).all() or
                not torch.isfinite(v).all()):
            raise ValueError(f"Invalid label or visual feature: {rid}")
        sat.append(s); stv.append(v); mask.append(m); labels.append(y)
    n_train, n_val = len(parts["train"]), len(parts["val"])
    return {"ids": ids, "sat": F.normalize(torch.stack(sat), dim=-1),
            "street": F.normalize(torch.stack(stv), dim=-1),
            "mask": torch.stack(mask), "labels": torch.tensor(labels),
            "train": torch.arange(n_train),
            "val": torch.arange(n_train, n_train + n_val),
            "test": torch.arange(n_train + n_val, len(ids))}


def load_semantics(path, data):
    if path.is_dir():
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(path.glob("*.json"))]
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        content = json.loads(path.read_text(encoding="utf-8"))
        rows = content if isinstance(content, list) else content.get("regions", [content])
    profiles = {str(row["region_id"]): row for row in rows}
    if len(profiles) != len(rows) or set(profiles) != set(data["ids"]):
        raise ValueError("Semantic profile IDs must match task IDs exactly")
    values, observed, uncertainties = [], [], []
    for rid in data["ids"]:
        row, scores, masks = profiles[rid], [], []
        for group, names in GROUPS.items():
            block = row[group]
            if set(block) != set(names):
                raise ValueError(f"Incorrect semantic group {group}: {rid}")
            for name in names:
                value = block[name]
                if value is None:
                    scores.append(0.0); masks.append(0.0)
                elif (isinstance(value, bool) or not isinstance(value, (float, int))
                      or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"Invalid semantic value {group}.{name}: {rid}")
                else:
                    scores.append(float(value)); masks.append(1.0)
        u = row["uncertainty"]
        if (isinstance(u, bool) or not isinstance(u, (float, int)) or
                not math.isfinite(u) or not 0 <= u <= 1):
            raise ValueError(f"Invalid uncertainty: {rid}")
        values.append(scores); observed.append(masks); uncertainties.append(float(u))
    values, observed = torch.tensor(values), torch.tensor(observed)
    uncertainty = torch.tensor(uncertainties)
    train = data["train"]
    count = observed[train].sum(0)
    mean = (values[train] * observed[train]).sum(0) / count.clamp_min(1)
    variance = (((values[train] - mean).square() * observed[train]).sum(0) /
                count.clamp_min(1))
    scaled = ((values - mean) / variance.sqrt().clamp_min(0.05)).clamp(-4, 4)
    state = torch.stack((scaled * observed, observed * (1 - uncertainty[:, None])), -1)
    return state, {"attributes": ATTRIBUTE_NAMES,
                   "train_observed_rate": float((count / len(train)).mean()),
                   "train_uncertainty_mean": float(uncertainty[train].mean())}


@torch.no_grad()
def retrieval_candidates(data, query, count, device, chunk, exclude_self=False):
    bank = data["train"]
    sat_bank = data["sat"][bank].to(device)
    stv_bank = data["street"][bank].to(device)
    mask_bank = data["mask"][bank].to(device)
    sat_rows, stv_rows = [], []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        sat_rows.append((data["sat"][q].to(device) @ sat_bank.T).cpu())
        views, view_mask = data["street"][q].to(device), data["mask"][q].to(device)
        sim = torch.einsum("bvd,nwd->bnvw", views, stv_bank)
        sim = sim.masked_fill(~(view_mask[:, None, :, None] &
                                mask_bank[None, :, None, :]), -1e4)
        forward = ((sim.max(-1).values * view_mask[:, None]).sum(-1) /
                   view_mask.sum(-1)[:, None].clamp_min(1))
        reverse = ((sim.max(-2).values * mask_bank[None]).sum(-1) /
                   mask_bank.sum(-1)[None].clamp_min(1))
        stv_rows.append((0.5 * (forward + reverse)).cpu())
    sat, stv = torch.cat(sat_rows), torch.cat(stv_rows)
    if exclude_self:
        sat[torch.arange(len(query)), query] = -torch.inf
        stv[torch.arange(len(query)), query] = -torch.inf
    rank_sat = (-sat).argsort(-1).argsort(-1).float()
    rank_stv = (-stv).argsort(-1).argsort(-1).float()
    fused = 0.5 * (1 / (61 + rank_sat) + 1 / (61 + rank_stv))
    sat_order = sat.argsort(-1, descending=True)
    stv_order = stv.argsort(-1, descending=True)
    fused_order = fused.argsort(-1, descending=True)
    k = min(count, len(bank) - int(exclude_self))
    candidates, relations = [], []
    for row in range(len(query)):
        selected, seen = [], set()
        for order, budget in ((sat_order[row], (k + 1) // 2),
                              (stv_order[row], k // 2), (fused_order[row], k)):
            added = 0
            for index in order.tolist():
                if index in seen:
                    continue
                selected.append(index); seen.add(index); added += 1
                if len(selected) >= k or added >= budget:
                    break
            if len(selected) >= k:
                break
        local = torch.tensor(selected[:k], dtype=torch.long)
        local = local[fused[row, local].argsort(descending=True)]
        chosen = bank[local]
        if exclude_self and (chosen == query[row]).any():
            raise RuntimeError("Training query was selected as its own support")
        candidates.append(chosen)
        relations.append(torch.stack((sat[row, local], stv[row, local]), -1))
    relation = torch.stack(relations)
    relation = ((relation - relation.mean(1, keepdim=True)) /
                relation.std(1, keepdim=True).clamp_min(1e-4))
    return torch.stack(candidates), relation


class RetrievalWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_logits = nn.Parameter(torch.zeros(2))
        self.raw_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, relation):
        channel = self.channel_logits.softmax(0)
        return (F.softplus(self.raw_strength) - math.log(2.0)) * (
            relation * channel).sum(-1)


class DirectionalVisual(nn.Module):
    def __init__(self, retrieval, dimension):
        super().__init__()
        self.retrieval = copy.deepcopy(retrieval).eval()
        for parameter in self.retrieval.parameters():
            parameter.requires_grad_(False)
        self.sat_value = nn.Linear(dimension, 1, bias=False)
        self.stv_value = nn.Linear(dimension, 1, bias=False)
        self.stv_attention = nn.Linear(dimension, 1, bias=False)
        for layer in (self.sat_value, self.stv_value, self.stv_attention):
            nn.init.zeros_(layer.weight)

    def potential(self, sat, views, mask):
        logits = self.stv_attention(views).squeeze(-1).masked_fill(~mask, -1e4)
        weight = logits.softmax(-1) * mask.float()
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        return self.sat_value(sat).squeeze(-1) + (
            weight * self.stv_value(views).squeeze(-1)).sum(-1)

    def forward(self, relation, supports, query, candidates, data, device):
        weights = self.retrieval(relation).softmax(-1)
        q = self.potential(data["sat"][query].to(device),
                           data["street"][query].to(device),
                           data["mask"][query].to(device))
        n, k = candidates.shape
        s = self.potential(
            data["sat"][candidates].reshape(n * k, -1).to(device),
            data["street"][candidates].reshape(n * k, 10, -1).to(device),
            data["mask"][candidates].reshape(n * k, 10).to(device)).reshape(n, k)
        return (weights * (supports + q[:, None] - s)).sum(-1), weights


class SemanticRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_weight = nn.Parameter(torch.zeros(1, 30))
        self.pair_weight = nn.Parameter(torch.zeros(1, 30))

    def forward(self, query, support, weights):
        qv, qc = query[..., 0], query[..., 1]
        sv, sc = support[..., 0], support[..., 1]
        query_effect = (qv * qc * self.query_weight).sum(-1)
        pair_effect = ((qv[:, None] - sv) * qc[:, None] * sc *
                       self.pair_weight).sum(-1)
        return query_effect + (weights * pair_effect).sum(-1)


@torch.no_grad()
def predict_visual(model, data, queries, candidates, relation, args, device):
    model.eval()
    predictions, weights = [], []
    for batch in torch.arange(len(queries)).split(args.batch_size):
        q, s = queries[batch], candidates[batch]
        prediction, weight = model(relation[batch].to(device),
                                   data["labels"][s].to(device), q, s, data, device)
        predictions.append(prediction.cpu()); weights.append(weight.cpu())
    return torch.cat(predictions), torch.cat(weights)


def fit_retrieval(data, tc, tr, vc, vr, args, device):
    model = RetrievalWeights().to(device)
    train, val = data["train"], data["val"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.visual_lr, weight_decay=0)
    best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    generator = torch.Generator().manual_seed(args.seed + 101)
    for epoch in range(args.visual_epochs + 1):
        if epoch:
            for batch in torch.randperm(len(train), generator=generator).split(args.batch_size):
                weight = model(tr[batch].to(device)).softmax(-1)
                prediction = (weight * data["labels"][tc[batch]].to(device)).sum(-1)
                loss = F.mse_loss(prediction, data["labels"][train[batch]].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            weight = model(vr.to(device)).softmax(-1)
            prediction = (weight * data["labels"][vc].to(device)).sum(-1).cpu()
        rmse = score(data["labels"][val], prediction)["rmse"]
        if rmse < best_rmse - 1e-4:
            best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), rmse, epoch, 0
        else:
            stale += 1
        if epoch and stale >= args.patience:
            break
    model.load_state_dict(best)
    return model, best_epoch, best_rmse


def fit_visual(retrieval, data, tc, tr, vc, vr, args, device):
    model = DirectionalVisual(retrieval, data["sat"].shape[-1]).to(device)
    train, val = data["train"], data["val"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.visual_transport_lr,
                                  weight_decay=args.weight_decay)
    best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(args.visual_transport_epochs + 1):
        if epoch:
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(
                args.seed + 401 + epoch))
            for batch in order.split(args.batch_size):
                q, s = train[batch], tc[batch]
                prediction, _ = model(tr[batch].to(device),
                                      data["labels"][s].to(device), q, s, data, device)
                loss = F.mse_loss(prediction, data["labels"][q].to(device))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
        prediction, _ = predict_visual(model, data, val, vc, vr, args, device)
        rmse = score(data["labels"][val], prediction)["rmse"]
        if rmse < best_rmse - 1e-4:
            best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), rmse, epoch, 0
        else:
            stale += 1
        if epoch and stale >= args.patience:
            break
    model.load_state_dict(best)
    return model, best_epoch, best_rmse


def fit_semantic(visual, data, state, tc, tr, vc, vr, args, device):
    train, val = data["train"], data["val"]
    train_base, train_weight = predict_visual(visual, data, train, tc, tr, args, device)
    val_base, val_weight = predict_visual(visual, data, val, vc, vr, args, device)
    model = SemanticRefinement().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.semantic_lr,
                                  weight_decay=args.weight_decay)
    best, best_rmse, best_epoch, stale = None, float("inf"), -1, 0
    history = []
    for epoch in range(1, args.semantic_epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            args.seed + 3000 + epoch))
        total = 0.0
        for batch in order.split(args.batch_size):
            q, s = train[batch], tc[batch]
            correction = model(state[q].to(device), state[s].to(device),
                               train_weight[batch].to(device))
            prediction = train_base[batch].to(device) + correction
            loss = F.mse_loss(prediction, data["labels"][q].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            prediction = val_base.to(device) + model(state[val].to(device),
                state[vc].to(device), val_weight.to(device))
        rmse = score(data["labels"][val], prediction.cpu())["rmse"]
        history.append({"epoch": epoch, "train_loss": total / len(train),
                        "val_rmse": rmse})
        if rmse < best_rmse - 1e-3:
            best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), rmse, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"semantic epoch={epoch} train_mse={total / len(train):.3f} "
                  f"val_rmse={rmse:.3f} best={best_epoch}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    return model, best_epoch, best_rmse, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("visual_features", "semantic_profiles", "task_json",
                "split_json", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--candidate_count", type=int, default=64)
    parser.add_argument("--visual_epochs", type=int, default=60)
    parser.add_argument("--visual_transport_epochs", type=int, default=50)
    parser.add_argument("--semantic_epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--visual_chunk", type=int, default=8)
    parser.add_argument("--visual_lr", type=float, default=2e-2)
    parser.add_argument("--visual_transport_lr", type=float, default=1e-3)
    parser.add_argument("--semantic_lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for key in ("visual_features", "semantic_profiles", "task_json", "split_json"):
        if not getattr(args, key).exists():
            parser.error(f"Missing input: {getattr(args, key)}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory must be empty: {args.output}")
    if min(args.candidate_count, args.visual_epochs, args.visual_transport_epochs,
           args.semantic_epochs, args.patience, args.batch_size, args.visual_chunk) < 1:
        parser.error("Counts and epoch limits must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data = load_inputs(args)
    state, semantic_info = load_semantics(args.semantic_profiles, data)
    print("Building train/validation support pools...", flush=True)
    tc, tr = retrieval_candidates(data, data["train"], args.candidate_count,
                                   device, args.visual_chunk, True)
    vc, vr = retrieval_candidates(data, data["val"], args.candidate_count,
                                   device, args.visual_chunk)
    retrieval, re, rv = fit_retrieval(data, tc, tr, vc, vr, args, device)
    visual, ve, vv = fit_visual(retrieval, data, tc, tr, vc, vr, args, device)
    semantic, se, sv, history = fit_semantic(visual, data, state,
                                             tc, tr, vc, vr, args, device)
    print("Building held-out test support pool...", flush=True)
    xc, xr = retrieval_candidates(data, data["test"], args.candidate_count,
                                   device, args.visual_chunk)
    test_base, test_weight = predict_visual(visual, data, data["test"],
                                            xc, xr, args, device)
    semantic.eval()
    with torch.no_grad():
        correction = semantic(state[data["test"]].to(device),
                              state[xc].to(device), test_weight.to(device)).cpu()
    final = test_base + correction
    truth = data["labels"][data["test"]]
    visual_test, final_test = score(truth, test_base), score(truth, final)
    args.output.mkdir(parents=True, exist_ok=True)
    visual_dir, semantic_dir = args.output / "visual", args.output / "semantic"
    visual_dir.mkdir(); semantic_dir.mkdir()
    torch.save({"retrieval": retrieval.state_dict(), "visual": visual.state_dict(),
                "retrieval_epoch": re, "visual_epoch": ve}, visual_dir / "model.pt")
    torch.save({"semantic": semantic.state_dict(), "attributes": ATTRIBUTE_NAMES,
                "best_epoch": se}, semantic_dir / "model.pt")
    (semantic_dir / "predictions.json").write_text(json.dumps([
        {"region_id": data["ids"][int(i)], "truth": float(y),
         "visual_prediction": float(v), "prediction": float(p)}
        for i, y, v, p in zip(data["test"], truth, test_base, final)
    ], indent=2), encoding="utf-8")
    visual_report = {"retrieval_validation_rmse": rv,
                     "visual_transport_validation_rmse": vv,
                     "visual_test": visual_test, "visual_epoch": re,
                     "visual_transport_epoch": ve}
    semantic_report = {"validation_rmse": sv, "test": final_test,
                       "best_epoch": se, "semantic_attributes": semantic_info,
                       "history": history,
                       "mean_absolute_test_correction": float(correction.abs().mean())}
    (visual_dir / "metrics.json").write_text(json.dumps(visual_report, indent=2),
                                             encoding="utf-8")
    (semantic_dir / "metrics.json").write_text(json.dumps(semantic_report, indent=2),
                                               encoding="utf-8")
    summary = {"inputs": {key: str(getattr(args, key).resolve()) for key in
                          ("visual_features", "semantic_profiles", "task_json", "split_json")},
               "candidate_count": args.candidate_count,
               "label_field": "reference_normalized",
               "visual_validation_rmse": vv, "visual_test": visual_test,
               "final_validation_rmse": sv, "final_test": final_test,
               "selected_epochs": {"retrieval": re,
                                   "visual_adjustment": ve,
                                   "semantic_refinement": se}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
