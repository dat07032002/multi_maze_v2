"""Train one route-conditioned policy from successful MPC demonstrations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from contract import policy_contract as pc
from control.imitation_policy import RouteConditionedPolicy
from train.segment_dataset import KINDS


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {
            "observations": np.asarray(data["observations"], dtype=np.float32),
            "actions": np.asarray(data["actions"], dtype=np.float32),
            "kinds": np.asarray(data["kinds"]).astype(str),
            "episode_ids": np.asarray(data["episode_ids"]).astype(str),
            "geometry_ids": np.asarray(data["geometry_ids"]).astype(str),
            "sources": np.asarray(data["sources"]).astype(str),
        }
    length = len(result["observations"])
    if any(len(value) != length for value in result.values()):
        raise ValueError(f"array length mismatch in {path}")
    if result["observations"].shape[1:] != (pc.OBSERVATION_SIZE,):
        raise ValueError(f"unexpected observation shape in {path}")
    if result["actions"].shape[1:] != (2,):
        raise ValueError(f"unexpected action shape in {path}")
    return result


def select_policy_observations(data: dict[str, np.ndarray],
                               omit_clearance: bool) -> dict[str, np.ndarray]:
    """Optionally expose the pre-clearance observation prefix to the policy."""
    if not omit_clearance:
        return data
    legacy_size = pc.OBSERVATION_SIZE - pc.LOOKAHEAD_COUNT
    result = dict(data)
    result["observations"] = data["observations"][..., :legacy_size]
    return result


class BalancedBatcher:
    """Uniform-over-maneuver minibatches sampled from transition arrays."""

    def __init__(self, kinds: np.ndarray, seed: int):
        self.rng = np.random.default_rng(seed)
        self.indices = {
            kind: np.flatnonzero(kinds == kind) for kind in KINDS
        }
        missing = [kind for kind, indices in self.indices.items() if not len(indices)]
        if missing:
            raise ValueError(f"missing maneuver classes: {missing}")

    def sample(self, batch_size: int) -> np.ndarray:
        base, remainder = divmod(batch_size, len(KINDS))
        pieces = []
        for index, kind in enumerate(KINDS):
            count = base + (index < remainder)
            pieces.append(self.rng.choice(
                self.indices[kind], size=count, replace=True))
        result = np.concatenate(pieces)
        self.rng.shuffle(result)
        return result


@torch.no_grad()
def evaluate(model: nn.Module, data: dict[str, np.ndarray], device,
             batch_size: int = 32768) -> dict:
    model.eval()
    predictions = []
    observations = data["observations"]
    for start in range(0, len(observations), batch_size):
        batch = torch.from_numpy(observations[start:start + batch_size]).to(
            device, non_blocking=True)
        predictions.append(model(batch).float().cpu().numpy())
    prediction = np.concatenate(predictions)
    error = prediction - data["actions"]
    by_kind = {}
    for kind in KINDS:
        mask = data["kinds"] == kind
        by_kind[kind] = {
            "transitions": int(np.sum(mask)),
            "mse": float(np.mean(np.square(error[mask]))),
            "mae": float(np.mean(np.abs(error[mask]))),
            "rmse": float(np.sqrt(np.mean(np.square(error[mask])))),
        }
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "balanced_mse": float(np.mean([
            row["mse"] for row in by_kind.values()])),
        "by_kind": by_kind,
    }


def save_checkpoint(path: Path, model: RouteConditionedPolicy, optimizer,
                    epoch: int, metrics: dict, hidden_sizes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema": "route_conditioned_bc_v1",
        "epoch": epoch,
        "model_config": {"hidden_sizes": list(hidden_sizes)},
        "observation_mean": model.observation_mean.detach().cpu().numpy(),
        "observation_std": model.observation_std.detach().cpu().numpy(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, temporary)
    temporary.replace(path)


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    data_dir = Path(args.data_dir)
    train_data = select_policy_observations(
        load_split(data_dir / "train.npz"), args.omit_clearance)
    validation_data = select_policy_observations(
        load_split(data_dir / "validation.npz"), args.omit_clearance)
    test_data = select_policy_observations(
        load_split(data_dir / "test.npz"), args.omit_clearance)
    mean = train_data["observations"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_data["observations"].std(axis=0, dtype=np.float64).astype(np.float32)
    hidden_sizes = tuple(args.hidden_sizes)
    model = RouteConditionedPolicy(mean, std, hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    batcher = BalancedBatcher(train_data["kinds"], args.seed)
    steps_per_epoch = args.steps_per_epoch or math.ceil(
        len(train_data["observations"]) / args.batch_size)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output_dir / "tb"))
    except ImportError:
        writer = None

    best_validation = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            indices = batcher.sample(args.batch_size)
            observations = torch.from_numpy(
                train_data["observations"][indices]).to(
                    device, non_blocking=True)
            targets = torch.from_numpy(train_data["actions"][indices]).to(
                device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda"):
                predictions = model(observations)
                loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate(model, validation_data, device)
        train_loss = float(np.mean(losses))
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch, "train_mse": train_loss,
            "validation": validation, "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if writer is not None:
            writer.add_scalar("loss/train_mse", train_loss, epoch)
            writer.add_scalar("loss/validation_mse", validation["mse"], epoch)
            writer.add_scalar(
                "loss/validation_balanced_mse", validation["balanced_mse"], epoch)
            for kind, values in validation["by_kind"].items():
                writer.add_scalar(f"validation/{kind}_mse", values["mse"], epoch)

        improved = validation["balanced_mse"] < best_validation - args.min_delta
        if improved:
            best_validation = validation["balanced_mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best_model.pt", model, optimizer, epoch, row,
                hidden_sizes)
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            output_dir / "last_model.pt", model, optimizer, epoch, row,
            hidden_sizes)
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} "
            f"val={validation['mse']:.6f} "
            f"balanced={validation['balanced_mse']:.6f} "
            f"best={best_validation:.6f}@{best_epoch} "
            f"elapsed={elapsed:.1f}s")
        if epochs_without_improvement >= args.patience:
            print(f"early stopping after {args.patience} unimproved epochs")
            break

    if writer is not None:
        writer.close()
    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_metrics = evaluate(model, test_data, device)
    example = torch.zeros(1, model.observation_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(output_dir / "best_policy.ts"))

    report = {
        "schema": "route_conditioned_bc_training_v1",
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "model": {
            "hidden_sizes": list(hidden_sizes),
            "observation_size": model.observation_size,
            "omit_clearance": args.omit_clearance,
            "parameters": sum(parameter.numel()
                              for parameter in model.parameters()),
        },
        "data": {
            "train_transitions": len(train_data["observations"]),
            "validation_transitions": len(validation_data["observations"]),
            "test_transitions": len(test_data["observations"]),
            "train_kind_counts": dict(sorted(Counter(
                train_data["kinds"]).items())),
        },
        "training": {
            "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_validation_balanced_mse": best_validation,
            "batch_size": args.batch_size,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": args.learning_rate,
        },
        "test": test_metrics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", default="artifacts/local_segments/imitation_v1")
    parser.add_argument(
        "--out-dir", default="artifacts/local_segments/bc_policy_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument(
        "--omit-clearance", action="store_true",
        help="train a 22-feature ablation on the pre-clearance prefix")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=(256, 256, 128))
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
