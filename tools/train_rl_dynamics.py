#!/usr/bin/env python3
"""Train and validate the probabilistic dynamics ensemble offline."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.rl.dynamics_torch import (  # noqa: E402
    EnsembleDynamics, EnsembleTrainer, require_torch)
from tag_vision.rl.replay import ReplayBuffer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path,
                        default=ROOT / "artifacts/rl/replay.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch = require_torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; refusing silent CPU fallback")
        return 2
    replay = ReplayBuffer.load(args.replay)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.out or ROOT / "artifacts/rl/models" / stamp
    output.mkdir(parents=True, exist_ok=False)
    ensemble = EnsembleDynamics(
        replay.observation_size, replay.action_size,
        members=args.members, device=args.device)
    trainer = EnsembleTrainer(ensemble)
    metrics = trainer.fit(replay, epochs=args.epochs,
                          batch_size=args.batch_size)
    ensemble.save_checkpoint(output / "dynamics.pt", metadata=metrics)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {output / 'dynamics.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
