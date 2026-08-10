"""Validate teacher shards and build geometry-disjoint imitation datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from contract import policy_contract as pc
from train.segment_dataset import KINDS


REQUIRED_ARRAYS = (
    "observations", "actions", "next_observations", "rewards", "dones",
    "truncated", "episode_ids", "kinds", "successful_episode",
)


def build_geometry_split(dataset: dict, seed: int = 20260807,
                         validation_per_kind: int = 5,
                         test_per_kind: int = 5) -> dict[str, str]:
    """Assign whole geometries while retaining source diversity in holdouts."""
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for kind in KINDS:
        rows = [row for row in dataset["geometries"] if row["kind"] == kind]
        by_source = defaultdict(list)
        for row in rows:
            by_source[row["source"]].append(row["id"])
        for ids in by_source.values():
            rng.shuffle(ids)

        validation, test = [], []
        # If a source has at least three geometries, keep one unseen example
        # in each holdout while retaining at least one for training.
        for source in sorted(by_source):
            ids = by_source[source]
            if len(ids) >= 3:
                validation.append(ids.pop())
                test.append(ids.pop())

        remaining = [item for ids in by_source.values() for item in ids]
        rng.shuffle(remaining)
        while len(validation) < validation_per_kind:
            validation.append(remaining.pop())
        while len(test) < test_per_kind:
            test.append(remaining.pop())

        validation_set, test_set = set(validation), set(test)
        if validation_set & test_set:
            raise RuntimeError(f"overlapping holdouts for {kind}")
        for row in rows:
            geometry_id = row["id"]
            if geometry_id in validation_set:
                assignment[geometry_id] = "validation"
            elif geometry_id in test_set:
                assignment[geometry_id] = "test"
            else:
                assignment[geometry_id] = "train"
    return assignment


def _append(target: dict[str, list], **arrays) -> None:
    for name, values in arrays.items():
        target[name].append(values)


def _save_split(output: Path, chunks: dict[str, list]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        name: np.concatenate(values, axis=0) for name, values in chunks.items()
    }
    np.savez_compressed(output, **combined)
    return len(combined["observations"])


def _nested_counts(rows, key_a, key_b=None):
    if key_b is None:
        return dict(sorted(Counter(row[key_a] for row in rows).items()))
    nested = defaultdict(Counter)
    for row in rows:
        nested[row[key_a]][row[key_b]] += 1
    return {key: dict(sorted(value.items()))
            for key, value in sorted(nested.items())}


def prepare(shard_dir: Path, dataset_path: Path, output_dir: Path,
            seed: int = 20260807) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    geometries = {row["id"]: row for row in dataset["geometries"]}
    assignment = build_geometry_split(dataset, seed=seed)
    shard_paths = sorted(shard_dir.glob("*.part-*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no demonstration shards in {shard_dir}")

    episode_records: dict[str, dict] = {}
    duplicate_records = []
    for shard_path in shard_paths:
        report_path = shard_path.with_suffix(".json")
        if not report_path.exists():
            raise ValueError(f"missing report for {shard_path.name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report["episodes"]:
            episode_id = row["episode_id"]
            if episode_id in episode_records:
                duplicate_records.append(episode_id)
            episode_records[episode_id] = row
    if duplicate_records:
        raise ValueError(f"duplicate episode reports: {duplicate_records[:5]}")

    chunks = {
        split: defaultdict(list) for split in ("train", "validation", "test")
    }
    transition_counts = Counter()
    observed_episode_counts = Counter()
    bad_shards = []
    for shard_number, shard_path in enumerate(shard_paths, start=1):
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                missing = set(REQUIRED_ARRAYS) - set(data.files)
                if missing:
                    raise ValueError(f"missing arrays {sorted(missing)}")
                lengths = {name: len(data[name]) for name in REQUIRED_ARRAYS}
                if len(set(lengths.values())) != 1:
                    raise ValueError(f"array length mismatch {lengths}")
                observations = np.asarray(data["observations"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.float32)
                episode_ids = np.asarray(data["episode_ids"]).astype(str)
                kinds = np.asarray(data["kinds"]).astype(str)
                successes = np.asarray(data["successful_episode"], dtype=bool)
                dones = np.asarray(data["dones"], dtype=bool)
                if observations.shape[1:] != (pc.OBSERVATION_SIZE,):
                    raise ValueError(f"bad observation shape {observations.shape}")
                if actions.shape[1:] != (2,):
                    raise ValueError(f"bad action shape {actions.shape}")
                if not np.all(np.isfinite(observations)):
                    raise ValueError("non-finite observations")
                if not np.all(np.isfinite(actions)):
                    raise ValueError("non-finite actions")
                if np.max(np.abs(actions), initial=0.0) > 1.00001:
                    raise ValueError("action outside [-1, 1]")

                unique_ids, counts = np.unique(episode_ids, return_counts=True)
                for episode_id, count in zip(unique_ids, counts):
                    observed_episode_counts[episode_id] += int(count)
                    record = episode_records.get(episode_id)
                    if record is None:
                        raise ValueError(f"transition episode missing report: {episode_id}")
                    expected_success = record["outcome"] == "goal"
                    mask = episode_ids == episode_id
                    if not np.all(successes[mask] == expected_success):
                        raise ValueError(f"success flag mismatch: {episode_id}")
                    if int(np.sum(dones[mask])) != 1:
                        raise ValueError(f"episode must have one terminal row: {episode_id}")
                    if int(count) != int(record["steps"]):
                        raise ValueError(
                            f"transition/step mismatch for {episode_id}: "
                            f"{count} != {record['steps']}")

                successful_mask = successes
                successful_ids = episode_ids[successful_mask]
                geometry_ids = np.asarray([
                    episode_records[item]["geometry_id"] for item in successful_ids
                ])
                sources = np.asarray([
                    geometries[item]["source"] for item in geometry_ids
                ])
                split_labels = np.asarray([assignment[item] for item in geometry_ids])
                for split in chunks:
                    mask = successful_mask.copy()
                    mask[successful_mask] = split_labels == split
                    count = int(np.sum(mask))
                    transition_counts[split] += count
                    if not count:
                        continue
                    selected_ids = episode_ids[mask]
                    selected_geometry = np.asarray([
                        episode_records[item]["geometry_id"] for item in selected_ids
                    ])
                    selected_sources = np.asarray([
                        geometries[item]["source"] for item in selected_geometry
                    ])
                    _append(
                        chunks[split], observations=observations[mask],
                        actions=actions[mask], episode_ids=selected_ids,
                        geometry_ids=selected_geometry, kinds=kinds[mask],
                        sources=selected_sources)
        except Exception as error:
            bad_shards.append({"file": str(shard_path), "error": str(error)})
            raise
        print(f"[{shard_number:>3}/{len(shard_paths)}] validated {shard_path.name}")

    missing_transition_episodes = sorted(
        set(episode_records) - set(observed_episode_counts))
    unexpected_transition_episodes = sorted(
        set(observed_episode_counts) - set(episode_records))
    if missing_transition_episodes or unexpected_transition_episodes:
        raise ValueError(
            f"episode coverage mismatch: missing={len(missing_transition_episodes)}, "
            f"unexpected={len(unexpected_transition_episodes)}")

    output_files = {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.npz"
        count = _save_split(path, chunks[split])
        output_files[split] = {"path": str(path.resolve()), "transitions": count}

    records = list(episode_records.values())
    successful_records = [row for row in records if row["outcome"] == "goal"]
    failed_records = [row for row in records if row["outcome"] != "goal"]
    split_records = defaultdict(list)
    for row in successful_records:
        split_records[assignment[row["geometry_id"]]].append(row)

    geometry_rows = []
    for geometry_id, split in assignment.items():
        row = dict(geometries[geometry_id])
        geometry_rows.append({
            "id": geometry_id, "kind": row["kind"],
            "source": row["source"], "split": split,
        })
    report = {
        "schema": "imitation_corpus_v1",
        "valid": not bad_shards,
        "seed": seed,
        "source": {
            "shard_directory": str(shard_dir.resolve()),
            "shards": len(shard_paths),
            "episode_records": len(records),
            "transition_rows": int(sum(observed_episode_counts.values())),
        },
        "outcomes": dict(sorted(Counter(
            row["outcome"] for row in records).items())),
        "successful_episodes_retained": len(successful_records),
        "failed_episodes_excluded": len(failed_records),
        "failed_episodes": failed_records,
        "geometry_assignment": geometry_rows,
        "splits": {},
        "outputs": output_files,
        "checks": {
            "duplicate_episode_records": duplicate_records,
            "missing_transition_episodes": missing_transition_episodes,
            "unexpected_transition_episodes": unexpected_transition_episodes,
            "bad_shards": bad_shards,
        },
    }
    for split, rows in split_records.items():
        geometry_subset = [row for row in geometry_rows if row["split"] == split]
        report["splits"][split] = {
            "geometries": len(geometry_subset),
            "episodes": len(rows),
            "transitions": output_files[split]["transitions"],
            "episodes_by_kind": _nested_counts(rows, "kind"),
            "geometries_by_kind": _nested_counts(geometry_subset, "kind"),
            "geometries_by_source": _nested_counts(geometry_subset, "source"),
            "geometry_ids": sorted(row["id"] for row in geometry_subset),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "integrity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved integrity report to {report_path.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shard-dir",
        default="artifacts/local_segments/server_full_5000")
    parser.add_argument(
        "--dataset", default="artifacts/local_segments/balanced_dataset.json")
    parser.add_argument(
        "--out-dir", default="artifacts/local_segments/imitation_v1")
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    report = prepare(
        Path(args.shard_dir), Path(args.dataset), Path(args.out_dir), args.seed)
    print(json.dumps({
        "valid": report["valid"],
        "outcomes": report["outcomes"],
        "retained": report["successful_episodes_retained"],
        "excluded": report["failed_episodes_excluded"],
        "splits": report["splits"],
    }, indent=2))


if __name__ == "__main__":
    main()
