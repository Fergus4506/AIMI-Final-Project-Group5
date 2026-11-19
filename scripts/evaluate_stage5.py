"""Evaluate frozen RETFound-Green features with the approved APTOS 5-NN rule."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np

from extract_stage4 import ROOT, load_manifests, validate_features
from prepare_stage3 import sha256

K = 5
GRADES = range(5)
TEST_SPLITS = {"aptos_test": "APTOS", "idrid_test": "IDRiD"}


def read_split(stage4, manifest, metadata):
    manifest_hash, rows = manifest
    path = stage4 / metadata["filename"]
    if manifest_hash != metadata["manifest_sha256"] or sha256(path) != metadata["feature_sha256"]:
        raise ValueError(f"Manifest or feature file changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        ids = data["image_ids"]
    expected_ids = [row["image_id"] for row in rows]
    if ids.tolist() != expected_ids or features.shape != tuple(metadata["shape"]):
        raise ValueError(f"Stage 4 feature IDs or shape do not match: {path}")
    validate_features(np, features, len(rows))
    labels = {row["image_id"]: int(row["grade"]) for row in rows}
    if set(labels.values()) != set(GRADES):
        raise ValueError(f"Expected all five DR grades in {path}")
    normalized = features.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    return ids, labels, normalized


def expand_aptos_test(reference, test, fraction, seed):
    ref_ids, ref_labels, ref_features = reference
    test_ids, test_labels, test_features = test
    if set(ref_ids).intersection(test_ids):
        raise ValueError("APTOS reference and test sets overlap")
    rng = random.Random(seed)
    moved_indices = []
    for grade in GRADES:
        candidates = [i for i, image_id in enumerate(ref_ids) if ref_labels[image_id] == grade]
        candidates.sort(key=lambda i: ref_ids[i])
        test_count = sum(label == grade for label in test_labels.values())
        additional = round((test_count + len(candidates)) * fraction) - test_count
        if not 0 <= additional < len(candidates):
            raise ValueError(f"Requested fraction cannot expand DR {grade} while retaining references")
        rng.shuffle(candidates)
        moved_indices.extend(candidates[:additional])
    if not moved_indices:
        raise ValueError("Requested fraction adds no test images")
    moved = sorted(moved_indices)
    moved_ids = ref_ids[moved]
    keep = np.ones(len(ref_ids), dtype=bool)
    keep[moved] = False
    reduced_ref = (ref_ids[keep], {i: ref_labels[i] for i in ref_ids[keep]}, ref_features[keep])
    expanded_test = (np.concatenate((test_ids, moved_ids)),
                     {**test_labels, **{i: ref_labels[i] for i in moved_ids}},
                     np.concatenate((test_features, ref_features[moved])))
    return reduced_ref, expanded_test, moved_ids.tolist()


def predict(test_ids, test_features, reference):
    ref_ids, labels, ref_features = reference
    by_id = np.argsort(ref_ids)
    ref_ids = ref_ids[by_id]
    ref_grades = np.array([labels[image_id] for image_id in ref_ids])
    similarity = test_features @ ref_features[by_id].T
    # Sorting stable on the ascending reference IDs resolves exact similarity ties.
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :K]
    for i, image_id in enumerate(test_ids):
        neighbors = nearest[i]
        grades = ref_grades[neighbors]
        votes = np.bincount(grades, minlength=len(GRADES))
        grade = next(int(candidate) for candidate in grades if votes[candidate] == votes.max())
        details = list(zip(ref_ids[neighbors], grades, similarity[i, neighbors]))
        yield image_id, grade, votes, details


def score(predictions, truth):
    ids = [row[0] for row in predictions]
    if len(ids) != len(truth) or set(ids) != set(truth):
        raise ValueError("Prediction and ground-truth IDs must match one to one")
    matrix = np.zeros((len(GRADES), len(GRADES)), dtype=int)
    for image_id, grade, _, _ in predictions:
        matrix[truth[image_id], grade] += 1
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Cannot score a split without all five grades")
    correct = np.diag(matrix)
    recall = correct / support
    f1 = 2 * correct / (support + matrix.sum(axis=0))
    return {
        "count": len(ids),
        "correct": int(correct.sum()),
        "accuracy": float(correct.sum() / len(ids)),
        "macro_f1": float(f1.mean()),
        "recall_by_grade": recall.tolist(),
        "f1_by_grade": f1.tolist(),
        "support_by_grade": support.tolist(),
        "confusion_matrix_true_rows": matrix.tolist(),
    }


def write_predictions(path, predictions, truth):
    fields = ["image_id", "true_grade", "predicted_grade", *(f"votes_{g}" for g in GRADES)]
    for rank in range(1, K + 1):
        fields.extend(f"neighbor_{rank}_{key}" for key in ("id", "grade", "cosine"))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for image_id, grade, votes, details in predictions:
            neighbors = [value for name, label, cosine in details for value in (name, label, float(cosine))]
            writer.writerow([image_id, truth[image_id], grade, *votes, *neighbors])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3", type=Path, default=ROOT / "artifacts/stage3")
    parser.add_argument("--stage4", type=Path, default=ROOT / "artifacts/stage4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aptos-test-fraction", type=float,
                        help="Expand the APTOS test split while retaining all original test IDs")
    args = parser.parse_args()
    if args.aptos_test_fraction is not None:
        if args.output is None:
            parser.error("provide a separate --output for the expanded APTOS split")
        if not 0 < args.aptos_test_fraction < 1:
            parser.error("--aptos-test-fraction must lie between 0 and 1")
    output = args.output if args.output is not None else ROOT / "artifacts/stage5"
    if output.exists():
        parser.error(f"output already exists: {output}")
    stage3_audit = json.loads((args.stage3 / "audit.json").read_text(encoding="utf-8"))
    stage4_audit = json.loads((args.stage4 / "audit.json").read_text(encoding="utf-8"))
    rules = {"k": K, "similarity": "cosine", "voting": "uniform",
             "vote_tie": "closest neighbor among tied grades", "similarity_tie": "ascending image ID"}
    if any(stage3_audit["evaluation"][key] != value for key, value in rules.items()):
        raise ValueError("Stage 3 evaluation rules do not match this classifier")
    manifests = load_manifests(args.stage3)
    datasets = {split: read_split(args.stage4, manifest, stage4_audit["splits"][split])
                for split, manifest in manifests.items()}
    reference = datasets["aptos_reference"]
    original_test_labels = datasets["aptos_test"][1]
    fraction = args.aptos_test_fraction
    if fraction is not None:
        reference, datasets["aptos_test"], moved_ids = expand_aptos_test(
            reference, datasets["aptos_test"], fraction, stage3_audit["seed"]
        )
    if len(reference[0]) < K:
        raise ValueError("Insufficient reference images for 5-NN")
    audit = {
        "method": "Frozen RETFound-Green features + APTOS reference 5-NN",
        "checkpoint": stage4_audit["checkpoint"],
        "grade_order": list(GRADES),
        "evaluation": rules,
        "numpy_version": np.__version__,
        "inputs": {split: {key: info[key] for key in ("manifest_sha256", "feature_sha256")}
                   for split, info in stage4_audit["splits"].items()},
        "reference_split": "aptos_reference",
        "reference_count": len(reference[0]),
        "limitations": stage3_audit["limitations"],
        "datasets": {},
    }
    if fraction is not None:
        aptos_count = len(datasets["aptos_test"][0])
        audit["aptos_resplit"] = {
            "requested_test_fraction": fraction,
            "actual_test_fraction": aptos_count / (len(reference[0]) + aptos_count),
            "seed": stage3_audit["seed"],
            "original_test_count": len(original_test_labels),
            "moved_from_reference_image_ids": moved_ids,
        }
    results = {}
    for split in TEST_SPLITS:
        ids, truth, features = datasets[split]
        predictions = list(predict(ids, features, reference))
        results[split] = (predictions, truth)
        audit["datasets"][split] = score(predictions, truth)
    if fraction is not None:
        common = [prediction for prediction in results["aptos_test"][0]
                  if prediction[0] in original_test_labels]
        audit["aptos_original_holdout"] = score(common, original_test_labels)
    aptos, idrid = (audit["datasets"][split] for split in TEST_SPLITS)
    audit["macro_f1_gap_aptos_minus_idrid"] = aptos["macro_f1"] - idrid["macro_f1"]

    output.mkdir(parents=True)
    for split, (predictions, truth) in results.items():
        path = output / f"{split}_predictions.csv"
        write_predictions(path, predictions, truth)
        audit["datasets"][split]["predictions_file"] = path.name
        audit["datasets"][split]["predictions_sha256"] = sha256(path)
    (output / "metrics.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for split, metrics in audit["datasets"].items():
        print(f"{TEST_SPLITS[split]}: N={metrics['count']}, accuracy={metrics['accuracy']:.6f}, "
              f"macro-F1={metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
