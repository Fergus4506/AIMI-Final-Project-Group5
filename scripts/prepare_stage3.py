"""Audit the official APTOS and IDRiD splits and build stage 3 manifests."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random

from PIL import Image, ImageOps

GRADES = range(5)
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(dataset, labels_path, images_path, id_column, grade_column, excluded):
    images = defaultdict(list)
    labels = defaultdict(list)

    def exclude(image_id, reason, detail=""):
        excluded.append((dataset, image_id, reason, detail))

    for path in sorted(images_path.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.stem].append(path)

    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        if not {id_column, grade_column}.issubset(reader.fieldnames):
            raise ValueError("Unexpected CSV columns: {}".format(labels_path))
        for line, row in enumerate(reader, 2):
            image_id = (row[id_column] or "").strip()
            if not image_id:
                exclude("", "missing_id", str(line))
                continue
            if Path(image_id).suffix.lower() in IMAGE_SUFFIXES:
                image_id = Path(image_id).stem
            labels[image_id].append((row[grade_column] or "").strip())

    for image_id in sorted(images.keys() - labels.keys()):
        exclude(image_id, "missing_label", str(images[image_id][0]))

    by_pixels = defaultdict(list)
    for image_id, grades in sorted(labels.items()):
        if len(set(grades)) != 1 or grades[0] not in {"0", "1", "2", "3", "4"}:
            exclude(image_id, "conflicting_or_invalid_grade", repr(grades))
            continue
        if len(images[image_id]) != 1:
            exclude(image_id, "missing_or_ambiguous_image", repr(images[image_id]))
            continue
        path = images[image_id][0]
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                digest = hashlib.sha256(str(image.size).encode("ascii"))
                digest.update(image.tobytes())
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            exclude(image_id, "decode_error", str(error))
            continue
        by_pixels[digest.hexdigest()].append({
            "image_id": image_id,
            "relative_path": Path(os.path.relpath(path, ROOT)).as_posix(),
            "grade": int(grades[0]),
            "pixel_sha256": digest.hexdigest(),
        })

    valid = []
    for duplicates in by_pixels.values():
        if len({row["grade"] for row in duplicates}) > 1:
            for row in duplicates:
                exclude(row["image_id"], "conflicting_duplicate_grade",
                        duplicates[0]["image_id"])
            continue
        valid.append(duplicates[0])
        for row in duplicates[1:]:
            exclude(row["image_id"], "duplicate_image", duplicates[0]["image_id"])
    return valid


def split_aptos(records):
    rng = random.Random(SEED)
    reference, test = [], []
    for grade in GRADES:
        rows = sorted((row for row in records if row["grade"] == grade),
                      key=lambda row: row["image_id"])
        if len(rows) < 2:
            raise ValueError("APTOS grade {} has {} valid images, needs at least 2"
                             .format(grade, len(rows)))
        rng.shuffle(rows)
        test_size = min(len(rows) - 1, max(1, round(len(rows) * 0.2)))
        test.extend(rows[:test_size])
        reference.extend(rows[test_size:])
    return reference, test


def class_counts(records):
    counts = Counter(row["grade"] for row in records)
    return {grade: counts[grade] for grade in GRADES}


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aptos-labels", type=Path, required=True)
    parser.add_argument("--aptos-images", type=Path, required=True)
    parser.add_argument("--idrid-testing-labels", type=Path, required=True)
    parser.add_argument("--idrid-testing-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory under this worktree")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        parser.error("output must be a new directory under the pilot worktree")
    for path in (args.aptos_labels, args.idrid_testing_labels):
        if not path.is_file():
            parser.error("missing label CSV: {}".format(path))
    for path in (args.aptos_images, args.idrid_testing_images):
        if not path.is_dir():
            parser.error("missing image directory: {}".format(path))

    excluded = []
    aptos = audit("APTOS", args.aptos_labels, args.aptos_images,
                  "id_code", "diagnosis", excluded)
    idrid = audit("IDRiD", args.idrid_testing_labels, args.idrid_testing_images,
                  "Image name", "Retinopathy grade", excluded)
    input_counts = {"APTOS": class_counts(aptos), "IDRiD_testing": class_counts(idrid)}
    problems = []
    try:
        reference, internal = split_aptos(aptos)
    except ValueError as error:
        problems.append(str(error))
        reference, internal = [], []

    reference_by_hash = {row["pixel_sha256"]: row["image_id"] for row in reference}
    external = []
    for row in idrid:
        overlap = reference_by_hash.get(row["pixel_sha256"])
        if overlap is None:
            external.append(row)
        else:
            excluded.append(("IDRiD", row["image_id"], "matches_aptos_reference", overlap))

    manifests = {"aptos_reference": reference, "aptos_test": internal, "idrid_test": external}
    output.mkdir(parents=True)
    write_csv(output / "exclusions.csv", ("dataset", "image_id", "reason", "detail"), excluded)
    metadata = {}
    columns = ("image_id", "relative_path", "grade", "pixel_sha256")
    for name, rows in manifests.items():
        rows.sort(key=lambda row: row["image_id"])
        counts = class_counts(rows)
        missing = [grade for grade, count in counts.items() if not count]
        if missing:
            problems.append("{} missing grades {}".format(name, missing))
        metadata[name] = {"count": len(rows), "by_grade": counts, "examples": rows[:3]}

    if not problems:
        for name, rows in manifests.items():
            path = output / (name + ".csv")
            write_csv(path, columns, ([row[key] for key in columns] for row in rows))
            metadata[name]["sha256"] = sha256(path)
    report = {
        "status": "blocked" if problems else "ready_for_review",
        "problems": problems,
        "paths_relative_to": "repository root",
        "sources": {
            "aptos": {"split": "official labeled train",
                      "url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/data",
                      "labels": str(args.aptos_labels.resolve()),
                      "labels_sha256": sha256(args.aptos_labels),
                      "images": str(args.aptos_images.resolve())},
            "idrid": {"split": "official Disease Grading testing",
                      "url": "https://idrid.grand-challenge.org/Data/",
                      "labels": str(args.idrid_testing_labels.resolve()),
                      "labels_sha256": sha256(args.idrid_testing_labels),
                      "images": str(args.idrid_testing_images.resolve())},
        },
        "seed": SEED,
        "valid_counts_before_split_and_cross_dataset_exclusions": input_counts,
        "aptos_actual_test_fraction": (len(internal) / len(aptos) if aptos else None),
        "manifests": metadata,
        "exclusion_counts": dict(Counter(row[2] for row in excluded)),
        "limitations": [
            "Official CSV columns provide image identifiers. Patient-level isolation "
            "requires separate verified patient identifiers.",
            "Duplicate detection compares exact decoded RGB pixels with EXIF orientation applied.",
            "Overlap with the model's pretraining images was not verified.",
        ],
        "evaluation": {"k": 5, "similarity": "cosine", "voting": "uniform",
                       "vote_tie": "closest neighbor among tied grades",
                       "similarity_tie": "ascending image ID",
                       "metrics": ["macro_f1_0_to_4", "accuracy", "per_grade_recall",
                                   "per_grade_support", "confusion_matrix_true_rows"]},
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("{}: {}".format(report["status"], output / "audit.json"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
