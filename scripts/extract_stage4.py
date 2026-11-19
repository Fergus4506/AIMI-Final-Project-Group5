"""Extract frozen RETFound-Green features from the approved stage 3 manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import platform
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

from prepare_stage3 import sha256

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth"
MODEL = "vit_small_patch14_reg4_dinov2"
SPLITS = ("aptos_reference", "aptos_test", "idrid_test")
SIZE = (392, 392)
FEATURE_DIM = 384


def load_manifests(stage3):
    audit = json.loads((stage3 / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "ready_for_review" or audit["problems"]:
        raise ValueError("Stage 3 audit did not pass")
    manifests = {}
    for split in SPLITS:
        path = stage3 / (split + ".csv")
        metadata = audit["manifests"][split]
        digest = sha256(path)
        if digest != metadata["sha256"]:
            raise ValueError(f"Manifest changed after stage 3: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [row["image_id"] for row in rows]
        if len(ids) != metadata["count"] or len(set(ids)) != len(ids):
            raise ValueError(f"Unexpected number or duplication of image IDs: {path}")
        manifests[split] = (digest, rows)
    return manifests


def validate_features(np, features, count):
    if features.dtype != np.float32 or features.shape != (count, FEATURE_DIM):
        raise ValueError(f"Unexpected feature shape or type: {features.shape}, {features.dtype}")
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or not np.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError("Features contain non-finite values or zero vectors")
    return float(norms.min()), float(norms.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "retfoundgreen_statedict.pth")
    parser.add_argument("--stage3", type=Path, default=ROOT / "artifacts/stage3")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint was not found: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"Output must be a new directory: {args.output}")

    import numpy as np
    from PIL import Image, ImageOps, __version__ as pillow_version
    import timm
    import torch
    import torchvision
    from torchvision.transforms import v2

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    manifests = load_manifests(args.stage3)
    checkpoint_hash = sha256(args.checkpoint)
    print(f"Loading {args.checkpoint.name}, SHA-256 {checkpoint_hash}", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model = timm.create_model(MODEL, img_size=SIZE, num_classes=0)
    model.load_state_dict(state, strict=True)
    del state
    model.global_pool = "avg"
    model.requires_grad_(False).eval().to(args.device)
    torch.set_num_threads(2 if args.device == "cuda" else 8)
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    def preprocess(row):
        path = ROOT / row["relative_path"]
        with Image.open(path) as raw:
            return transform(ImageOps.exif_transpose(raw).convert("RGB"))

    def infer(rows, workers):
        batch = torch.stack(list(workers.map(preprocess, rows)))
        if batch.shape != (len(rows), 3, *SIZE) or batch.dtype != torch.float32:
            raise ValueError(f"Unexpected input shape or dtype: {tuple(batch.shape)}, {batch.dtype}")
        with torch.inference_mode():
            features = model(batch.to(args.device)).float().cpu().numpy()
        return features, list(batch.shape)

    samples = [(split, row) for split, (_, rows) in manifests.items() for row in rows[:2]]
    metadata = {
        "checkpoint": {"filename": args.checkpoint.name, "release": "v0.1",
                       "source": SOURCE, "sha256": checkpoint_hash},
        "model": {"architecture": MODEL, "num_classes": 0, "global_pool": "avg",
                  "mode": "eval + inference_mode"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "timm": timm.__version__,
                        "numpy": np.__version__, "pillow": pillow_version,
                        "device": args.device, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None},
        "preprocess": {"color_mode": "RGB", "exif_transpose": True, "size": SIZE,
                       "resize": "bilinear with antialias", "pixel_scale": "uint8 to float32 [0,1]",
                       "mean": [0.5] * 3, "std": [0.5] * 3},
        "batch_size": args.batch_size,
        "splits": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as workers, TemporaryDirectory(
        prefix=".stage4-", dir=args.output.parent
    ) as workspace:
        staged = Path(workspace) / "results"
        staged.mkdir()
        smoke, input_shape = infer([row for _, row in samples], workers)
        minimum, maximum = validate_features(np, smoke, len(samples))
        metadata["smoke"] = {
            "samples": [{"split": split, "image_id": row["image_id"]} for split, row in samples],
            "input_shape": input_shape, "output_shape": list(smoke.shape),
            "finite": True, "min_l2_norm": minimum, "max_l2_norm": maximum,
        }
        print(f"Smoke check passed: {input_shape} -> {list(smoke.shape)}", flush=True)

        for split, (manifest_hash, rows) in manifests.items():
            started = perf_counter()
            features = np.empty((len(rows), FEATURE_DIM), dtype=np.float32)
            for start in range(0, len(rows), args.batch_size):
                group = rows[start:start + args.batch_size]
                features[start:start + len(group)], _ = infer(group, workers)
                if (start // args.batch_size + 1) % 32 == 0:
                    print(f"{split}: {start + len(group)}/{len(rows)}", flush=True)

            ids = np.asarray([row["image_id"] for row in rows])
            path = staged / (split + ".npz")
            np.savez(path, image_ids=ids, features=features)
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["image_ids"], ids):
                    raise ValueError(f"Feature IDs do not follow manifest order: {split}")
                minimum, maximum = validate_features(np, saved["features"], len(rows))
            metadata["splits"][split] = {
                "filename": path.name, "shape": list(features.shape),
                "manifest_sha256": manifest_hash, "feature_sha256": sha256(path),
                "min_l2_norm": minimum, "max_l2_norm": maximum,
                "seconds": round(perf_counter() - started, 2),
            }
            print(f"{split}: verified {features.shape}", flush=True)

        (staged / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        args.output.mkdir()
        published = []
        try:
            for name in [metadata["splits"][split]["filename"] for split in SPLITS] + ["audit.json"]:
                destination = args.output / name
                published.append(destination)
                copyfile(staged / name, destination)
        except OSError:
            for path in published:
                path.unlink(missing_ok=True)
            args.output.rmdir()
            raise
    print(f"ready_for_review: {args.output}", flush=True)


if __name__ == "__main__":
    main()
