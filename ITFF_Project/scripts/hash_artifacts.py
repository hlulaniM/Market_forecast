import hashlib
import json
from pathlib import Path

ARTIFACTS = [
    "models/training/btcusdt_direction_best.keras",
    "models/training/btcusdt_direction_transformer_best.keras",
    "models/training/btcusdt_level_up_best.keras",
    "models/training/ethusdt_direction_best.keras",
    "models/training/ethusdt_direction_transformer_best.keras",
    "models/training/ethusdt_level_up_best.keras",
    "data/datasets/btcusdt_feature_scaler.pkl",
    "data/datasets/ethusdt_feature_scaler.pkl",
]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    records = []
    for rel in ARTIFACTS:
        path = Path(rel)
        if not path.exists():
            records.append({"path": rel, "sha256": None, "status": "missing"})
            continue
        digest = sha256sum(path)
        records.append({"path": rel, "sha256": digest, "status": "ok"})

    output_path = Path("deploy") / "artifact_hashes.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()

