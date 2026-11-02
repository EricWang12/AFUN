"""Check completeness of an extracted AFUN training release."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    samples = json.loads((args.root / "manifest.json").read_text())["samples"]
    if not samples:
        raise ValueError("Manifest contains no samples.")
    seen = set()
    for sample in samples:
        path = Path(sample["path"])
        if path.is_absolute() or ".." in path.parts or str(path) in seen:
            raise ValueError(f"Invalid or duplicate path: {path}")
        seen.add(str(path))
        if not (sample.get("language") or sample.get("queries")):
            raise ValueError(f"Missing instruction: {path}")
        for filename in ("obs_frame.png", "obs_frame_depth.npy", "sam_mask.png", "trajectory.json"):
            if not (args.root / path / filename).is_file():
                raise FileNotFoundError(args.root / path / filename)
    print(f"Verified {len(samples)} sample folders. No metadata conversion is needed.")


if __name__ == "__main__":
    main()
