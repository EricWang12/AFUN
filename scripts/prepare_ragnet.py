"""Link the HANDAL images and part masks that RAGNet's data.zip does not include."""
import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="RAGNet/data/data directory containing the PKL annotations.")
    parser.add_argument("--handal-root", type=Path, required=True,
                        help="HOVA-500K/HANDAL/images, or an official HANDAL root with handal_dataset_* folders.")
    args = parser.parse_args()
    annotations = sorted(args.root.glob("handal*.pkl"))
    if not annotations:
        raise FileNotFoundError(f"No HANDAL annotations in {args.root}")
    # Both HANDAL sources use the layout RAGNet expects under HANDAL/without_depth.
    link = args.root / "HANDAL/without_depth"
    if not link.exists():
        link.unlink(missing_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(args.handal_root.resolve(), target_is_directory=True)
    paths = set()
    for annotation in annotations:
        with annotation.open("rb") as handle:
            records = pickle.load(handle)
        if isinstance(records, list):
            paths.update(path for record in records for path in (record["frame_path"], record["mask_path"]))
        else:
            paths.update(path for key in ("images", "labels") for group in records[key].values() for path in group)
    # Annotation paths start with the original repository's data/ folder.
    missing = [path for path in sorted(paths) if not (args.root / Path(*Path(path).parts[1:])).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} HANDAL files are missing, e.g. {missing[0]}")
    print(f"Verified {len(paths)} HANDAL images and masks under {link}.")


if __name__ == "__main__":
    main()
