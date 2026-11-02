"""Download public AFUN training and benchmark datasets."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPOSITORIES = {
    "visual-genome": "jn12/VisualGenome",
    "afun-train": "AFUN-dataset/AFUN", "afun-eval": "AFUN-dataset/AFUN_eval",
    "reasonaff": "hqking/affordance-R1", "ragnet": "wudongming/RAGNet", "hova500k": "JiaaZ/HOVA-500K",
}
# InstructPart is not listed: its Hugging Face mirror lacks the annotations, and Google Drive
# throttles scripted downloads. Download it from the authors' Drive folder (see the README).
FOLDERS = {"visual-genome": "visual_genome", "afun-train": "afun_train", "afun-eval": "afun_eval",
           "reasonaff": "reasonaff", "ragnet": "RAGNet", "hova500k": "HOVA-500K"}


def extract(archive, destination):
    """Extract an archive without accepting paths outside the destination."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                if not (destination / name).resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"Unsafe archive member: {name}")
            handle.extractall(destination)
    elif str(archive).endswith(".zst"):
        import zstandard
        with open(archive, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as handle:
                handle.extractall(destination, filter="data")
    else:
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")


def write_manifest(path, original, samples):
    payload = {**original, "samples": samples, "total": len(samples)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def download_afun(args, repository, revision, destination):
    def get(name, folder):
        return Path(hf_hub_download(repository, name, repo_type="dataset", revision=revision, local_dir=folder))
    if args.dataset == "afun-train":
        manifest = json.loads(get("manifest.json", destination / ".downloads").read_text())
        samples = manifest["samples"]
        if args.source:
            samples = [s for s in samples if s["dataset"] == args.source]
        if not samples:
            raise ValueError(f"Source not found: {args.source}")
        for shard in sorted({s["shard"] for s in samples}):
            extract(get(shard, destination / ".downloads"), destination)
        local = destination / "manifest.json"
        if local.exists():  # keep sources downloaded by earlier --source runs
            have = {s["path"] for s in samples}
            samples += [s for s in json.loads(local.read_text())["samples"] if s["path"] not in have]
        write_manifest(local, manifest, samples)
    else:
        subsets = [args.subset] if args.subset else ["afun_test", "scenefun3d_test", "robomind2_test"]
        for subset in subsets:
            manifest = json.loads(get(f"{subset}/manifest.json", destination / ".downloads").read_text())
            samples = manifest["samples"][:args.limit]
            for sample in samples:
                relative = Path(sample["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid sample path: {relative}")
                for name in ("obs_frame.png", "obs_frame_depth.npy", "sam_mask.png", "trajectory.json"):
                    get(f"{subset}/{relative}/{name}", destination)
            write_manifest(destination / subset / "manifest.json", manifest, samples)


def download_visual_genome(args, repository, revision, destination):
    import hashlib
    archive_dir = destination / ".downloads"
    annotation = hf_hub_download(repository, "region_descriptions.json.zip", repo_type="dataset",
                                 revision=revision, local_dir=archive_dir)
    extract(annotation, destination)
    with (destination / "region_descriptions.json").open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if checksum != "19624f10afb10bf69e30ecf48d85b1a71420c7432e8c88f1011a3c4fd60ec768":
        raise ValueError("Visual Genome annotations differ from the verified release.")
    if not args.annotations_only:
        # images.zip -> VG_100K/, images2.zip -> VG_100K_2/ (~15 GB in total)
        for filename in ("images.zip", "images2.zip"):
            extract(hf_hub_download(repository, filename, repo_type="dataset", revision=revision,
                                    local_dir=archive_dir), destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=REPOSITORIES)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--source", help="AFUN source, or one HOVA archive group (3doi, Ego4D, HANDAL, epic-100). "
                                         "HOVA defaults to the groups AFUN trains on: 3doi, Ego4D, HANDAL.")
    parser.add_argument("--subset", choices=("afun_test", "scenefun3d_test", "robomind2_test"))
    parser.add_argument("--limit", type=int, help="AFUN evaluation samples per subset for a smoke test.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--annotations-only", action="store_true", help="Download only Visual Genome captions.")
    args = parser.parse_args()
    if args.limit is not None and (args.dataset != "afun-eval" or args.limit < 1):
        parser.error("--limit is a positive number for afun-eval only")
    if args.annotations_only and args.dataset != "visual-genome":
        parser.error("--annotations-only is for visual-genome")
    if args.subset and args.dataset != "afun-eval":
        parser.error("--subset is for afun-eval")
    if args.source and args.dataset not in ("afun-train", "hova500k"):
        parser.error("--source is for afun-train or hova500k")
    if args.dataset == "hova500k" and args.source and args.source not in ("3doi", "Ego4D", "HANDAL", "epic-100"):
        parser.error("HOVA --source must be 3doi, Ego4D, HANDAL, or epic-100")
    repository = REPOSITORIES[args.dataset]
    destination = args.root / FOLDERS[args.dataset]
    destination.mkdir(parents=True, exist_ok=True)
    # Uses your Hugging Face login when present; ReasonAFF is gated and requires it.
    revision = HfApi().dataset_info(repository, revision=args.revision).sha
    if args.dataset == "visual-genome":
        download_visual_genome(args, repository, revision, destination)
    elif args.dataset.startswith("afun-"):
        download_afun(args, repository, revision, destination)
    elif args.dataset == "hova500k":
        patterns = ["annotations.tar.gz"]
        groups = [args.source] if args.source else ["3doi", "Ego4D", "HANDAL"]
        for group in groups:
            patterns.append("3doi.tar.gz" if group == "3doi" else f"{group}/part_*")
        archive_dir = destination / ".downloads"
        snapshot_download(repository, repo_type="dataset", revision=revision, local_dir=archive_dir,
                          allow_patterns=patterns)
        extract(archive_dir / "annotations.tar.gz", destination)
        for group in groups:
            archive = archive_dir / f"{group}.tar.gz"
            if group != "3doi":
                parts = sorted((archive_dir / group).glob("part_*"))
                if not parts:
                    raise FileNotFoundError(f"No archive parts for {group}")
                with archive.open("wb") as handle:
                    for part in parts:
                        with part.open("rb") as source:
                            shutil.copyfileobj(source, handle)
            extract(archive, destination)
            if group != "3doi":
                archive.unlink()  # the downloaded parts remain in .downloads
    else:
        snapshot_download(repository, repo_type="dataset", revision=revision, local_dir=destination)
        if args.dataset == "ragnet":
            extract(destination / "data.zip", destination / "data")
    (destination / "download.json").write_text(json.dumps({"repository": repository, "revision": revision,
                                                         "source": args.source, "limit": args.limit}, indent=2) + "\n")
    print(f"Downloaded {repository}@{revision} to {destination}")


if __name__ == "__main__":
    main()
