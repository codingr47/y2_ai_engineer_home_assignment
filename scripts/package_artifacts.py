"""Package every model and runtime artifact into one zip, with repo-relative paths, to ship with the solution.

    python -m scripts.package_artifacts                         # -> dist/yad2_artifacts_<commit>.zip
    python -m scripts.package_artifacts --verify dist/yad2_artifacts_<commit>.zip     # check a zip
    python -m scripts.package_artifacts --verify-tree           # check an unpacked checkout against MANIFEST.json

What goes in (everything the API loads; unzip at the repository root and each file lands where the settings expect it):
    models/fine-tuned-tagger/        the 3 ensemble members + encoder config/tokenizer   (git-ignored, ~1.1 GB)
    models/multilingual-e5-small/    Frozen Fusion's E5 encoder (second opinion)          (git-ignored, ~0.5 GB)
    src/artifacts/                   classifier, Frozen Fusion gate, tagger artifacts + gate
    src/extraction/artifacts/        extractor lexicon, taxonomy, span tagger

MANIFEST.json (inside the zip) lists every file with its size and SHA-256, plus the git commit it was built from.
Bytes are stored exactly: both confidence gates hash these files and refuse to load anything altered. Weights are
stored uncompressed, since they don't compress; text is deflated.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("models/fine-tuned-tagger", "models/multilingual-e5-small", "src/artifacts", "src/extraction/artifacts")
SKIP_PARTS = {"__pycache__", ".cache", ".locks"}
BINARY = {".pt", ".safetensors", ".bin", ".joblib", ".model", ".onnx"}
REQUIRED = ("models/fine-tuned-tagger/member_s42.pt", "models/fine-tuned-tagger/member_s13.pt",
            "models/fine-tuned-tagger/member_s1337.pt", "models/fine-tuned-tagger/encoder/config.json",
            "models/multilingual-e5-small/model.safetensors", "src/artifacts/classifier.joblib",
            "src/artifacts/confidence.joblib", "src/artifacts/tagger/gate.joblib",
            "src/artifacts/tagger/members.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files() -> list[Path]:
    out = []
    for top in INCLUDE:
        base = ROOT / top
        if not base.is_dir():
            raise SystemExit(f"missing {top}: provision it first (docs/tagger-parser.md, docs/optional-parser.md)")
        out += [p for p in sorted(base.rglob("*")) if p.is_file() and not SKIP_PARTS & set(p.relative_to(ROOT).parts)]
    missing = [r for r in REQUIRED if not (ROOT / r).is_file()]
    if missing:
        raise SystemExit(f"required artifacts missing: {missing}")
    return out


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_members(manifest: dict):
    """The tagger members in the package must be exactly the ones members.json (and so the gate) was built for."""
    spec = json.loads((ROOT / "src/artifacts/tagger/members.json").read_text(encoding="utf-8"))
    by_path = {f["path"]: f["sha256"] for f in manifest["files"]}
    for m in spec["members"]:
        path = f"models/fine-tuned-tagger/{m['file']}"
        if by_path.get(path) != m["sha256"]:
            raise SystemExit(f"{path} does not match src/artifacts/tagger/members.json")


def pack(out_dir: Path) -> Path:
    paths = files()
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit(),
                "unpack": "unzip at the repository root; then: python -m scripts.package_artifacts --verify-tree",
                "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in paths]}
    check_members(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"yad2_artifacts_{manifest['commit']}.zip"
    with zipfile.ZipFile(target, "w", allowZip64=True) as z:
        for p, entry in zip(paths, manifest["files"]):
            kind = zipfile.ZIP_STORED if p.suffix in BINARY else zipfile.ZIP_DEFLATED
            z.write(p, entry["path"], compress_type=kind)
        z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    total = sum(f["bytes"] for f in manifest["files"])
    print(f"{target} | {len(paths)} files, {total / 1e9:.2f} GB of artifacts | commit {manifest['commit']}")
    return target


def verify_zip(path: Path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("MANIFEST.json"))
        bad = []
        for f in manifest["files"]:
            h = hashlib.sha256()
            with z.open(f["path"]) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != f["sha256"]:
                bad.append(f["path"])
    print(f"{path}: {len(manifest['files'])} files, " + ("all SHA-256 match" if not bad else f"MISMATCH: {bad}"))
    return not bad


def verify_tree(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = [f["path"] for f in manifest["files"] if not (ROOT / f["path"]).is_file() or sha256(ROOT / f["path"]) != f["sha256"]]
    print(f"{len(manifest['files'])} files checked against {manifest_path.name}: " +
          ("all present and matching" if not bad else f"MISSING OR CHANGED: {bad}"))
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--verify", type=Path, help="verify a packaged zip against its MANIFEST.json")
    ap.add_argument("--verify-tree", nargs="?", const=ROOT / "MANIFEST.json", type=Path,
                    help="verify an unpacked checkout against MANIFEST.json (default: ./MANIFEST.json)")
    a = ap.parse_args()
    if a.verify:
        raise SystemExit(0 if verify_zip(a.verify) else 1)
    if a.verify_tree:
        raise SystemExit(0 if verify_tree(a.verify_tree) else 1)
    pack(a.out)


if __name__ == "__main__":
    main()
