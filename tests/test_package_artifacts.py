"""The artifact package verifies its contents against the manifest it carries."""
import hashlib
import json
import zipfile

from scripts.package_artifacts import verify_zip


def make(path, files, manifest_files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("MANIFEST.json", json.dumps({"files": manifest_files}))


def entry(name, data):
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_verify_accepts_intact_and_rejects_altered_packages(tmp_path):
    data = b"weights"
    make(tmp_path / "ok.zip", {"models/x/member_s42.pt": data}, [entry("models/x/member_s42.pt", data)])
    assert verify_zip(tmp_path / "ok.zip")
    make(tmp_path / "bad.zip", {"models/x/member_s42.pt": b"tampered"}, [entry("models/x/member_s42.pt", data)])
    assert not verify_zip(tmp_path / "bad.zip")
