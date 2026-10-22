"""Local entry points for frozen V14 evaluation and isolated reproduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def restore(root):
    bundle = root / "baselines/v14/reproduction_bundle.tar.gz"
    manifest = json.loads((root / "provenance/v14_extension_manifest.json").read_text())
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
        raise RuntimeError("V14 reproduction bundle hash mismatch")
    destination = root / "legacy/v14/research_result"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise RuntimeError("unsafe reproduction bundle member")
            relative = str(Path("research_result") / member.name)
            expected = manifest["archive_file_hashes"][relative]
            if resolved.exists():
                if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"existing historical file differs: {member.name}")
                continue
            archive.extract(member, destination, filter="data")
            if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"restored historical file differs: {member.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("evaluate", "training-smoke", "tests", "audit", "restore", "campaign"))
    parser.add_argument("--profile", choices=("smoke", "dev"), default="smoke")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.operation == "restore":
        restore(root)
        return
    if args.operation in ("evaluate", "audit", "campaign"):
        restore(root)
    legacy = root / "legacy/v14"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(legacy / "src")
    command = [sys.executable, str(legacy / "tools.py"), args.operation, "--root", str(root), "--profile", args.profile]
    completed = subprocess.run(command, cwd=legacy, env=environment)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
