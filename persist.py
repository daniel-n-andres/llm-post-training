"""
Persist run artifacts so a run outlives the runtime.

Colab wipes /content when the session ends. Everything expensive lives there:
the results files, the transcripts, and the trained LoRA adapter that cost GPU
minutes to produce. A snapshot is a single zip written to Google Drive (or any
directory you point at), so a fresh runtime can pick up where the last one
stopped instead of re-running the battery.

    from persist import save, snapshots, restore

    save("base", note="base battery, CoT budget 512")
    snapshots()                      # what is stored
    restore()                        # newest snapshot -> out/
    restore("_base")                 # or match part of a name

Each snapshot carries a manifest recording the model, adapter, problem-set
fingerprint, difficulties and ops of every results file inside it. That is what
makes a restored run safe to compare against a new one: `analyze.py` checks
fingerprints, and without the manifest you would be trusting a filename.
"""

import json
import os
import pathlib
import time
import zipfile

DEFAULT_SUBDIR = "sandbag-runs"

# Everything under out/ plus the source that produced it. An allowlist of
# filename patterns was the earlier design and it was wrong: a file added
# later is silently left out, and you only discover it when a restore comes
# back short. The rule is now "all of out/, and the code", with exclusions
# named explicitly so anything skipped is a decision rather than an omission.
SOURCE_PATTERNS = ["*.py", "requirements.txt"]
EXCLUDED_NAMES = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv"}

# The base model is a download, not a result — nothing that size belongs in a
# snapshot. The adapter is well under this and is the point of the exercise.
MAX_FILE_BYTES = 200 * 1024 ** 2


def _try_mount() -> bool:
    """Mount Drive, tolerating the two ways it usually refuses."""
    try:
        from google.colab import drive
    except ImportError:
        return False                      # not Colab
    for kwargs in ({}, {"force_remount": True}):
        try:
            drive.mount("/content/drive", **kwargs)
            if pathlib.Path("/content/drive/MyDrive").is_dir():
                return True
        except Exception as error:
            print(f"drive.mount{kwargs or ''} failed: {error}")
    return False


def storage_root(mount: bool = True, root: str | None = None) -> pathlib.Path:
    """Where snapshots live.

    Order: an explicit path, then SANDBAG_STORE, then an already-mounted
    Drive, then a mount attempt, then the working directory as a last resort —
    which is fine on a laptop and ephemeral on Colab, so it says so.
    """
    if root:
        chosen = pathlib.Path(root).expanduser()
    elif os.environ.get("SANDBAG_STORE"):
        chosen = pathlib.Path(os.environ["SANDBAG_STORE"]).expanduser()
    elif pathlib.Path("/content/drive/MyDrive").is_dir():
        chosen = pathlib.Path("/content/drive/MyDrive") / DEFAULT_SUBDIR
    elif mount and _try_mount():
        chosen = pathlib.Path("/content/drive/MyDrive") / DEFAULT_SUBDIR
    else:
        chosen = pathlib.Path.cwd() / DEFAULT_SUBDIR
        print(f"! Drive unavailable — using {chosen}, which does NOT survive a "
              "Colab session. Download it before the runtime recycles.")
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen


def _describe_runs(files: list[pathlib.Path]) -> dict:
    """Lift the identifying metadata out of each results file.

    A snapshot that cannot say which problems it was scored on is not evidence
    of anything three weeks later.
    """
    described = {}
    for path in files:
        if not path.name.startswith("results_"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        described[path.stem] = {key: data.get(key) for key in
                                ("model", "adapter", "n", "data",
                                 "fingerprint", "difficulties", "ops")}
    return described


def _collect(out_dir: str) -> tuple[list[tuple[pathlib.Path, str]], list[str]]:
    """Every file worth archiving, as (path on disk, name inside the zip)."""
    collected, skipped = [], []

    out = pathlib.Path(out_dir)
    if out.is_dir():
        for path in sorted(out.rglob("*")):
            if not path.is_file():
                continue
            if EXCLUDED_NAMES.intersection(path.parts):
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped.append(f"{path} ({path.stat().st_size:,} B, over cap)")
                continue
            collected.append((path, f"out/{path.relative_to(out)}"))

    for pattern in SOURCE_PATTERNS:
        for path in sorted(pathlib.Path(".").glob(pattern)):
            if path.is_file() and not EXCLUDED_NAMES.intersection(path.parts):
                collected.append((path, f"src/{path.name}"))

    return collected, skipped


def save(tag: str, out_dir: str = "out", note: str = "",
         root: str | None = None) -> pathlib.Path:
    """Zip the current artifacts and source into a timestamped snapshot."""
    out = pathlib.Path(out_dir)
    collected, skipped = _collect(out_dir)
    files = [path for path, arcname in collected if arcname.startswith("out/")]

    if not collected:
        raise SystemExit(f"nothing to save: {out.resolve()} and ./*.py are empty")

    destination = storage_root(root=root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = destination / f"{stamp}_{tag}.zip"

    manifest = {"tag": tag, "saved_at": stamp, "note": note,
                "cwd": str(pathlib.Path.cwd()),
                "runs": _describe_runs(files), "files": []}

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path, arcname in collected:
            zipped.write(path, arcname)
            manifest["files"].append(arcname)
        zipped.writestr("manifest.json", json.dumps(manifest, indent=2))

    sources = sum(1 for name in manifest["files"] if name.startswith("src/"))
    print(f"saved {archive}")
    print(f"  {len(manifest['files']) - sources} artifacts + {sources} source "
          f"files, {archive.stat().st_size:,} bytes")
    for line in skipped:
        print(f"  SKIPPED {line}")
    for name, meta in manifest["runs"].items():
        print(f"  {name}: fingerprint {meta.get('fingerprint')}  "
              f"difficulties {meta.get('difficulties')}  ops {meta.get('ops')}")
    return archive


def snapshots(root: str | None = None, mount: bool = True) -> list[pathlib.Path]:
    """List what is stored, newest last."""
    destination = storage_root(mount=mount, root=root)
    archives = sorted(destination.glob("*.zip"))
    if not archives:
        print(f"no snapshots in {destination}")
        return []
    print(f"{destination}\n")
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as zipped:
                manifest = json.loads(zipped.read("manifest.json"))
            note = f"   {manifest['note']}" if manifest.get("note") else ""
            print(f"{archive.name:<32}{len(manifest['files']):>4} files "
                  f"{archive.stat().st_size:>11,} B{note}")
            for name, meta in manifest.get("runs", {}).items():
                print(f"{'':<32}  {name}: {meta.get('fingerprint')} "
                      f"d{meta.get('difficulties')} {meta.get('ops')}")
        except Exception as error:
            print(f"{archive.name:<32}  unreadable ({error})")
    return archives


def _extract(zipped: zipfile.ZipFile, members: list[str]) -> int:
    """Unpack, mapping archive names back to where the scripts expect them.

    `src/model.py` belongs at ./model.py, not ./src/model.py — the prefix
    exists inside the zip so source and artifacts are distinguishable, not
    because the layout on disk has a src directory.
    """
    written = 0
    for name in members:
        if name.startswith("src/"):
            target = pathlib.Path(name[len("src/"):])
        elif name.startswith("out/"):
            target = pathlib.Path("out") / name[len("out/"):]
        else:
            target = pathlib.Path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zipped.read(name))
        written += 1
    return written


def restore(name: str | None = None, root: str | None = None) -> pathlib.Path:
    """Extract a snapshot back over the working directory (newest by default).

    Artifacts land in ./out and the modules land beside the notebook, which is
    the layout every script assumes. Existing files of the same name are
    overwritten — restore into a clean runtime, or know what you are replacing.
    """
    destination = storage_root(root=root)
    archives = sorted(destination.glob("*.zip"))
    if not archives:
        raise SystemExit(f"no snapshots in {destination}")

    if name is None:
        archive = archives[-1]
    else:
        matches = [a for a in archives if name in a.name]
        if not matches:
            raise SystemExit(f"no snapshot matching {name!r}. Available: "
                             + ", ".join(a.name for a in archives))
        archive = matches[-1]

    with zipfile.ZipFile(archive) as zipped:
        members = [n for n in zipped.namelist() if n != "manifest.json"]
        written = _extract(zipped, members)
        manifest = json.loads(zipped.read("manifest.json"))

    sources = sum(1 for n in members if n.startswith("src/"))
    print(f"restored {archive.name}   {written - sources} artifacts + "
          f"{sources} source files")
    if manifest.get("note"):
        print(f"  note: {manifest['note']}")
    for run_name, meta in manifest.get("runs", {}).items():
        print(f"  {run_name}: fingerprint {meta.get('fingerprint')}  "
              f"difficulties {meta.get('difficulties')}  ops {meta.get('ops')}")
    return archive
