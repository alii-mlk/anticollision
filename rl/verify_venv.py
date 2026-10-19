#!/usr/bin/env python3
"""Check every package in the RL venv for damaged files, and repair them.

The first install ran while the Mac's disk filled up, and VMware
stopped the VM in the middle of writing. That left files truncated, silently:
a CUDA library ("file too short"), and cloudpickle, whose __init__.py was left
at 0 bytes so the module imported but had no dumps(). That broke saving a
trained model hours later. Any other package written at the same moment may be
damaged in the same way.

pip records a SHA-256 hash and a size for every file it installs, in each
package's dist-info/RECORD. This script recomputes them for every package in
the venv and lists the ones that do not match. It also flags:

  - a package whose own METADATA is empty or unreadable. pip cannot uninstall
    such a package, so a plain reinstall fails on it. Its name and version are
    taken from the dist-info folder name instead, so it is never skipped.
  - code files that exist in a package's folder but are missing from its
    RECORD, which is what a RECORD cut short looks like.

With --fix it repairs them. Packages pip can still read are reinstalled by
pip; packages with broken metadata are removed by hand first and then
installed fresh. PyTorch always comes from its CPU index, so the CUDA build
does not come back, and stray NVIDIA libraries are removed. Finally the
compiled bytecode is cleared, since it is not hashed but was written at the
same time.

Only the venv's own packages are checked, not the system or ROS ones.

Usage (in the VM, after  source rl/env.sh):
  python rl/verify_venv.py
  python rl/verify_venv.py --fix
"""

import argparse
import base64
import hashlib
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def site_packages():
    if sys.prefix == sys.base_prefix:
        sys.exit("ERROR: not running inside the venv; run  source rl/env.sh  first")
    return Path(sysconfig.get_paths()["purelib"])


def identify(dist_info):
    """Name, version, and whether the package's own METADATA could be read."""
    stem = dist_info.name[: -len(".dist-info")]
    folder_name, _, folder_version = stem.rpartition("-")
    name = version = None
    meta = dist_info / "METADATA"
    try:
        text = meta.read_text(errors="replace") if meta.exists() else ""
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith("Name:") and name is None:
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:") and version is None:
            version = line.split(":", 1)[1].strip()
        if name and version:
            break
    readable = bool(name and version)
    return name or folder_name, version or folder_version, readable


def check(dist_info, root):
    """Return a list of problems for one installed package."""
    record = dist_info / "RECORD"
    if not record.exists():
        return ["no RECORD file"]
    if record.stat().st_size == 0:
        return ["RECORD file is empty"]

    problems = []
    listed = set()
    top_dirs = set()
    for line in record.read_text(errors="replace").splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            continue
        rel, digest, size = parts
        path = (root / rel).resolve()
        listed.add(path)
        first = rel.split("/", 1)[0]
        if "/" in rel and first != ".." and not first.endswith(".dist-info"):
            top_dirs.add(first)
        if not digest:
            continue      # RECORD itself and compiled .pyc files carry no hash
        if not path.exists():
            problems.append(f"missing {rel}")
            continue
        algo, _, expected = digest.partition("=")
        data = path.read_bytes()
        if size and len(data) != int(size):
            problems.append(f"{rel}: {len(data)} bytes, expected {size}")
            continue
        actual = base64.urlsafe_b64encode(hashlib.new(algo, data).digest()).rstrip(b"=").decode()
        if actual != expected:
            problems.append(f"{rel}: contents differ")

    # A RECORD that was itself cut short lists only some of the files, and
    # the ones missing from it would never be checked above. So any code file
    # in the package's folders that the RECORD does not mention is suspect.
    for top in sorted(top_dirs):
        folder = root / top
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if (path.is_file() and path.suffix in (".py", ".so")
                    and "__pycache__" not in path.parts
                    and path.resolve() not in listed):
                problems.append(f"{path.relative_to(root)}: not listed in RECORD "
                                f"(RECORD may be truncated)")
    return problems


def top_level_names(dist_info, name):
    """The module folders and files a package put into site-packages."""
    tops = set()
    for source in ("top_level.txt", "RECORD"):
        path = dist_info / source
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            entry = line.split(",", 1)[0].strip()
            if not entry:
                continue
            first = entry.split("/", 1)[0]
            if first in ("..", "") or first.endswith(".dist-info"):
                continue
            tops.add(first[:-3] if first.endswith(".py") else first)
    if not tops:
        tops.add(name.replace("-", "_").lower())
    return tops


def remove_by_hand(dist_info, name, root):
    """Delete a package pip cannot uninstall because its metadata is broken."""
    for top in sorted(top_level_names(dist_info, name)):
        for candidate in (root / top, root / f"{top}.py"):
            if candidate.is_dir():
                shutil.rmtree(candidate)
                print(f"  removed {candidate}")
            elif candidate.exists():
                candidate.unlink()
                print(f"  removed {candidate}")
    shutil.rmtree(dist_info)
    print(f"  removed {dist_info}")


def pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    print("  $ " + " ".join(cmd[2:]), flush=True)
    return subprocess.run(cmd).returncode == 0


def install(name, version, force):
    spec = f"{name}=={version}" if version else name
    extra = ["--force-reinstall"] if force else []
    if name.lower() == "torch":
        return pip("install", *extra, "--no-deps", "--no-cache-dir", spec,
                   "--index-url", TORCH_CPU_INDEX)
    return pip("install", *extra, "--no-deps", "--no-cache-dir", spec)


def report_cloudpickle():
    try:
        import importlib.metadata
        import os
        import cloudpickle
        has = hasattr(cloudpickle, "dumps")
        try:
            version = importlib.metadata.version("cloudpickle")
        except Exception:
            version = "unknown"
        print(f"cloudpickle {version} from {cloudpickle.__file__} "
              f"({os.path.getsize(cloudpickle.__file__)} bytes): "
              f"dumps {'present' if has else 'MISSING'}")
    except Exception as exc:
        print(f"cloudpickle does not import: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fix", action="store_true", help="repair the damaged packages")
    args = ap.parse_args()

    root = site_packages()
    print(f"checking {root}")
    damaged = []
    total = 0
    for dist_info in sorted(root.glob("*.dist-info")):
        name, version, readable = identify(dist_info)
        total += 1
        problems = check(dist_info, root)
        if not readable:
            problems.insert(0, "METADATA is empty or unreadable, so pip cannot uninstall it")
        if problems:
            damaged.append((dist_info, name, version, readable))
            print(f"  DAMAGED  {name} {version}: {len(problems)} problem(s)")
            for p in problems[:3]:
                print(f"           {p}")
            if len(problems) > 3:
                print(f"           ... and {len(problems) - 3} more")

    print(f"{total} packages checked, {len(damaged)} damaged")
    report_cloudpickle()

    if not damaged:
        return 0
    if not args.fix:
        print("\nrun again with --fix to repair them")
        return 1

    print("\nrepairing")
    ok = True
    for dist_info, name, version, readable in damaged:
        # Folder names spell packages with underscores (nvidia_cufft), METADATA
        # with hyphens (nvidia-cufft); compare in one form.
        lower = name.lower().replace("_", "-")
        if lower.startswith("nvidia-") or lower == "triton":
            if readable:
                ok &= pip("uninstall", "-y", name)
            else:
                remove_by_hand(dist_info, name, root)
        elif readable:
            ok &= install(name, version, force=True)
        else:
            remove_by_hand(dist_info, name, root)
            ok &= install(name, version, force=False)

    caches = list(root.rglob("__pycache__"))
    for c in caches:
        shutil.rmtree(c, ignore_errors=True)
    print(f"  cleared {len(caches)} __pycache__ folders (rebuilt automatically)")

    print("\nrepair finished" if ok else "\nsome repairs failed, see above")
    print("run  python rl/verify_venv.py  again to confirm everything is intact")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
