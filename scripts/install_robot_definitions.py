#!/usr/bin/env python
"""Expose REALM's robot definitions to OmniGibson 3.9.1.

Since the Isaac Sim 5.1 migration, OmniGibson has no robot classes: a robot is a `RobotDefinition`
YAML that `omnigibson.robots.__init__` discovers by globbing `<gm.DATA_PATH>/*/models/*/*.yaml`
(keeping only files whose stem equals their parent directory name), and `Robot.__init__` then loads
by absolute path from `<gm.DATA_PATH>/omnigibson-robot-assets/models/<model>/<model>.yaml`.

Because that load path is hardcoded to the `omnigibson-robot-assets` dataset, REALM's definitions
cannot simply live in the repo -- they have to appear under that dataset directory. This script
symlinks them there so they stay version-controlled in REALM while OmniGibson can still find them.

Run inside the container after mounting the dataset:

    python /app/scripts/install_robot_definitions.py
    python /app/scripts/install_robot_definitions.py --copy    # copy instead of symlink
"""

import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFINITIONS_DIR = os.path.join(REPO_ROOT, "realm", "robots", "definitions")
DATASET_NAME = "omnigibson-robot-assets"
CONTAINER_ROOT = "/app/"


def rewrite_container_paths(directory):
    """Re-anchor the container-absolute paths in the copied definitions to this checkout.

    The definitions point at their USD with an absolute `/app/realm/robots/...`, which is
    deliberate: OmniGibson joins `usd_path` onto the robot-assets root, and an absolute second
    argument wins, so the asset comes straight from the repository rather than the dataset.
    That trick assumes the repository is bound at `/app`, as REALM's own Dockerfile does.

    Run outside the container and the assumption fails silently in the worst way on Windows,
    where `os.path.join("D:\\\\data\\\\...", "/app/realm/...")` yields `D:/app/realm/...` -- a
    path that does not exist, reported far from its cause. Rewriting the copies keeps the
    versioned originals untouched.
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(root, filename)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            if CONTAINER_ROOT not in text:
                continue
            # Forward slashes throughout: OmniGibson and USD both accept them on Windows, and a
            # backslash would need escaping inside the YAML string.
            rewritten = text.replace(CONTAINER_ROOT, REPO_ROOT.replace(os.sep, "/").rstrip("/") + "/")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(rewritten)
            print(f"  rewrote container paths in {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copy", action="store_true", help="Copy the definitions instead of symlinking them.")
    parser.add_argument("--data-path", default=os.environ.get("OMNIGIBSON_DATA_PATH", "/data"),
                        help="OmniGibson data path (default: $OMNIGIBSON_DATA_PATH or /data).")
    args = parser.parse_args()

    if not args.copy and REPO_ROOT != CONTAINER_ROOT.rstrip("/"):
        # Symlinking installs the definitions verbatim, so their `/app/...` usd_path stays as
        # written and resolves to nothing outside the container. On Windows os.symlink also needs
        # administrator rights or Developer Mode, which is a second reason to prefer --copy there.
        print(f"warning: this checkout is at {REPO_ROOT}, not /app -- symlinked definitions will "
              f"keep their container-absolute usd_path and fail to load.\n"
              f"         Use --copy to install rewritten copies instead.", file=sys.stderr)

    models_dir = os.path.join(args.data_path, DATASET_NAME, "models")
    if not os.path.isdir(models_dir):
        sys.exit(f"error: {models_dir} does not exist -- is the BEHAVIOR-1K 3.9.1 dataset mounted at {args.data_path}?")

    names = sorted(n for n in os.listdir(DEFINITIONS_DIR) if os.path.isdir(os.path.join(DEFINITIONS_DIR, n)))
    if not names:
        sys.exit(f"error: no robot definitions found in {DEFINITIONS_DIR}")

    for name in names:
        src = os.path.join(DEFINITIONS_DIR, name)
        dst = os.path.join(models_dir, name)
        definition = os.path.join(src, f"{name}.yaml")
        if not os.path.isfile(definition):
            sys.exit(f"error: {src} has no {name}.yaml -- the stem must match the directory name to be discovered")

        # Only ever replace links/dirs we installed ourselves; never clobber a stock robot.
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.exists(dst):
            sys.exit(f"error: {dst} already exists and is not a symlink -- refusing to overwrite it")

        if args.copy:
            shutil.copytree(src, dst)
            print(f"installed {name}\t{dst} (copy of {src})")
            if REPO_ROOT != CONTAINER_ROOT.rstrip("/"):
                rewrite_container_paths(dst)
        else:
            os.symlink(src, dst)
            print(f"installed {name}\t{dst} -> {src}")

    print(f"\n{len(names)} definition(s) installed. Verify with:\n"
          f"  python -c 'from omnigibson.robots import REGISTERED_ROBOTS; print(REGISTERED_ROBOTS)'")


if __name__ == "__main__":
    main()
