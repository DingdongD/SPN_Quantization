#!/usr/bin/env python3
"""Check dependencies and external model paths before a migrated run."""

from __future__ import print_function

import argparse
import importlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def package_status(name):
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    external_root = Path(os.environ.get(
        "SPN_EXTERNAL_ROOT",
        str(REPO_ROOT.parent / "external_depth_completion_models")))
    completionformer_root = Path(os.environ.get(
        "COMPLETIONFORMER_ROOT", str(REPO_ROOT.parent / "CompletionFormer")))
    data_root = Path(os.environ.get("SPN_DATA_ROOT", str(REPO_ROOT)))
    report = {
        "repository": str(REPO_ROOT),
        "data_root": {"path": str(data_root), "exists": data_root.exists()},
        "external_models": {
            "DySPN": {"path": str(external_root / "DySPN"),
                       "exists": (external_root / "DySPN").exists()},
            "NLSPN": {"path": str(external_root / "NLSPN_ECCV20"),
                       "exists": (external_root / "NLSPN_ECCV20").exists()},
            "CompletionFormer": {"path": str(completionformer_root),
                                  "exists": completionformer_root.exists()},
        },
        "packages": {
            name: package_status(name)
            for name in ("torch", "torchvision", "numpy", "pandas", "h5py", "PIL")
        },
        "models": ["cspn", "dyspn", "nlspn", "completionformer"],
    }
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("repository: %s" % report["repository"])
        print("data_root: %s [%s]" % (
            report["data_root"]["path"],
            "ok" if report["data_root"]["exists"] else "missing"))
        for name, item in report["external_models"].items():
            print("%-16s %s [%s]" % (
                name, item["path"], "ok" if item["exists"] else "missing"))
        missing = [name for name, item in report["packages"].items()
                   if not item["available"]]
        print("packages: %s" % ("ok" if not missing else "missing " + ", ".join(missing)))
        print("models: %s" % ", ".join(report["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
