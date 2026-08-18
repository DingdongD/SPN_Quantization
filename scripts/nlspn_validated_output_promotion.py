"""Recoverably promote a validated sibling output directory."""

import os
from pathlib import Path


def promote_validated_output(staging, target, backup, validator):
    staging = Path(staging).resolve()
    target = Path(target).resolve()
    backup = Path(backup).resolve()
    if len({staging, target, backup}) != 3:
        raise ValueError("promotion paths must be distinct")
    if not (staging.parent == target.parent == backup.parent):
        raise ValueError("promotion paths must be siblings")
    if not staging.is_dir() or not target.is_dir():
        raise ValueError("staging and target must be existing directories")
    if backup.exists():
        raise FileExistsError("backup path already exists: {}".format(backup))
    if not callable(validator):
        raise TypeError("promotion validator must be callable")

    validator(staging)
    os.replace(str(target), str(backup))
    promoted = False
    try:
        os.replace(str(staging), str(target))
        promoted = True
        validator(target)
    except Exception:
        if promoted and target.exists() and not staging.exists():
            os.replace(str(target), str(staging))
        if backup.exists() and not target.exists():
            os.replace(str(backup), str(target))
        raise
    return {
        "target": str(target),
        "backup": str(backup),
    }


def promote_new_validated_output(staging, target, validator):
    """Atomically publish a new validated sibling output directory.

    Existing targets are never replaced.  If validation after the move fails,
    the output is moved back to staging so it remains available for diagnosis.
    """
    staging = Path(staging).resolve()
    target = Path(target).resolve()

    if staging == target or staging.parent != target.parent:
        raise ValueError("staging and target must be distinct siblings")
    if not staging.is_dir():
        raise ValueError("staging must be an existing directory")
    if target.exists():
        raise FileExistsError("target path already exists: {}".format(target))
    if not callable(validator):
        raise TypeError("promotion validator must be callable")

    validator(staging)
    os.replace(str(staging), str(target))
    try:
        validator(target)
    except Exception:
        if target.exists() and not staging.exists():
            os.replace(str(target), str(staging))
        raise

    return {"target": str(target)}
