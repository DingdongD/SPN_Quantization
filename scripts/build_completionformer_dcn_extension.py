#!/usr/bin/env python3
"""Build the official CompletionFormer modulated DCN for current PyTorch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


VISION_SOURCE = """\
#include "modulated_deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward,
        "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward,
        "modulated_deform_conv_backward");
}
"""


SETUP_SOURCE = """\
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="DCN",
    version="1.0",
    ext_modules=[CUDAExtension(
        "DCN",
        sources=[
            "source/vision.cpp",
            "source/cuda/modulated_deform_conv_cuda.cu",
        ],
        include_dirs=[str(ROOT / "source")],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
"""


def replace_exact(text: str, old: str, new: str,
                  expected_count: int) -> str:
    count = text.count(old)
    if count != int(expected_count):
        raise RuntimeError(
            "expected %d occurrences of %r, found %d" %
            (int(expected_count), old, count))
    return text.replace(old, new)


def transform_modulated_cuda(text: str) -> str:
    for role in ("input", "weight", "bias", "offset", "mask"):
        text = replace_exact(
            text,
            "%s.type().is_cuda()" % role,
            "%s.is_cuda()" % role,
            expected_count=2)
    return replace_exact(
        text,
        "AT_DISPATCH_FLOATING_TYPES(input.type(),",
        "AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),",
        expected_count=2)


def transform_modulated_header(text: str) -> str:
    return replace_exact(
        text, "input.type().is_cuda()", "input.is_cuda()",
        expected_count=2)


def prepare_source(completionformer_root: Path, out_dir: Path) -> None:
    official = completionformer_root / "src" / "model" / "deformconv" / "src"
    required = (
        official / "cuda" / "modulated_deform_conv_cuda.cu",
        official / "cuda" / "modulated_deform_im2col_cuda.cuh",
        official / "cpu" / "modulated_deform_conv_cpu.h",
        official / "modulated_deform_conv.h",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing official DCN sources: %s" % missing)
    if out_dir.exists():
        raise FileExistsError(
            "DCN output directory already exists: %s" % out_dir)

    source = out_dir / "source"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(official / "cuda", source / "cuda")
    shutil.copytree(official / "cpu", source / "cpu")
    shutil.copy2(official / "modulated_deform_conv.h", source)

    cuda_path = source / "cuda" / "modulated_deform_conv_cuda.cu"
    cuda_path.write_text(
        transform_modulated_cuda(cuda_path.read_text(encoding="utf-8")),
        encoding="utf-8")
    header_path = source / "modulated_deform_conv.h"
    header_path.write_text(
        transform_modulated_header(header_path.read_text(encoding="utf-8")),
        encoding="utf-8")
    (source / "vision.cpp").write_text(
        textwrap.dedent(VISION_SOURCE), encoding="utf-8")
    (out_dir / "setup.py").write_text(
        textwrap.dedent(SETUP_SOURCE), encoding="utf-8")
    (out_dir / "build_temp").mkdir()
    (out_dir / "lib").mkdir()


def build_extension(out_dir: Path, cuda_arch: str, jobs: int) -> Path:
    jobs = int(jobs)
    if jobs <= 0:
        raise ValueError("build jobs must be positive")
    environment = dict(os.environ)
    environment["MAX_JOBS"] = str(jobs)
    environment["TORCH_CUDA_ARCH_LIST"] = str(cuda_arch)
    subprocess.run([
        sys.executable,
        "setup.py",
        "build_ext",
        "--build-temp", str(out_dir / "build_temp"),
        "--build-lib", str(out_dir / "lib"),
    ], cwd=out_dir, env=environment, check=True)
    libraries = list((out_dir / "lib").glob("DCN*.so"))
    if len(libraries) != 1:
        raise RuntimeError(
            "expected one DCN library, found %d" % len(libraries))
    return libraries[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completionformer-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--jobs", required=True, type=int)
    args = parser.parse_args()

    completionformer_root = Path(args.completionformer_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    prepare_source(completionformer_root, out_dir)
    library = build_extension(out_dir, args.cuda_arch, args.jobs)
    print("DCN_LIBRARY=%s" % library, flush=True)
    print("export PYTHONPATH=%s:$PYTHONPATH" % library.parent, flush=True)


if __name__ == "__main__":
    main()
