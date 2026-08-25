#!/usr/bin/env python3
"""Official NYU model runtime shared by selected quantization methods."""

from __future__ import annotations

from argparse import Namespace
import importlib
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from spn_quant.experiment_config import ModelExperimentConfig  # noqa: E402


class NYUModelRuntime(object):
    """Loads one selected official model and normalizes its NYU interface."""

    def __init__(self, runtime_args: Namespace, saved_args: Namespace) -> None:
        self.runtime_args = runtime_args
        self.saved_args = saved_args
        self.model_name = runtime_args.model
        self.run_dir = Path(runtime_args.run_dir)
        self.checkpoint = Path(runtime_args.checkpoint)
        self.expected_architecture_class = runtime_args.expected_architecture_class
        self.required_cuda_extension = runtime_args.required_cuda_extension
        self.propagation_iterations = int(runtime_args.propagation_iterations)
        self.data_root = Path(runtime_args.data_root)
        self.closed = False

    @classmethod
    def from_config(cls, config: ModelExperimentConfig) -> "NYUModelRuntime":
        return cls.from_args(config.runtime_args())

    @classmethod
    def from_args(cls, runtime_args: Namespace) -> "NYUModelRuntime":
        run_dir = Path(runtime_args.run_dir)
        checkpoint = Path(runtime_args.checkpoint)
        args_path = run_dir / "args.json"
        saved_payload = json.loads(args_path.read_text(encoding="utf-8"))
        saved_model = str(saved_payload["model"])
        saved_iteration = int(saved_payload["iteration"])
        if saved_model != runtime_args.model:
            raise ValueError("checkpoint model does not match runtime model")
        if saved_iteration != int(runtime_args.propagation_iterations):
            raise ValueError("checkpoint iteration does not match propagation iterations")
        if checkpoint.parent != run_dir:
            raise ValueError("checkpoint must belong to the configured run directory")
        if not checkpoint.is_file():
            raise FileNotFoundError("checkpoint not found: %s" % checkpoint)
        return cls(runtime_args, Namespace(**saved_payload))

    def _assert_open(self) -> None:
        if self.closed:
            raise RuntimeError("NYU model runtime is closed")

    def _assert_required_cuda_extension(self) -> None:
        module_name, separator, attribute_name = \
            self.required_cuda_extension.rpartition(".")
        if separator:
            module = importlib.import_module(module_name)
            if not hasattr(module, attribute_name):
                raise RuntimeError(
                    "required CUDA extension is unavailable: %s" %
                    self.required_cuda_extension)
            return
        module = importlib.import_module(self.required_cuda_extension)
        if module is None:
            raise RuntimeError("required CUDA extension is unavailable: %s" %
                               self.required_cuda_extension)

    def _clear_official_model_modules(self) -> None:
        if self.model_name not in ("nlspn", "completionformer"):
            return
        module_names = tuple(sys.modules)
        for module_name in module_names:
            if module_name == "model" or module_name.startswith("model."):
                del sys.modules[module_name]
        if "DCN" in sys.modules:
            del sys.modules["DCN"]

    def build_model(self, device: torch.device) -> nn.Module:
        self._assert_open()
        self._clear_official_model_modules()
        model, _ = sweep.BUILDERS[self.model_name](self.saved_args, device)
        self._assert_required_cuda_extension()
        if type(model).__name__ != self.expected_architecture_class:
            raise ValueError("official architecture class does not match config")
        payload = torch.load(str(self.checkpoint), map_location=device)
        model.load_state_dict(payload["net"], strict=True)
        model.eval()
        return model

    def build_dataset(self, split: str):
        self._assert_open()
        if split == "train":
            csv_file = self.saved_args.train_list
        elif split == "val":
            csv_file = self.saved_args.eval_list
        else:
            raise ValueError("unsupported NYU split: %s" % split)
        return sweep.NyuHdf5Dataset(
            csv_file=csv_file,
            root_dir=str(self.data_root),
            split=split,
            n_sample=self.saved_args.n_sample,
            seed=self.saved_args.seed,
        )

    def model_input(self, sample, device: torch.device):
        self._assert_open()
        return sweep.batch_to_model_input(self.model_name, sample, device)

    def prediction(self, output) -> torch.Tensor:
        self._assert_open()
        return sweep.extract_pred(output)

    def close(self) -> None:
        self.closed = True
