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


def activate_explicit_cuda_device(device, family: str) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda" or requested.index is None:
        raise RuntimeError("%s requires an explicit indexed CUDA device" %
                           family)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for %s: %s" %
                           (family, requested))
    if int(requested.index) >= int(torch.cuda.device_count()):
        raise RuntimeError("%s CUDA device is unavailable: %s" %
                           (family, requested))
    torch.cuda.set_device(int(requested.index))
    current = int(torch.cuda.current_device())
    if current != int(requested.index):
        raise RuntimeError(
            "%s current CUDA device differs: cuda:%d != %s" %
            (family, current, requested))
    return requested


class NYUModelRuntime(object):
    """Loads one selected official model and normalizes its NYU interface."""

    def __init__(self, runtime_args: Namespace, saved_args: Namespace,
                 saved_meta) -> None:
        self.runtime_args = runtime_args
        self.saved_args = saved_args
        self.saved_meta = saved_meta
        self.model_name = runtime_args.model
        self.run_dir = Path(runtime_args.run_dir)
        self.checkpoint = Path(runtime_args.checkpoint)
        self.device = torch.device(runtime_args.device)
        self.expected_architecture_class = runtime_args.expected_architecture_class
        self.checkpoint_architecture = runtime_args.checkpoint_architecture
        self.required_cuda_extension = runtime_args.required_cuda_extension
        self.native_cuda_operator = runtime_args.native_cuda_operator
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
        meta_path = run_dir / "meta.json"
        saved_payload = json.loads(args_path.read_text(encoding="utf-8"))
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_model = str(saved_payload["model"])
        saved_iteration = int(saved_payload["iteration"])
        if saved_model != runtime_args.model:
            raise ValueError("checkpoint model does not match runtime model")
        if saved_iteration != int(runtime_args.propagation_iterations):
            raise ValueError("checkpoint iteration does not match propagation iterations")
        if saved_meta["architecture"] != runtime_args.checkpoint_architecture:
            raise ValueError("checkpoint architecture does not match sidecar")
        if int(saved_meta["iteration"]) != int(
                runtime_args.propagation_iterations):
            raise ValueError("checkpoint iteration does not match sidecar")
        if checkpoint.parent != run_dir:
            raise ValueError("checkpoint must belong to the configured run directory")
        if not checkpoint.is_file():
            raise FileNotFoundError("checkpoint not found: %s" % checkpoint)
        return cls(runtime_args, Namespace(**saved_payload), saved_meta)

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
        else:
            module = importlib.import_module(self.required_cuda_extension)
            if module is None:
                raise RuntimeError(
                    "required CUDA extension is unavailable: %s" %
                    self.required_cuda_extension)
        if self.native_cuda_operator is not None:
            kernel_query = getattr(
                torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
            if kernel_query is not None:
                has_cuda_kernel = kernel_query(
                    self.native_cuda_operator, "CUDA")
            else:
                dispatch_table = torch._C._dispatch_dump_table(
                    self.native_cuda_operator)
                has_cuda_kernel = any(
                    row.startswith("CUDA:")
                    for row in dispatch_table.splitlines())
            if not has_cuda_kernel:
                raise RuntimeError(
                    "required native CUDA operator is unavailable: %s" %
                    self.native_cuda_operator)

    def _assert_configured_cuda_device(self, device: torch.device) -> None:
        if device != self.device:
            raise RuntimeError("runtime requires configured CUDA device: %s" %
                               self.device)
        activate_explicit_cuda_device(
            self.device, "%s official runtime" % self.model_name)

    def _validate_checkpoint_identity(self, payload) -> None:
        checkpoint_args = payload["args"]
        checkpoint_meta = payload["meta"]
        if checkpoint_args["model"] != self.model_name:
            raise ValueError("checkpoint args model does not match runtime model")
        if checkpoint_args["model"] != self.saved_args.model:
            raise ValueError("checkpoint args model does not match sidecar")
        if int(checkpoint_args["iteration"]) != self.propagation_iterations:
            raise ValueError(
                "checkpoint args iteration does not match propagation iterations")
        if int(checkpoint_args["iteration"]) != int(self.saved_args.iteration):
            raise ValueError("checkpoint args iteration does not match sidecar")
        if checkpoint_meta["architecture"] != self.checkpoint_architecture:
            raise ValueError("checkpoint meta architecture does not match config")
        if checkpoint_meta["architecture"] != self.saved_meta["architecture"]:
            raise ValueError("checkpoint meta architecture does not match sidecar")
        if int(checkpoint_meta["iteration"]) != self.propagation_iterations:
            raise ValueError(
                "checkpoint meta iteration does not match propagation iterations")
        if int(checkpoint_meta["iteration"]) != int(self.saved_meta["iteration"]):
            raise ValueError("checkpoint meta iteration does not match sidecar")

    def _clear_official_model_modules(self) -> None:
        if self.model_name not in ("nlspn", "completionformer"):
            return
        module_names = tuple(sys.modules)
        for module_name in module_names:
            if module_name == "model" or module_name.startswith("model."):
                del sys.modules[module_name]
        if "DCN" in sys.modules:
            del sys.modules["DCN"]

    def _checkpoint_builder_args(self) -> Namespace:
        values = dict(vars(self.saved_args))
        if self.model_name == "cspn":
            values["from_scratch"] = True
        return Namespace(**values)

    def build_model(self, device: torch.device) -> nn.Module:
        self._assert_open()
        self._assert_configured_cuda_device(device)
        self._assert_required_cuda_extension()
        payload = torch.load(str(self.checkpoint), map_location=device)
        self._validate_checkpoint_identity(payload)
        self._clear_official_model_modules()
        model, _ = sweep.BUILDERS[self.model_name](
            self._checkpoint_builder_args(), device)
        if type(model).__name__ != self.expected_architecture_class:
            raise ValueError("official architecture class does not match config")
        state_dict = dict(payload["net"])
        fixed_sum_key = "post_process_layer.sum_conv.weight"
        if self.model_name == "cspn" and fixed_sum_key in state_dict:
            fixed_sum = state_dict.pop(fixed_sum_key)
            if tuple(fixed_sum.shape) != (1, 8, 1, 1, 1) or not bool(
                    torch.all(fixed_sum == 1).item()):
                raise RuntimeError("invalid CSPN fixed sum kernel")
        model.load_state_dict(state_dict, strict=True)
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
        dataset_class = sweep.CspnOfficialDataset \
            if self.model_name == "cspn" else sweep.NyuHdf5Dataset
        return dataset_class(
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
