#!/usr/bin/env python3
"""Train/evaluate NYU depth-completion models for propagation-iteration sweeps.

The script uses the CSPN NYU HDF5 loader so every model sees the same sparse
depth sampling and metric implementation. Run it once per model/iteration from
the conda environment that supports that model's dependencies.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path(os.environ.get(
    "SPN_EXTERNAL_ROOT",
    str(REPO_ROOT / "external")))
COMPLETIONFORMER_ROOT = Path(
    os.environ.get("COMPLETIONFORMER_ROOT",
                   str(REPO_ROOT / "external" / "CompletionFormer")))
DATA_ROOT = Path(os.environ.get("SPN_DATA_ROOT", str(REPO_ROOT)))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def add_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def resolve_data_root(args):
    """Resolve dataset paths for both new and legacy run metadata."""
    value = getattr(args, "data_root", None)
    if value is None:
        value = os.environ.get("SPN_DATA_ROOT", str(REPO_ROOT))
    return Path(value)


add_path(REPO_ROOT)
add_path(REPO_ROOT / "models")


METRIC_KEYS = [
    "MSE", "RMSE", "MAE", "ABS_REL", "DELTA1.02", "DELTA1.05",
    "DELTA1.10", "DELTA1.25", "DELTA1.25^2", "DELTA1.25^3",
]

MODEL_RECIPES = {
    "cspn": {
        "epochs": 40, "lr": 1e-2, "loss": "l1",
        "weight_decay": 1e-4, "warm_up": False,
    },
    "dyspn": {
        "epochs": 100, "lr": 5e-4, "loss": "l1l2",
        "weight_decay": 0.0, "warm_up": False,
    },
    "nlspn": {
        "epochs": 20, "lr": 1e-3, "loss": "l1l2",
        "weight_decay": 0.0, "warm_up": True,
    },
    "completionformer": {
        "epochs": 72, "lr": 1e-3, "loss": "l1l2",
        "weight_decay": 1e-2, "warm_up": True,
    },
}


def resolve_training_config(args):
    recipe = MODEL_RECIPES[args.model]
    return {
        key: recipe[key] if getattr(args, key, None) is None else getattr(args, key)
        for key in ("epochs", "lr", "loss", "weight_decay", "warm_up")
    }


def apply_training_config(args):
    for key, value in resolve_training_config(args).items():
        setattr(args, key, value)
    return args


def _dyspn_parameter_groups(model, weight_decay):
    decay = []
    norm_weight = []
    bias = []
    norm_types = tuple(
        module for name, module in torch.nn.__dict__.items()
        if isinstance(module, type) and "Norm" in name)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if name == "bias":
                bias.append(parameter)
            elif name == "weight" and isinstance(module, norm_types):
                norm_weight.append(parameter)
            else:
                decay.append(parameter)
    groups = []
    if bias:
        groups.append({"params": bias, "weight_decay": 0.0})
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if norm_weight:
        groups.append({"params": norm_weight, "weight_decay": 0.0})
    return groups


def make_optimizer_scheduler(args, model):
    if args.model == "cspn":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3,
            threshold=1e-4, min_lr=1e-6)
    elif args.model == "dyspn":
        optimizer = optim.Adam(
            _dyspn_parameter_groups(model, args.weight_decay), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.5)
    elif args.model == "nlspn":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.999), eps=1e-8)

        def lr_factor(epoch):
            if epoch < 10:
                return 1.0
            if epoch < 15:
                return 0.2
            return 0.04

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    elif args.model == "completionformer":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[36, 48, 56, 64], gamma=0.5)
    else:
        raise ValueError(args.model)
    return optimizer, scheduler


class ConvergenceTracker(object):
    def __init__(self, max_epochs, patience=8,
                 min_relative_improvement=0.001):
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_relative_improvement = float(min_relative_improvement)
        self.best_rmse = float("inf")
        self.best_epoch = 0
        self.significant_best_rmse = float("inf")
        self.no_improvement_epochs = 0
        self.lr_reduced = False
        self.history = []
        self.reason = "running"

    def update(self, epoch, rmse, current_lr, initial_lr):
        epoch = int(epoch)
        rmse = float(rmse)
        self.history.append(rmse)
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_epoch = epoch

        threshold = 1.0 - self.min_relative_improvement
        if (not math.isfinite(self.significant_best_rmse) or
                rmse < self.significant_best_rmse * threshold):
            self.significant_best_rmse = rmse
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

        reduced_now = float(current_lr) < float(initial_lr) * (1.0 - 1e-12)
        if reduced_now and not self.lr_reduced:
            self.lr_reduced = True
            self.no_improvement_epochs = 0
        if self.lr_reduced and self.no_improvement_epochs >= self.patience:
            self.reason = "plateau_after_lr_reduction"
            return True
        if epoch >= self.max_epochs:
            window = self.history[-5:]
            if len(window) >= 5:
                relative_gain = (window[0] - min(window)) / max(window[0], 1e-12)
                if relative_gain < self.min_relative_improvement:
                    self.reason = "max_epoch_plateau"
                    return True
            self.reason = "max_epoch_not_converged"
        return False

    def state_dict(self):
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_relative_improvement": self.min_relative_improvement,
            "best_rmse": self.best_rmse,
            "best_epoch": self.best_epoch,
            "significant_best_rmse": self.significant_best_rmse,
            "no_improvement_epochs": self.no_improvement_epochs,
            "lr_reduced": self.lr_reduced,
            "history": list(self.history),
            "reason": self.reason,
        }

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def torch_load_trusted(path, map_location):
    try:
        return torch.load(str(path), map_location=map_location,
                          weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def save_training_checkpoint(path, model, optimizer, scheduler, epoch,
                             tracker, val_metrics, meta, args_dict,
                             include_training_state=True):
    payload = {
        "net": model.state_dict(),
        "epoch": int(epoch),
        "tracker": tracker.state_dict(),
        "val": dict(val_metrics),
        "meta": dict(meta),
        "args": dict(args_dict),
    }
    if include_training_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    atomic_torch_save(payload, path)


def load_training_checkpoint(path, model, optimizer, scheduler, tracker,
                             device, max_epochs=None):
    checkpoint = torch_load_trusted(path, map_location=device)
    state = checkpoint["net"]
    dynamic_keys = [key for key in state if key.endswith("sum_conv.weight")]
    if dynamic_keys:
        state = dict((key, value) for key, value in state.items()
                     if key not in dynamic_keys)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.endswith("sum_conv.weight")]
        if missing or unexpected:
            raise RuntimeError("checkpoint mismatch: missing=%s unexpected=%s" %
                               (missing, unexpected))
    else:
        model.load_state_dict(state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    tracker.load_state_dict(checkpoint["tracker"])
    if max_epochs is not None and int(max_epochs) > int(tracker.max_epochs):
        tracker.max_epochs = int(max_epochs)
        if tracker.reason.startswith("max_epoch"):
            tracker.reason = "running"
    return checkpoint


def make_run_summary(model, iteration, final_epoch, tracker, final_lr):
    converged_reasons = {
        "plateau_after_lr_reduction",
        "max_epoch_plateau",
    }
    return {
        "model": model,
        "iteration": int(iteration),
        "final_epoch": int(final_epoch),
        "best_epoch": int(tracker.best_epoch),
        "best_rmse": float(tracker.best_rmse),
        "final_lr": float(final_lr),
        "reason": tracker.reason,
        "converged": tracker.reason in converged_reasons,
    }


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pil_resample(name):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def create_sparse_depth(depth, n_sample, generator=None):
    valid = torch.nonzero(depth.reshape(-1) > 0.0001,
                          as_tuple=False).reshape(-1)
    count = min(int(n_sample), int(valid.numel()))
    if count <= 0:
        return torch.zeros_like(depth)
    order = torch.randperm(valid.numel(), generator=generator)[:count]
    mask = torch.zeros(depth.numel(), dtype=depth.dtype)
    mask[valid[order]] = 1.0
    return depth * mask.reshape_as(depth)


def normalize_imagenet(rgb):
    mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


def legacy_cspn_rgb(rgb):
    normalized = normalize_imagenet(rgb)
    return (normalized * 255.0).to(torch.uint8).to(rgb.dtype).div(255.0)


class NyuHdf5Dataset(torch.utils.data.Dataset):
    """Small NYU HDF5 loader without pandas so every conda env can run it."""

    def __init__(self, csv_file, root_dir, split, n_sample=500, seed=123):
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_sample = n_sample
        self.seed = seed
        self.samples = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row["Name"])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_image, depth_image = self.load_h5(self.root_dir / self.samples[idx])
        if self.split == "train":
            rgb_image, depth_image = self.transform_train(rgb_image, depth_image)
        else:
            rgb_image, depth_image = self.transform_val(rgb_image, depth_image)
        generator = None
        if self.split != "train":
            generator = torch.Generator().manual_seed(self.seed + int(idx))
        sparse_image = create_sparse_depth(
            depth_image, self.n_sample, generator=generator)
        return {"rgbd": torch.cat((rgb_image, sparse_image), 0),
                "depth": depth_image}

    def load_h5(self, path):
        with h5py.File(str(path), "r") as f:
            rgb = f["rgb"][:].transpose(1, 2, 0)
            depth = f["depth"][:]
        return (Image.fromarray(rgb, mode="RGB"),
                Image.fromarray(depth.astype("float32"), mode="F"))

    def transform_train(self, rgb_image, depth_image):
        scale = np.random.uniform(1.0, 1.5)
        resize_to = int(240 * scale)
        degree = np.random.uniform(-5.0, 5.0)
        rgb_image = TF.resize(rgb_image, resize_to,
                              interpolation=_pil_resample("BILINEAR"))
        depth_image = TF.resize(depth_image, resize_to,
                                interpolation=_pil_resample("NEAREST"))
        rgb_image = TF.rotate(rgb_image, degree,
                              interpolation=_pil_resample("BILINEAR"))
        depth_image = TF.rotate(depth_image, degree,
                                interpolation=_pil_resample("NEAREST"))
        rgb_image = self.color_jitter(rgb_image)
        rgb_image = TF.center_crop(rgb_image, (228, 304))
        depth_image = TF.center_crop(depth_image, (228, 304))
        if np.random.uniform() < 0.5:
            rgb_image = TF.hflip(rgb_image)
            depth_image = TF.hflip(depth_image)
        rgb_tensor = TF.to_tensor(rgb_image)
        depth_tensor = self.depth_to_tensor(depth_image).div(float(scale))
        return rgb_tensor, depth_tensor

    def transform_val(self, rgb_image, depth_image):
        rgb_image = TF.resize(rgb_image, 240,
                              interpolation=_pil_resample("BILINEAR"))
        depth_image = TF.resize(depth_image, 240,
                                interpolation=_pil_resample("NEAREST"))
        rgb_image = TF.center_crop(rgb_image, (228, 304))
        depth_image = TF.center_crop(depth_image, (228, 304))
        return TF.to_tensor(rgb_image), self.depth_to_tensor(depth_image)

    @staticmethod
    def depth_to_tensor(depth_image):
        depth = np.array(depth_image, dtype=np.float32, copy=True)
        return torch.from_numpy(depth).unsqueeze(0)


class CspnOfficialDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, split, n_sample=500, seed=123):
        import nyu_dataset_loader

        self.dataset = nyu_dataset_loader.NyuDepthDataset(
            csv_file=csv_file, root_dir=root_dir, split=split,
            n_sample=n_sample, input_format="hdf5")
        self.split = split
        self.n_sample = n_sample
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        depth = sample["depth"]
        generator = None
        if self.split != "train":
            generator = torch.Generator().manual_seed(self.seed + int(idx))
        sparse = create_sparse_depth(depth, self.n_sample, generator=generator)
        rgbd = torch.cat((sample["rgbd"][:3], sparse), dim=0)
        return {
            "rgbd": rgbd,
            "depth": depth,
            "cspn_preprocessed": True,
        }

def evaluate_error(gt_depth, pred_depth):
    depth_mask = gt_depth > 0.0001
    error = dict((k, 0.0) for k in METRIC_KEYS)
    pred = pred_depth[depth_mask].clamp_min(1e-6)
    gt = gt_depth[depth_mask]
    n_valid = gt.numel()
    if n_valid == 0:
        return error
    diff = torch.abs(gt - pred)
    rel = diff / gt
    mse = torch.sum(diff.pow(2)) / n_valid
    error["MSE"] = float(mse.item())
    error["RMSE"] = math.sqrt(error["MSE"])
    error["MAE"] = float((torch.sum(diff) / n_valid).item())
    error["ABS_REL"] = float((torch.sum(rel) / n_valid).item())
    max_ratio = torch.max(gt / pred, pred / gt)
    error["DELTA1.02"] = float((max_ratio < 1.02).float().mean().item())
    error["DELTA1.05"] = float((max_ratio < 1.05).float().mean().item())
    error["DELTA1.10"] = float((max_ratio < 1.10).float().mean().item())
    error["DELTA1.25"] = float((max_ratio < 1.25).float().mean().item())
    error["DELTA1.25^2"] = float((max_ratio < 1.25 ** 2).float().mean().item())
    error["DELTA1.25^3"] = float((max_ratio < 1.25 ** 3).float().mean().item())
    return error


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(ds, max_samples):
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds
    return Subset(ds, list(range(max_samples)))


def make_loaders(args):
    dataset_class = CspnOfficialDataset if args.model == "cspn" else NyuHdf5Dataset
    split_manifest = getattr(args, "split_manifest", "")
    eval_list = args.train_list if split_manifest else args.eval_list
    trainset = dataset_class(
        csv_file=args.train_list,
        root_dir=str(resolve_data_root(args)),
        split="train",
        n_sample=args.n_sample,
        seed=args.seed,
    )
    valset = dataset_class(
        csv_file=eval_list,
        root_dir=str(resolve_data_root(args)),
        split="val",
        n_sample=args.n_sample,
        seed=args.seed,
    )
    if split_manifest:
        split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        train_indices = [int(value) for value in split["train_indices"]]
        dev_indices = [int(value) for value in split["dev_indices"]]
        if set(train_indices).intersection(dev_indices):
            raise ValueError("split manifest train/dev indices overlap")
        if any(index < 0 or index >= len(trainset)
               for index in train_indices + dev_indices):
            raise ValueError("split manifest index is out of range")
        if not getattr(args, "train_full_data", False):
            trainset = Subset(trainset, train_indices)
        valset = Subset(valset, dev_indices)
    trainset = limit_dataset(trainset, args.max_train_samples)
    valset = limit_dataset(valset, args.max_val_samples)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed)),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return trainloader, valloader


def build_cspn(args, device):
    encoder_spec_path = getattr(args, "cspn_encoder_spec", "")
    if encoder_spec_path:
        from cspn_encoder_nas import build_cspn_nas
        from spn_quant.nas.spec import DecoderSpec, EncoderSpec
        from spn_quant.nas.weights import transfer_prefix_state

        spec_payload = json.loads(
            Path(encoder_spec_path).read_text(encoding="utf-8"))
        spec = EncoderSpec.from_dict(spec_payload)
        decoder_spec_path = getattr(args, "cspn_decoder_spec", "")
        decoder_spec = None
        if decoder_spec_path:
            decoder_payload = json.loads(
                Path(decoder_spec_path).read_text(encoding="utf-8"))
            decoder_spec = DecoderSpec.from_dict(decoder_payload)
        net = build_cspn_nas(
            spec, cspn_step=args.iteration, decoder_spec=decoder_spec)
        metadata = {
            "architecture": "CSPN encoder NAS",
            "iteration": args.iteration,
            "from_scratch": args.from_scratch,
            "encoder_spec": spec.to_dict(),
            "encoder_spec_sha256": file_sha256(encoder_spec_path),
        }
        if decoder_spec is not None:
            metadata["decoder_spec"] = decoder_spec.to_dict()
            metadata["decoder_spec_sha256"] = file_sha256(decoder_spec_path)
        control_checkpoint = getattr(args, "cspn_control_checkpoint", "")
        if control_checkpoint:
            checkpoint = torch_load_trusted(control_checkpoint, map_location="cpu")
            state = checkpoint.get("net", checkpoint) \
                if isinstance(checkpoint, dict) else checkpoint
            metadata["weight_transfer"] = transfer_prefix_state(net, state)
            metadata["control_checkpoint"] = str(Path(control_checkpoint).resolve())
            metadata["control_checkpoint_sha256"] = file_sha256(control_checkpoint)
        return net.to(device), metadata

    import torch_resnet_cspn_nyu as cspn_model

    cfg = {"step": args.iteration, "kernel": 3, "norm_type": "8sum"}
    net = getattr(cspn_model, args.cspn_backbone)(pretrained=not args.from_scratch,
                                                  cspn_config=cfg)
    return net.to(device), {
        "architecture": "CSPN %s" % args.cspn_backbone,
        "iteration": args.iteration,
        "from_scratch": args.from_scratch,
    }


def resolve_run_name(args):
    return getattr(args, "run_name", "") or "%s_iter%d" % (
        args.model, args.iteration)


def build_dyspn(args, device):
    add_path(EXTERNAL_ROOT / "DySPN")
    from DySPN.base import Model

    net = Model(
        iteration=args.iteration,
        num_neighbor=args.dyspn_neighbors,
        mode="dyspn",
        res=args.dyspn_resnet,
        bm=args.dyspn_basemodel,
        stodepth=not args.from_scratch,
        norm_depth=[0.0, 10.0],
        norm_layer="bn",
    )
    return net.to(device), {
        "architecture": "DySPN %s %s neighbor=%d" % (
            args.dyspn_resnet, args.dyspn_basemodel, args.dyspn_neighbors),
        "iteration": args.iteration,
        "from_scratch": args.from_scratch,
    }


def nlspn_namespace(args):
    from argparse import Namespace

    return Namespace(
        network=args.nlspn_network,
        from_scratch=args.from_scratch,
        prop_time=args.iteration,
        prop_kernel=3,
        conf_prop=True,
        affinity="TGASS",
        affinity_gamma=0.5,
        preserve_input=False,
        legacy=False,
        lr=args.lr,
    )


def build_nlspn(args, device):
    root = EXTERNAL_ROOT / "NLSPN_ECCV20"
    add_path(root / "src")
    add_path(root / "src" / "model" / "deformconv")
    from model.nlspnmodel import NLSPNModel

    old_cwd = os.getcwd()
    try:
        os.chdir(str(root / "src"))
        ns = nlspn_namespace(args)
        net = NLSPNModel(ns)
    finally:
        os.chdir(old_cwd)
    return net.to(device), {
        "architecture": "NLSPN %s" % args.nlspn_network,
        "iteration": args.iteration,
        "from_scratch": args.from_scratch,
    }


def completionformer_namespace(args):
    from argparse import Namespace

    return Namespace(
        model=args.completionformer_model,
        from_scratch=args.from_scratch,
        prop_time=args.iteration,
        prop_kernel=3,
        conf_prop=True,
        affinity="TGASS",
        affinity_gamma=0.5,
        preserve_input=False,
        legacy=False,
        max_depth=10.0,
    )


def build_completionformer(args, device):
    root = COMPLETIONFORMER_ROOT
    add_path(root / "src")
    add_path(root / "src" / "model" / "deformconv")
    from model.completionformer import CompletionFormer

    old_cwd = os.getcwd()
    try:
        os.chdir(str(root / "src"))
        ns = completionformer_namespace(args)
        net = CompletionFormer(ns)
    finally:
        os.chdir(old_cwd)
    return net.to(device), {
        "architecture": args.completionformer_model,
        "iteration": args.iteration,
        "from_scratch": args.from_scratch,
    }


BUILDERS = {
    "cspn": build_cspn,
    "dyspn": build_dyspn,
    "nlspn": build_nlspn,
    "completionformer": build_completionformer,
}


def batch_to_model_input(model_name, sample, device):
    rgbd = sample["rgbd"].to(device, non_blocking=True)
    gt = sample["depth"].to(device, non_blocking=True)
    rgb = rgbd[:, :3, :, :]
    dep = rgbd[:, 3:4, :, :]
    if model_name == "cspn":
        preprocessed = sample.get("cspn_preprocessed", False)
        if torch.is_tensor(preprocessed):
            preprocessed = bool(preprocessed.all().item())
        cspn_rgb = rgb if preprocessed else legacy_cspn_rgb(rgb)
        cspn_input = torch.cat((cspn_rgb, dep), dim=1)
        return (cspn_input,), gt
    if model_name == "dyspn":
        return (rgb, dep), gt
    if model_name in ("nlspn", "completionformer"):
        return ({"rgb": rgb, "dep": dep},), gt
    raise ValueError(model_name)


def extract_pred(output):
    if isinstance(output, dict):
        return output["pred"]
    return output


def masked_l1(pred, gt):
    mask = gt > 0.0001
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    return torch.mean(torch.abs(pred[mask] - gt[mask]))


def masked_l1_l2(pred, gt):
    mask = gt > 0.0001
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    diff = pred[mask] - gt[mask]
    return torch.mean(torch.abs(diff)) + torch.mean(diff * diff)


def compute_loss(args, pred, gt):
    if args.loss == "l1l2":
        return masked_l1_l2(pred, gt)
    return masked_l1(pred, gt)


def validate_batch_numerics(pred, gt, loss=None):
    if not torch.isfinite(gt).all():
        raise FloatingPointError("ground truth contains non-finite values")
    valid = gt[gt > 0.0001]
    if valid.numel() == 0:
        raise RuntimeError("batch has no valid NYU depth values")
    if float(valid.max().item()) > 20.0:
        raise RuntimeError("NYU target depth is not in meters: max=%.4f" %
                           float(valid.max().item()))
    if not torch.isfinite(pred).all():
        raise FloatingPointError("prediction contains non-finite values")
    if float(pred.detach().abs().max().item()) > 1e4:
        raise FloatingPointError("prediction magnitude is inconsistent with meters")
    if loss is not None and not torch.isfinite(loss).all():
        raise FloatingPointError("loss is non-finite")


def global_gradient_norm(model):
    total = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        raise RuntimeError("model produced no gradients")
    norm = torch.sqrt(total)
    if not torch.isfinite(norm):
        raise FloatingPointError("gradient norm is non-finite")
    if float(norm.item()) <= 0.0:
        raise RuntimeError("gradient norm is zero")
    return float(norm.item())


def update_prediction_stats(stats, pred):
    detached = pred.detach()
    stats["pred_min"] = min(stats["pred_min"], float(detached.min().item()))
    stats["pred_max"] = max(stats["pred_max"], float(detached.max().item()))
    stats["pred_sum"] += float(detached.double().sum().item())
    stats["pred_count"] += int(detached.numel())


def empty_metric_sum():
    return dict((k, 0.0) for k in METRIC_KEYS)


def add_metric_weighted(metric_sum, metric, batch_size):
    for key in METRIC_KEYS:
        metric_sum[key] += float(metric.get(key, 0.0)) * batch_size


def metric_average(metric_sum, total):
    return dict((key, metric_sum[key] / max(total, 1)) for key in METRIC_KEYS)


def train_one_epoch(args, model, loader, optimizer, device, epoch):
    model.train()
    loss_sum = 0.0
    total = 0
    metric_sum = empty_metric_sum()
    stats = {"pred_min": float("inf"), "pred_max": -float("inf"),
             "pred_sum": 0.0, "pred_count": 0, "grad_norm": 0.0}
    start = time.time()
    for step, sample in enumerate(loader):
        if args.warm_up and epoch == 1:
            warm_lr = args.lr * float(step + 1) / float(max(len(loader), 1))
            for group in optimizer.param_groups:
                group["lr"] = warm_lr
        model_args, gt = batch_to_model_input(args.model, sample, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(*model_args)
        pred = extract_pred(output)
        loss = compute_loss(args, pred, gt)
        validate_batch_numerics(pred, gt, loss)
        loss.backward()
        grad_norm = global_gradient_norm(model)
        optimizer.step()

        bs = gt.size(0)
        total += bs
        loss_sum += float(loss.item()) * bs
        metric = evaluate_error(gt_depth=gt.detach(), pred_depth=pred.detach())
        add_metric_weighted(metric_sum, metric, bs)
        update_prediction_stats(stats, pred)
        stats["grad_norm"] += grad_norm * bs

        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            print("epoch=%d step=%d/%d loss=%.5f rmse=%.5f" % (
                epoch, step + 1, len(loader), loss_sum / total,
                metric_average(metric_sum, total)["RMSE"]), flush=True)
    out = metric_average(metric_sum, total)
    out["loss"] = loss_sum / max(total, 1)
    out["seconds"] = time.time() - start
    out["pred_min"] = stats["pred_min"]
    out["pred_max"] = stats["pred_max"]
    out["pred_mean"] = stats["pred_sum"] / max(stats["pred_count"], 1)
    out["grad_norm"] = stats["grad_norm"] / max(total, 1)
    out["lr"] = float(optimizer.param_groups[0]["lr"])
    return out


def evaluate(args, model, loader, device):
    model.eval()
    loss_sum = 0.0
    total = 0
    metric_sum = empty_metric_sum()
    stats = {"pred_min": float("inf"), "pred_max": -float("inf"),
             "pred_sum": 0.0, "pred_count": 0}
    start = time.time()
    with torch.no_grad():
        for sample in loader:
            model_args, gt = batch_to_model_input(args.model, sample, device)
            pred = extract_pred(model(*model_args))
            loss = compute_loss(args, pred, gt)
            validate_batch_numerics(pred, gt, loss)
            bs = gt.size(0)
            total += bs
            loss_sum += float(loss.item()) * bs
            metric = evaluate_error(gt_depth=gt.detach(), pred_depth=pred.detach())
            add_metric_weighted(metric_sum, metric, bs)
            update_prediction_stats(stats, pred)
    out = metric_average(metric_sum, total)
    out["loss"] = loss_sum / max(total, 1)
    out["seconds"] = time.time() - start
    out["pred_min"] = stats["pred_min"]
    out["pred_max"] = stats["pred_max"]
    out["pred_mean"] = stats["pred_sum"] / max(stats["pred_count"], 1)
    out["grad_norm"] = ""
    return out


def write_epoch_row(csv_path, row):
    exists = csv_path.exists()
    fieldnames = [
        "model", "iteration", "epoch", "split", "loss", "MSE", "RMSE",
        "MAE", "ABS_REL", "DELTA1.02", "DELTA1.05", "DELTA1.10",
        "DELTA1.25", "DELTA1.25^2", "DELTA1.25^3", "seconds", "lr",
        "pred_min", "pred_max", "pred_mean", "grad_norm",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(dict((k, row.get(k, "")) for k in fieldnames))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=("cspn", "dyspn", "nlspn", "completionformer"))
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--loss", default=None, choices=("l1", "l1l2"))
    parser.add_argument("--warm-up", action="store_true", dest="warm_up")
    parser.add_argument("--no-warm-up", action="store_false", dest="warm_up")
    parser.set_defaults(warm_up=None)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--train-list", default=str(REPO_ROOT / "datalist" / "nyudepth_hdf5_train.csv"))
    parser.add_argument("--eval-list", default=str(REPO_ROOT / "datalist" / "nyudepth_hdf5_val.csv"))
    parser.add_argument("--data-root", default=str(DATA_ROOT),
                        help="root used to resolve dataset paths from CSV files")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-root", default=str(REPO_ROOT / "output" / "nyu_iteration_sweep"))
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--convergence-patience", type=int, default=8)
    parser.add_argument("--min-relative-improvement", type=float, default=0.001)

    parser.add_argument("--cspn-backbone", default="resnet18",
                        choices=("resnet18", "resnet34", "resnet50"))
    parser.add_argument("--cspn-encoder-spec", default="")
    parser.add_argument("--cspn-decoder-spec", default="")
    parser.add_argument("--cspn-control-checkpoint", default="")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument(
        "--train-full-data", action="store_true",
        help="train on the full list while using manifest dev indices for monitoring")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dyspn-resnet", default="res34", choices=("res18", "res34"))
    parser.add_argument("--dyspn-basemodel", default="v1", choices=("v1", "v2"))
    parser.add_argument("--dyspn-neighbors", type=int, default=5)
    parser.add_argument("--nlspn-network", default="resnet34", choices=("resnet18", "resnet34"))
    parser.add_argument("--completionformer-model", default="CompletionFormer")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_training_config(args)
    torch.set_num_threads(max(int(args.torch_threads), 1))
    seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these model training runs")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    save_dir = Path(args.save_root) / resolve_run_name(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    csv_path = save_dir / "metrics.csv"

    trainloader, valloader = make_loaders(args)
    if len(trainloader) == 0 or len(valloader) == 0:
        raise RuntimeError("empty train or validation loader")
    print("model=%s iteration=%d train_batches=%d val_batches=%d save_dir=%s" % (
        args.model, args.iteration, len(trainloader), len(valloader), save_dir), flush=True)

    model, meta = BUILDERS[args.model](args, device)
    meta.update({
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "train_samples": len(trainloader.dataset),
        "val_samples": len(valloader.dataset),
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
    })
    if args.split_manifest:
        meta["split_manifest"] = str(Path(args.split_manifest).resolve())
        meta["split_manifest_sha256"] = file_sha256(args.split_manifest)
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    optimizer, scheduler = make_optimizer_scheduler(args, model)
    tracker = ConvergenceTracker(
        max_epochs=args.epochs,
        patience=args.convergence_patience,
        min_relative_improvement=args.min_relative_improvement)
    start_epoch = 1
    if args.resume is not None:
        resume_path = save_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError("resume checkpoint not found: %s" % resume_path)
        checkpoint = load_training_checkpoint(
            resume_path, model, optimizer, scheduler, tracker, device,
            max_epochs=args.epochs)
        start_epoch = int(checkpoint["epoch"]) + 1
        print("resumed checkpoint=%s next_epoch=%d best_rmse=%.5f" % (
            resume_path, start_epoch, tracker.best_rmse), flush=True)

    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        final_epoch = epoch
        train_metrics = train_one_epoch(args, model, trainloader, optimizer, device, epoch)
        val_metrics = evaluate(args, model, valloader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        val_metrics["lr"] = current_lr
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            row = {"model": args.model, "iteration": args.iteration,
                   "epoch": epoch, "split": split}
            row.update(metrics)
            write_epoch_row(csv_path, row)
            print("%s epoch=%d loss=%.5f rmse=%.5f mae=%.5f abs_rel=%.5f seconds=%.1f" % (
                split, epoch, metrics["loss"], metrics["RMSE"],
                metrics["MAE"], metrics["ABS_REL"], metrics["seconds"]), flush=True)

        if args.model == "cspn":
            scheduler.step(val_metrics["MAE"])
        else:
            scheduler.step()
        next_lr = float(optimizer.param_groups[0]["lr"])
        converged = tracker.update(
            epoch, val_metrics["RMSE"], next_lr, args.lr)
        save_training_checkpoint(
            save_dir / "last.pt", model, optimizer, scheduler, epoch,
            tracker, val_metrics, meta, vars(args))
        if tracker.best_epoch == epoch:
            save_training_checkpoint(
                save_dir / "best.pt", model, optimizer, scheduler, epoch,
                tracker, val_metrics, meta, vars(args),
                include_training_state=False)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_training_checkpoint(
                save_dir / ("epoch_%03d.pt" % epoch), model, optimizer,
                scheduler, epoch, tracker, val_metrics, meta, vars(args))
        if converged:
            print("converged epoch=%d reason=%s best_epoch=%d best_rmse=%.5f" % (
                epoch, tracker.reason, tracker.best_epoch, tracker.best_rmse),
                flush=True)
            break

    summary = make_run_summary(
        args.model, args.iteration, final_epoch, tracker,
        float(optimizer.param_groups[0]["lr"]))
    write_json_atomic(save_dir / "run_summary.json", summary)
    print("done model=%s iteration=%d best_rmse=%.5f reason=%s save_dir=%s" % (
        args.model, args.iteration, tracker.best_rmse, tracker.reason, save_dir),
        flush=True)


if __name__ == "__main__":
    main()
