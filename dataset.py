from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp")
Samples = List[Tuple[str, int]]  # [(img_path, class_idx), ...]


# ──────────────────────────────────────────────────────────────
# YAML config — chỉ cần path (+ nc/names tùy chọn)
# ──────────────────────────────────────────────────────────────

class DataConfig:
    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path).resolve()
        raw = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}

        root_raw = raw.get("path")
        if not root_raw:
            self.root = self.yaml_path.parent
        else:
            root = Path(root_raw)
            if root == Path(self.yaml_path.name):
                self.root = self.yaml_path.parent
            else:
                self.root = root if root.is_absolute() else (self.yaml_path.parent / root).resolve()

        if not self.root.exists():
            raise FileNotFoundError(f"[dataset] Không tìm thấy dataset root: {self.root}")

        self.nc: Optional[int] = raw.get("nc")
        self.names: Optional[List[str]] = raw.get("names") or None

    def __repr__(self):
        return f"DataConfig(root={self.root}, nc={self.nc}, names={self.names})"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _normalize_class_name(raw: str) -> str:
    """'healthy1' -> 'healthy' (CCMT hay đánh số hậu tố cho class trùng tên)."""
    return re.sub(r"\d+$", "", raw).strip()


# ──────────────────────────────────────────────────────────────
# CCMT structure — tự phát hiện 1 trong 2 kiểu:
#   flat  : root/<group>/<class>/*.jpg
#   split : root/<group>/<train_set|test_set>/<class>/*.jpg
# ──────────────────────────────────────────────────────────────

def _detect_structure(root: Path) -> str:
    group_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not group_dirs:
        raise RuntimeError(f"Không có group folder nào trong: {root}")

    first_sub = next((d for d in group_dirs[0].iterdir() if d.is_dir()), None)
    if first_sub is None:
        raise RuntimeError(f"Group '{group_dirs[0].name}' không có subfolder nào.")

    has_deeper = any(d.is_dir() for d in first_sub.iterdir())
    return "split" if has_deeper else "flat"


def _find_split_folder(root: Path, prefer: str) -> str:
    first_group = next(d for d in root.iterdir() if d.is_dir())
    candidates = [d.name for d in first_group.iterdir() if d.is_dir()]
    match = next((c for c in candidates if re.search(prefer, c, re.IGNORECASE)), None)
    return match or candidates[0]


def _scan_flat_ccmt(root: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    raw: List[Tuple[str, str]] = []
    class_names: set = set()
    for group_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for cd in sorted(d for d in group_dir.iterdir() if d.is_dir()):
            full_name = f"{group_dir.name}_{_normalize_class_name(cd.name)}"
            class_names.add(full_name)
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix in IMAGE_EXTS:
                    raw.append((str(f), full_name))
    if not raw:
        raise RuntimeError(f"Không có ảnh nào trong: {root}")
    return raw, sorted(class_names)


def _scan_ccmt_split(root: Path, split_name: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    raw: List[Tuple[str, str]] = []
    class_names: set = set()
    for group_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        split_dir = group_dir / split_name
        if not split_dir.exists():
            continue
        for cd in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            full_name = f"{group_dir.name}_{_normalize_class_name(cd.name)}"
            class_names.add(full_name)
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix in IMAGE_EXTS:
                    raw.append((str(f), full_name))
    if not raw:
        raise RuntimeError(f"Không có ảnh trong split '{split_name}' tại: {root}")
    return raw, sorted(class_names)


def _to_indexed(raw: List[Tuple[str, str]], cls_to_idx: Dict[str, int]) -> Samples:
    return [(p, cls_to_idx[c]) for p, c in raw if c in cls_to_idx]


def _stratified_split_three(
    samples: Samples, val_ratio: float = 0.15, test_ratio: float = 0.20, seed: int = 42
) -> Tuple[Samples, Samples, Samples]:
    """Tách dữ liệu thành 3 phần Train/Val/Test theo tỷ lệ phân bổ của từng class."""
    rng = random.Random(seed)
    by_class: Dict[int, Samples] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)

    train_s, val_s, test_s = [], [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n_total = len(items)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)

        val_s.extend(items[:n_val])
        test_s.extend(items[n_val : n_val + n_test])
        train_s.extend(items[n_val + n_test :])

    return train_s, val_s, test_s


def _stratified_split(samples: Samples, ratio: float, seed: int) -> Tuple[Samples, Samples]:
    """Tách `ratio` tỷ lệ mỗi class ra làm phần b, phần còn lại là a."""
    rng = random.Random(seed)
    by_class: Dict[int, Samples] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)

    a, b = [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n = max(1, int(len(items) * ratio))
        b.extend(items[:n])
        a.extend(items[n:])
    return a, b


# ──────────────────────────────────────────────────────────────
# Dataset + Transforms
# ──────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, samples: Samples, transform=None, target_size: int = 224, pre_resize: int = 0):
        self.samples = samples
        self.transform = transform
        self.target_size = target_size
        self.pre_resize = pre_resize or int(round(target_size * 1.15))
        self._err_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            try:
                img.draft("RGB", (self.pre_resize, self.pre_resize))
            except Exception:
                pass
            img = img.convert("RGB")

            w, h = img.size
            if w != h:
                scale = self.pre_resize / min(w, h)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                img = img.resize((nw, nh), Image.BILINEAR)
                left, top = (nw - self.pre_resize) // 2, (nh - self.pre_resize) // 2
                img = img.crop((left, top, left + self.pre_resize, top + self.pre_resize))
            elif w != self.pre_resize:
                img = img.resize((self.pre_resize, self.pre_resize), Image.BILINEAR)
            return img
        except Exception:
            self._err_count += 1
            return None

    def __getitem__(self, idx: int):
        for offset in range(5):
            path, label = self.samples[(idx + offset) % len(self.samples)]
            img = self._load(path)
            if img is not None:
                if self.transform:
                    img = self.transform(img)
                return img, label
        return torch.zeros(3, self.target_size, self.target_size), label

    def error_summary(self) -> int:
        return self._err_count


def build_train_transform(img_size: int) -> transforms.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(img_size: int) -> transforms.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _print_summary(dcfg: DataConfig, class_names, train_s, val_s, test_s, strategy: str) -> None:
    n_tr, n_val, n_te = len(train_s), len(val_s), len(test_s or [])
    print(f"\n{'='*62}")
    print(f"  Data yaml : {dcfg.yaml_path.name}")
    print(f"  Strategy  : {strategy}")
    print(f"  Classes   : {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"    [{i:2d}] {name}")
    print(f"  Images    : {n_tr + n_val + n_te:,} total")
    print(f"    train   : {n_tr:,}")
    print(f"    val     : {n_val:,}")
    print(f"    test    : {n_te:,}" + (" (N/A)" if n_te == 0 else ""))
    print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────
# Main factory
# ──────────────────────────────────────────────────────────────

def get_dataloaders(
    cfg,
    data_yaml: str | Path,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    strict_num_classes: bool = False,
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    handle_imbalance: bool = True,
    seed: int = 42,
):
    dcfg = DataConfig(data_yaml)
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    img_size = int(getattr(cfg, "img_size", 224))

    struct = _detect_structure(dcfg.root)
    if struct == "flat":
        strategy = f"CCMT flat (group/class) — Tách 70% Train / 10% Val / 20% Test từ toàn bộ dữ liệu"
        all_raw, all_cls = _scan_flat_ccmt(dcfg.root)
        class_names = dcfg.names or all_cls
        cls_to_idx = {c: i for i, c in enumerate(class_names)}
        all_s = _to_indexed(all_raw, cls_to_idx)
        train_s, val_s, test_s = _stratified_split_three(
            all_s, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        train_name = _find_split_folder(dcfg.root, prefer="train")
        test_name = _find_split_folder(dcfg.root, prefer="test")
        strategy = f"CCMT split ('{train_name}'/'{test_name}') — val {val_ratio:.0%} tách từ train"
        train_raw, tr_cls = _scan_ccmt_split(dcfg.root, train_name)
        test_raw, te_cls = _scan_ccmt_split(dcfg.root, test_name)
        class_names = dcfg.names or sorted(set(tr_cls) | set(te_cls))
        cls_to_idx = {c: i for i, c in enumerate(class_names)}
        train_all = _to_indexed(train_raw, cls_to_idx)
        test_s = _to_indexed(test_raw, cls_to_idx)
        train_s, val_s = _stratified_split(train_all, val_ratio, seed)

    if dcfg.nc is not None and dcfg.nc != len(class_names):
        print(f"  [⚠] yaml nc={dcfg.nc} nhưng scan ra {len(class_names)} class → dùng {len(class_names)}")
    if strict_num_classes and dcfg.nc is not None and len(class_names) != dcfg.nc:
        raise ValueError(f"[dataset] Số lớp thực tế ({len(class_names)}) không khớp nc={dcfg.nc}.")

    train_tf = build_train_transform(img_size)
    val_tf = build_val_transform(img_size)

    train_ds = ImageDataset(train_s, train_tf, target_size=img_size)
    val_ds = ImageDataset(val_s, val_tf, target_size=img_size)
    test_ds = ImageDataset(test_s, val_tf, target_size=img_size) if test_s else None

    sampler = None
    if handle_imbalance:
        labels = [s[1] for s in train_s]
        counts = np.bincount(labels, minlength=len(class_names)).astype(float)
        ratio = counts.max() / (counts.min() + 1e-8)
        if ratio > 1.5:
            weights = 1.0 / torch.tensor(counts[labels], dtype=torch.float)
            sampler = WeightedRandomSampler(weights, len(train_s), replacement=True, generator=g)
            print(f"  [sampler] imbalance={ratio:.1f}× → WeightedRandomSampler ON")

    train_kw = dict(num_workers=num_workers, pin_memory=pin_memory,
                     persistent_workers=(persistent_workers and num_workers > 0), generator=g)
    eval_workers = max(1, min(2, num_workers))
    eval_kw = dict(num_workers=eval_workers, pin_memory=pin_memory, persistent_workers=False)
    if num_workers > 0:
        train_kw.update(worker_init_fn=seed_worker, prefetch_factor=2)
    if eval_workers > 0:
        eval_kw.update(worker_init_fn=seed_worker, prefetch_factor=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               shuffle=(sampler is None), drop_last=True, **train_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **eval_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **eval_kw) if test_ds else None

    _print_summary(dcfg, class_names, train_s, val_s, test_s, strategy)
    return train_loader, val_loader, test_loader, class_names