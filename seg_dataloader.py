from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
Pairs = List[Tuple[str, str]]  # [(img_path, mask_path), ...]


# ──────────────────────────────────────────────────────────────
# Reproducibility (giữ nguyên style từ code classification gốc)
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
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


# ──────────────────────────────────────────────────────────────
# Scan folder images/ + masks/ (cùng filename, khác/giống ext đều ok)
# ──────────────────────────────────────────────────────────────

def scan_kvasir(root: str | Path) -> Pairs:
    root = Path(root)
    img_dir = root / "images"
    mask_dir = root / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Cần đúng cấu trúc '{root}/images/' và '{root}/masks/'. "
            f"Kiểm tra lại đường dẫn sau khi giải nén Kvasir-SEG."
        )

    mask_lookup = {f.stem: f for f in mask_dir.iterdir() if f.suffix in IMAGE_EXTS}

    pairs: Pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix not in IMAGE_EXTS:
            continue
        mask_path = mask_lookup.get(img_path.stem)
        if mask_path is None:
            continue  # bỏ qua ảnh thiếu mask thay vì crash
        pairs.append((str(img_path), str(mask_path)))

    if not pairs:
        raise RuntimeError(f"Không match được cặp ảnh/mask nào trong: {root}")
    return pairs


# ──────────────────────────────────────────────────────────────
# Split 880 / 120 (đúng convention paper gốc Kvasir-SEG)
# ──────────────────────────────────────────────────────────────

def split_kvasir(
    pairs: Pairs,
    n_test: int = 120,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[Pairs, Pairs, Pairs]:
    """
    Mặc định: 880 train / 0 val / 120 test (convention gốc của paper Kvasir-SEG).
    Nếu muốn tách thêm val từ 880 train (vd val_ratio=0.1 -> ~792 train/88 val/120 test),
    set val_ratio > 0.
    """
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    if n_test >= len(shuffled):
        raise ValueError(f"n_test={n_test} >= tổng số ảnh ({len(shuffled)})")

    test_s = shuffled[:n_test]
    remain = shuffled[n_test:]

    if val_ratio > 0:
        n_val = max(1, int(len(remain) * val_ratio))
        val_s = remain[:n_val]
        train_s = remain[n_val:]
    else:
        val_s = []
        train_s = remain

    return train_s, val_s, test_s


# ──────────────────────────────────────────────────────────────
# Joint transform — QUAN TRỌNG: augment phải áp dụng ĐỒNG BỘ lên ảnh + mask
# (khác với classification transforms.Compose thông thường)
# ──────────────────────────────────────────────────────────────

class JointTransform:
    def __init__(
        self,
        img_size: int = 320,
        train: bool = True,
        hflip_p: float = 0.5,
        vflip_p: float = 0.3,
        rotate_deg: float = 15.0,
        color_jitter: bool = True,
        mask_mode: str = "binary",
    ):
        self.img_size = img_size
        self.train = train
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_deg = rotate_deg
        self.mask_mode = mask_mode
        self.color_jitter = (
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15)
            if (train and color_jitter) else None
        )
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img: Image.Image, mask: Image.Image):
        # resize: ảnh dùng BILINEAR, mask PHẢI dùng NEAREST để giữ mask nhị phân sắc nét
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if random.random() < self.hflip_p:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < self.vflip_p:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if self.rotate_deg > 0 and random.random() < 0.5:
                angle = random.uniform(-self.rotate_deg, self.rotate_deg)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=255 if self.mask_mode == "ade150" else 0,
                )
            if self.color_jitter is not None:
                img = self.color_jitter(img)  # chỉ áp lên ảnh, KHÔNG áp lên mask

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=self.mean, std=self.std)

        if self.mask_mode in ("ade150", "voc21"):
            # ADE uses labels 1..150; VOC uses labels 0..20. Both preserve 255 ignore.
            mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
            if self.mask_mode == "ade150":
                mask_t = torch.where(
                    (mask_t >= 1) & (mask_t <= 150),
                    mask_t - 1,
                    torch.full_like(mask_t, 255),
                )
            else:
                mask_t = torch.where(
                    (mask_t <= 20) | (mask_t == 255),
                    mask_t,
                    torch.full_like(mask_t, 255),
                )
        else:
            mask_t = TF.to_tensor(mask)          # (1, H, W), giá trị [0,1]
            mask_t = (mask_t > 0.5).float()      # nhị phân hóa sau resize/rotate

        return img_t, mask_t


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, pairs: Pairs, transform: JointTransform):
        self.pairs = pairs
        self.transform = transform
        self._err_count = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        for offset in range(5):
            img_path, mask_path = self.pairs[(idx + offset) % len(self.pairs)]
            try:
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                return self.transform(image, mask)
            except Exception:
                self._err_count += 1
                continue
        size = self.transform.img_size
        if self.transform.mask_mode in ("ade150", "voc21"):
            empty_mask = torch.full((size, size), 255, dtype=torch.long)
        else:
            empty_mask = torch.zeros(1, size, size)
        return torch.zeros(3, size, size), empty_mask

    def error_summary(self) -> int:
        return self._err_count


KvasirSegDataset = SegmentationDataset


# ──────────────────────────────────────────────────────────────
# Factory chính — gọi hàm này là xong
# ──────────────────────────────────────────────────────────────

def get_kvasir_dataloaders(
    root: str | Path,
    img_size: int = 320,
    batch_size: int = 8,
    n_test: int = 120,
    val_ratio: float = 0.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 42,
):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    pairs = scan_kvasir(root)
    train_s, val_s, test_s = split_kvasir(pairs, n_test=n_test, val_ratio=val_ratio, seed=seed)

    train_tf = JointTransform(img_size=img_size, train=True)
    eval_tf = JointTransform(img_size=img_size, train=False)

    train_ds = KvasirSegDataset(train_s, train_tf)
    val_ds = KvasirSegDataset(val_s, eval_tf) if val_s else None
    test_ds = KvasirSegDataset(test_s, eval_tf)

    common_kw = dict(num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(persistent_workers and num_workers > 0))
    if num_workers > 0:
        common_kw.update(worker_init_fn=seed_worker, prefetch_factor=2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **common_kw
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common_kw)
        if val_ds is not None else None
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common_kw)

    print(f"\n{'='*56}")
    print(f"  Kvasir-SEG dataset : {root}")
    print(f"  Image size         : {img_size}x{img_size}")
    print(f"  Train / Val / Test : {len(train_s)} / {len(val_s)} / {len(test_s)}")
    print(f"{'='*56}\n")

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# ADE20K/ADEChallengeData2016 loader — binary foreground mask
# ──────────────────────────────────────────────────────────────

def scan_ade(root: str | Path, split: str) -> Pairs:
    root = Path(root)
    image_dir = root / "images" / split
    annotation_dir = root / "annotations" / split
    if not image_dir.exists() or not annotation_dir.exists():
        raise FileNotFoundError(
            f"Cần đúng cấu trúc '{root}/images/{split}/' và "
            f"'{root}/annotations/{split}/'."
        )

    annotation_lookup = {
        path.stem: path
        for path in annotation_dir.iterdir()
        if path.suffix in IMAGE_EXTS
    }
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix not in IMAGE_EXTS:
            continue
        annotation_path = annotation_lookup.get(image_path.stem)
        if annotation_path is not None:
            pairs.append((str(image_path), str(annotation_path)))

    if not pairs:
        raise RuntimeError(f"Không match được ảnh/mask ADE trong: {image_dir}")
    return pairs


ADESegDataset = SegmentationDataset


def get_ade_dataloaders(
    root: str | Path = r"D:\ADEChallengeData2016\ADEChallengeData2016",
    img_size: int = 320,
    batch_size: int = 8,
    val_batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 42,
):
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_pairs = scan_ade(root, "training")
    val_pairs = scan_ade(root, "validation")
    train_transform = JointTransform(img_size=img_size, train=True, mask_mode="ade150")
    eval_transform = JointTransform(img_size=img_size, train=False, mask_mode="ade150")
    train_dataset = ADESegDataset(train_pairs, train_transform)
    val_dataset = ADESegDataset(val_pairs, eval_transform)

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        common_kwargs.update(worker_init_fn=seed_worker, prefetch_factor=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, val_batch_size),
        shuffle=False,
        **common_kwargs,
    )

    print(f"\n{'=' * 56}")
    print(f"  ADE dataset         : {root}")
    print(f"  Image size          : {img_size}x{img_size}")
    print(f"  Train / Val         : {len(train_pairs)} / {len(val_pairs)}")
    print("  Classes             : 150 (ADE labels 1..150)")
    print(f"{'=' * 56}\n")
    return train_loader, val_loader, val_loader


def scan_voc(root: str | Path, split: str) -> Pairs:
    root = Path(root)
    image_dir = root / "JPEGImages"
    mask_dir = root / "SegmentationClass"
    split_file = root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not image_dir.exists() or not mask_dir.exists() or not split_file.exists():
        raise FileNotFoundError(
            f"Cần cấu trúc Pascal VOC: '{root}/JPEGImages/', "
            f"'{root}/SegmentationClass/' và '{split_file}'."
        )

    pairs = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        image_id = line.strip().split()[0]
        if not image_id:
            continue
        image_path = image_dir / f"{image_id}.jpg"
        mask_path = mask_dir / f"{image_id}.png"
        if image_path.exists() and mask_path.exists():
            pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise RuntimeError(f"Không tìm thấy cặp ảnh/mask VOC cho split '{split}' trong {root}")
    return pairs


def get_voc_dataloaders(
    root: str | Path = r"D:\VOCdevkit\VOC2012",
    img_size: int = 320,
    batch_size: int = 8,
    val_batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 42,
):
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_pairs = scan_voc(root, "train")
    val_pairs = scan_voc(root, "val")
    train_dataset = SegmentationDataset(
        train_pairs,
        JointTransform(img_size=img_size, train=True, mask_mode="voc21"),
    )
    val_dataset = SegmentationDataset(
        val_pairs,
        JointTransform(img_size=img_size, train=False, mask_mode="voc21"),
    )
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        common_kwargs.update(worker_init_fn=seed_worker, prefetch_factor=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, val_batch_size),
        shuffle=False,
        **common_kwargs,
    )
    print(f"\n{'=' * 56}")
    print(f"  Pascal VOC 2012    : {root}")
    print(f"  Image size          : {img_size}x{img_size}")
    print(f"  Train / Val         : {len(train_pairs)} / {len(val_pairs)}")
    print("  Classes             : 21 (VOC labels 0..20)")
    print(f"{'=' * 56}\n")
    return train_loader, val_loader, val_loader


# ──────────────────────────────────────────────────────────────
# Sanity check — tự tạo data giả để test code chạy đúng, không cần dataset thật
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    n_fake = 30
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        (root / "masks").mkdir()
        for i in range(n_fake):
            img = Image.fromarray((np.random.rand(400, 400, 3) * 255).astype(np.uint8))
            mask = Image.fromarray((np.random.rand(400, 400) > 0.7).astype(np.uint8) * 255)
            img.save(root / "images" / f"img_{i}.jpg")
            mask.save(root / "masks" / f"img_{i}.jpg")

        train_loader, val_loader, test_loader = get_kvasir_dataloaders(
            root=root, img_size=224, batch_size=4, n_test=6, val_ratio=0.1, num_workers=0,
        )

        xb, yb = next(iter(train_loader))
        print("Train batch — image:", xb.shape, "mask:", yb.shape, "mask unique vals:", torch.unique(yb))
        xb, yb = next(iter(test_loader))
        print("Test batch  — image:", xb.shape, "mask:", yb.shape)
