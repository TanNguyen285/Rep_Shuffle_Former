# RepShuffleFormer

Repo triển khai mô hình **RepShuffleFormer** cho bài toán phân loại ảnh. Mô hình sử dụng backbone MetaFormer nhẹ với `TokenMixer`, `ConvFFN`, `LayerScale`, `DropPath` và cơ chế re-parameterization Conv + BatchNorm để hỗ trợ chuyển sang chế độ deploy.

## Tính năng

- Huấn luyện phân loại ảnh bằng PyTorch.
- Tự phát hiện hai kiểu cấu trúc dataset: thư mục phẳng hoặc đã chia `train_set`/`test_set`.
- Tách validation theo từng class để hạn chế mất class trong các tập dữ liệu mất cân bằng.
- Tự phát hiện tên class từ thư mục; tên cuối cùng có dạng `<group>_<class>`.
- WeightedRandomSampler khi chênh lệch số lượng ảnh giữa các class lớn hơn `1.5x`.
- Warm-up + cosine learning-rate, AdamW, label smoothing, gradient clipping và EMA.
- Lưu checkpoint, log, biểu đồ loss/accuracy và confusion matrix cho mỗi lần chạy.

## Cấu trúc repo

```text
.
├── trainer.py          # Điểm vào huấn luyện và đánh giá
├── dataset.py          # Đọc YAML, quét dataset, split và DataLoader
├── model.py            # Model = RepShuffle backbone + classification head
├── RepShuffle.py       # Backbone và re-parameterization
├── Mixer_Token.py      # Token mixer
├── Channel_MLP.py      # ConvFFN / channel mixing
├── Head.py             # GAP + Dropout + Linear
├── visual.py           # Biểu đồ và confusion matrix
├── data.yaml           # Cấu hình đường dẫn dataset
└── runs/train/         # Kết quả sinh ra sau khi train
```

## Yêu cầu môi trường

- Python 3.10 trở lên khuyến nghị.
- PyTorch và torchvision phù hợp với CUDA nếu muốn train bằng GPU.
- Các thư viện Python:

```bash
pip install torch torchvision numpy pyyaml pillow tqdm matplotlib seaborn scikit-learn
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu không có CUDA, code tự chuyển sang CPU. Việc huấn luyện sẽ chậm hơn đáng kể.

## Chuẩn bị dataset

Chỉnh trường `path` trong [data.yaml](data.yaml). `nc` và `names` là tùy chọn; code vẫn quét class thực tế từ dataset. Ví dụ:

```yaml
path: C:/data/CCMT-Dataset
nc: 22
names: []
```

### Cấu trúc 1: chưa chia train/test

```text
dataset_root/
├── GroupA/
│   ├── healthy/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── diseased/
│       └── image_003.jpg
└── GroupB/
	└── healthy/
		└── image_004.jpg
```

Code tự tách theo từng class thành khoảng **70% train / 10% validation / 20% test**. Tên class sẽ là `GroupA_healthy`, `GroupA_diseased`, ...

### Cấu trúc 2: đã có train/test

```text
dataset_root/
├── GroupA/
│   ├── train_set/
│   │   ├── healthy/
│   │   └── diseased/
│   └── test_set/
│       ├── healthy/
│       └── diseased/
└── GroupB/
	├── train_set/
	└── test_set/
```

Code dùng nguyên `train_set` và `test_set`, sau đó tách **15% từ train thành validation**. Các định dạng ảnh được hỗ trợ gồm JPG, JPEG, PNG, BMP và WebP.

## Huấn luyện

Lệnh tối thiểu:

```bash
python trainer.py --data data.yaml
```

Ví dụ chạy scale L trong 120 epoch:

```bash
python trainer.py --data data.yaml --epochs 120 --batch-size 32 --scale l
```

Chọn scale bằng `s`, `m` hoặc `l`:

```bash
python trainer.py --data data.yaml --scale s --name baseline_s
python trainer.py --data data.yaml --scale m --name baseline_m
python trainer.py --data data.yaml --scale l --name baseline_l
```

Mặc định chương trình dùng ảnh `224x224`, batch size `32`, `120` epoch, `4` worker và tự chọn CUDA khi khả dụng. Có thể xem toàn bộ tham số bằng:

```bash
python trainer.py --help
```

### Các tham số chính

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--data` | bắt buộc | Đường dẫn tới file YAML |
| `--scale` | `m` | Kích thước model: `s`, `m`, `l` |
| `--img-size` | `224` | Kích thước ảnh đầu vào |
| `--batch-size` | `32` | Batch train; validation/test dùng gấp đôi |
| `--epochs` | `120` | Số epoch |
| `--lr` | `3e-4` | Learning rate ban đầu |
| `--min-lr` | `1e-6` | Learning rate tối thiểu |
| `--warmup-epochs` | `3` | Số epoch warm-up |
| `--weight-decay` | `5e-2` | Weight decay của AdamW |
| `--label-smoothing` | `0.05` | Label smoothing |
| `--dropout` | `0.2` | Dropout ở classification head |
| `--ema-decay` | `0.999` | EMA cho đánh giá và best checkpoint |
| `--val-loss-cap` | `None` | Không lưu best checkpoint nếu validation loss đạt/ngang ngưỡng này |
| `--num-workers` | `4` | Số worker đọc dữ liệu |
| `--project` | `runs/train` | Thư mục gốc lưu kết quả |
| `--name` | `exp` | Tên experiment |

## Kết quả đầu ra

Mỗi lần chạy tạo một thư mục mới, ví dụ `runs/train/exp_l` hoặc `runs/train/exp_l2` nếu tên đã tồn tại:

```text
runs/train/exp_l/
├── weights/
│   ├── best.pt              # State dict có validation accuracy tốt nhất
│   └── last.pt              # State dict sau epoch cuối
├── train_log.txt            # Log theo từng epoch
├── best_loss_log.txt        # Các epoch có validation loss tốt nhất
├── test_log.txt             # Tóm tắt đánh giá cuối
├── summary.txt              # Accuracy/loss và thông tin run
├── results.png              # Đường loss và validation accuracy
└── confusion_matrix.png     # Confusion matrix của best model
```

`best.pt` được chọn ưu tiên theo validation accuracy; nếu accuracy bằng nhau thì chọn validation loss thấp hơn. Sau khi train, model tốt nhất được đánh giá trên test set nếu test set tồn tại.

## Nạp checkpoint để suy luận

Checkpoint chỉ chứa `state_dict`, vì vậy cần khởi tạo model với đúng scale và số class:

```python
import torch
from PIL import Image
from torchvision import transforms

from model import Model

class_names = ["GroupA_healthy", "GroupA_diseased"]  # dùng đúng thứ tự dataset đã in ra
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model(scale="L", num_classes=len(class_names), dropout=0.2)
state_dict = torch.load("runs/train/exp_l/weights/best.pt", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device).eval()

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = transform(Image.open("image.jpg").convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
	predicted_index = model(image).argmax(dim=1).item()
print(class_names[predicted_index])
```

## Ghi chú

- Dataset được cố định seed `42` để split và huấn luyện có thể tái lập ở mức thực tế.
- Nếu `nc` trong YAML không khớp số class quét được, chương trình cảnh báo và dùng số class thực tế khi `strict_num_classes=False`.
- Tên class phải được thể hiện qua thư mục; `names` trong YAML chỉ nên dùng khi muốn chủ động chỉ định danh sách class theo đúng thứ tự.
- Trước khi train nhiều experiment, kiểm tra dung lượng đĩa vì mỗi run lưu checkpoint và ảnh biểu đồ riêng.

## Kiểm tra nhanh model

```bash
python model.py
```

Lệnh này tạo input giả `2 x 3 x 224 x 224` và kiểm tra output logits cũng như feature maps của backbone.
