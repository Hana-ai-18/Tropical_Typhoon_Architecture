# MGTCF & Phys-Diff — Hướng dẫn chạy trên server Linux (A100)

Hướng dẫn này dành cho 2 baseline mới (**MGTCF** — multi-generator GAN,
**Phys-Diff** — latent diffusion + PIGA), chạy trên server Linux có GPU
NVIDIA A100. Nếu bạn cũng cần chạy 5 model kia (LSTM/GRU/RNN/ST-Trans/FM),
quy trình cài đặt/dataset giống hệt bên dưới — chỉ khác script train cuối.

---

## 1. Cài đặt môi trường

```bash
# Kiểm tra GPU/driver trước tiên
nvidia-smi
```

Cột `CUDA Version` trong output `nvidia-smi` là bản CUDA **cao nhất** driver
hỗ trợ (không nhất thiết là bản bạn phải cài) — dùng nó để chọn đúng dòng
`pip install torch` bên dưới.

```bash
# Tạo virtual environment (khuyến nghị, tránh xung đột package hệ thống)
python3 -m venv venv
source venv/bin/activate

# Cài numpy trước
pip install numpy>=1.23.0

# Cài torch ĐÚNG bản CUDA của server (KHÔNG dùng "pip install torch" trần trụi
# -- có thể cài nhầm bản CPU-only hoặc bản CUDA không khớp driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Nếu nvidia-smi báo CUDA Version thấp hơn 12.1, đổi cu121 -> cu118:
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
# Bảng tương thích đầy đủ: https://pytorch.org/get-started/locally/
```

**Kiểm tra GPU đã được nhận diện đúng:**

```bash
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
print('Compute capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')
"
```

Kỳ vọng: `CUDA available: True`, `Device: NVIDIA A100...`, `Compute
capability: (8, 0)`. Nếu `CUDA available: False`, quay lại bước cài torch —
95% nguyên nhân là cài sai bản (CPU-only) hoặc driver server quá cũ so với
bản CUDA vừa cài.

**Yêu cầu Python ≥ 3.9** (code dùng cú pháp built-in generic `tuple[...]`,
không chạy được trên Python 3.8 trở xuống — kiểm tra bằng `python3 --version`).

### Tối ưu hiệu năng cho A100 (khuyến nghị, không bắt buộc)

A100 (kiến trúc Ampere) hỗ trợ TF32 — chế độ tính toán nhanh hơn FP32 chuẩn
đáng kể với độ chính xác gần như không đổi cho hầu hết workload deep
learning. PyTorch ≥ 2.0 có thể tự động bật TF32 cho phép nhân ma trận, nhưng
để chắc chắn, thêm 2 dòng sau vào đầu `train_mgtcf.py`/`train_physdiff.py`
(hoặc set biến môi trường trước khi chạy):

```bash
# Cách nhanh nhất: set qua biến môi trường, không cần sửa code
export NVIDIA_TF32_OVERRIDE=1
```

hoặc trong Python (nếu bạn tự thêm vào đầu script):
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 2. Chuẩn bị dataset trên server

Upload `Dataset_TC.zip` (hoặc dataset đã chuẩn bị) lên server bằng `scp`:

```bash
# Từ máy local (Linux/macOS/WSL):
scp /path/to/Dataset_TC.zip user@your-server-ip:/home/user/

# Giải nén trên server
ssh user@your-server-ip
unzip Dataset_TC.zip -d Dataset_TC/
cd Dataset_TC && ls
```

Cấu trúc **bắt buộc** để `--dataset_root` nhận diện đúng (script tự động
tìm thư mục chứa `Data1d/` — xem `_find_tcnd_root()` trong
`Model/data/loader_training.py`):

```
Dataset_TC/
├── Data1d/
│   ├── train/     <- file .txt track sau khi prepare_dataset.py chia
│   ├── val/
│   └── test/
├── Data3d/
│   └── <year>/<storm_name>/*.npy
└── Env_data/
    └── <year>/<storm_name>/*.npy
```

Nếu dataset chưa được chia `train/val/test`, chạy `prepare_dataset.py`
trước (đã có sẵn từ trước, không thuộc phạm vi hướng dẫn này):

```bash
python fix_discontinuity_and_sync.py --root Dataset_TC --apply
python prepare_dataset.py --root Dataset_TC \
    --obs_len 8 --pred_len 12 \
    --test_min_storms 10 --test_max_storms 10 \
    --test_min_vn 4 --test_min_easy 5 --val_ratio 0.20 \
    --check_data3d_shape --check_env_content --check_with_real_code \
    --apply
```

---

## 3. Cấu trúc thư mục code trên server

```
project_root/
├── requirements.txt
├── train_mgtcf.py
├── train_physdiff.py
├── Model/
│   ├── __init__.py
│   ├── mgtcf_model.py
│   ├── physdiff_model.py
│   ├── paper_baseline_model.py      <- PaperEncoder, dùng chung
│   ├── FNO3D_encoder.py
│   ├── mamba_encoder.py
│   ├── env_net_transformer_gphsplit.py
│   └── data/
│       ├── __init__.py
│       ├── loader_training.py
│       └── trajectoriesWithMe_unet_training.py
└── Dataset_TC/                       <- dataset đã giải nén ở bước 2
    ├── Data1d/{train,val,test}/
    ├── Data3d/
    └── Env_data/
```

Nếu thiếu `__init__.py`, tạo file rỗng:

```bash
touch Model/__init__.py Model/data/__init__.py
```

---

## 4. Chạy train

### 4.1. Test nhanh (1 epoch, kiểm tra pipeline trước khi train dài)

```bash
python train_mgtcf.py \
    --dataset_root Dataset_TC \
    --output_dir runs/mgtcf_test \
    --num_epochs 1 --val_freq 1 \
    --batch_size 32 --gpu_num 0

python train_physdiff.py \
    --dataset_root Dataset_TC \
    --output_dir runs/physdiff_test \
    --num_epochs 1 --val_freq 1 \
    --batch_size 32 --gpu_num 0
```

Nếu cả 2 lệnh chạy hết 1 epoch không lỗi (in ra dòng `[VAL ep0] ADE=...`),
pipeline đã đúng — có thể train đầy đủ.

### 4.2. Tận dụng VRAM A100 — batch size lớn hơn T4

A100 có 40GB hoặc 80GB VRAM (so với 16GB của T4) — có thể tăng
`--batch_size` đáng kể so với cấu hình mặc định (90) nếu muốn tăng tốc độ
train tổng thể. Thử tăng dần và theo dõi `nvidia-smi` (cột `Memory-Usage`)
để tìm mức phù hợp:

```bash
watch -n 1 nvidia-smi   # chạy ở terminal khác trong lúc train để theo dõi VRAM
```

### 4.3. Train đầy đủ, 3 seed

```bash
# MGTCF — 3 seed
for seed in 0 1 2; do
python train_mgtcf.py \
    --dataset_root Dataset_TC \
    --output_dir   runs/mgtcf_seed${seed} \
    --n_generators 6 --best_k 6 \
    --seed ${seed} --gpu_num 0 \
    --test_at_end
done

# Phys-Diff — 3 seed
for seed in 0 1 2; do
python train_physdiff.py \
    --dataset_root Dataset_TC \
    --output_dir   runs/physdiff_seed${seed} \
    --d_model 64 --num_blocks 3 \
    --seed ${seed} --gpu_num 0 \
    --test_at_end
done
```

Chạy nền bằng `tmux` (khuyến nghị — không mất tiến trình nếu mất SSH, dễ
theo dõi log trực tiếp):

```bash
tmux new -s physdiff_train
python train_physdiff.py --dataset_root Dataset_TC --output_dir runs/physdiff_seed0 --seed 0 --gpu_num 0 --test_at_end
# Ctrl+B rồi D để detach, "tmux attach -t physdiff_train" để xem lại
```

hoặc `nohup`:

```bash
nohup python train_mgtcf.py --dataset_root Dataset_TC --output_dir runs/mgtcf_seed0 --seed 0 --gpu_num 0 --test_at_end > log_mgtcf_seed0.txt 2>&1 &
```

### 4.4. Phys-Diff — lưu ý riêng về tốc độ reverse sampling

Phys-Diff (DDPM) khác các baseline khác: mỗi lần đánh giá (`sample()`)
phải chạy **đủ `T_diffusion=1000` bước liên tiếp** (đúng công thức paper,
không hỗ trợ rút gọn bước kiểu DDIM — xem chi tiết trong docstring
`_ddpm_reverse_sample()` trong `Model/physdiff_model.py`). Điều này khiến
`sample()` chậm hơn hẳn so với các baseline khác dù đã tối ưu cache
(`ConditionalEncoder.encode_static()` — tính phần không đổi theo `t` đúng
1 lần thay vì lặp lại 1000 lần).

- `--val_freq` mặc định đã tăng lên 25 (thay vì 10 như baseline khác) và
  `--val_subset` giảm còn 100 để hạn chế tần suất/khối lượng đánh giá tốn
  thời gian trong lúc train.
- Nếu muốn theo dõi xu hướng hội tụ sát hơn (khuyến nghị vì DDPM có thể
  cần nhiều epoch hơn để `denoiser` học tốt ở mọi mức nhiễu), có thể giảm
  `--val_freq` xuống 10 và tăng `--patience`/`--num_epochs` để không dừng
  sớm trước khi thấy xu hướng thật:

```bash
python train_physdiff.py \
    --dataset_root Dataset_TC \
    --output_dir runs/physdiff_seed0 \
    --val_freq 10 --patience 300 --num_epochs 1200 \
    --seed 0 --gpu_num 0 --test_at_end
```

---

## 5. Sau khi train xong — tổng hợp/so sánh với 5 model kia

```bash
python evaluate_multi_model.py \
    --dataset_root Dataset_TC --split test \
    --fm_checkpoints        runs/fm_seed0/best_model.pth runs/fm_seed1/best_model.pth runs/fm_seed2/best_model.pth \
    --st_trans_checkpoints  runs/st_trans_seed0/best_model.pth runs/st_trans_seed1/best_model.pth runs/st_trans_seed2/best_model.pth \
    --lstm_checkpoints      runs/lstm_seed0/best_model.pth runs/lstm_seed1/best_model.pth runs/lstm_seed2/best_model.pth \
    --gru_checkpoints       runs/gru_seed0/best_model.pth runs/gru_seed1/best_model.pth runs/gru_seed2/best_model.pth \
    --rnn_checkpoints       runs/rnn_seed0/best_model.pth runs/rnn_seed1/best_model.pth runs/rnn_seed2/best_model.pth \
    --mgtcf_checkpoints     runs/mgtcf_seed0/best_model.pth runs/mgtcf_seed1/best_model.pth runs/mgtcf_seed2/best_model.pth \
    --physdiff_checkpoints  runs/physdiff_seed0/best_model.pth runs/physdiff_seed1/best_model.pth runs/physdiff_seed2/best_model.pth \
    --output_dir eval_multi/

python generate_paper_report.py \
    --records eval_multi/multi_model_test.json \
    --baseline_model FM \
    --compare_against ST-Trans RNN GRU LSTM MGTCF Phys-Diff \
    --output_dir eval_multi/
```

Kết quả: bảng số liệu (`eval_multi/paper_tables.json`) + 15 hình biểu đồ
so sánh (`eval_multi/*.png`), sẵn sàng đưa vào paper.

---

## 6. Xử lý sự cố thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `torch.cuda.is_available()` trả về `False` | Cài nhầm bản torch CPU-only, hoặc driver server không đủ mới cho bản CUDA vừa cài | Xem lại mục 1 — chạy `nvidia-smi` trước, chọn đúng `cu1xx` theo CUDA Version hiển thị |
| `ModuleNotFoundError: No module named 'Model'` | Chạy script không đúng từ `project_root/` | `cd` về đúng thư mục gốc trước khi chạy `python train_mgtcf.py` |
| `RuntimeError: CUDA out of memory` | `--batch_size` quá lớn dù A100 có nhiều VRAM (thường không xảy ra với batch mặc định 90, nhưng có thể nếu tăng quá cao ở mục 4.2) | Giảm `--batch_size`, theo dõi `nvidia-smi` |
| `FileNotFoundError` liên quan `Data1d`/`Data3d` | `--dataset_root` trỏ sai, hoặc dataset chưa giải nén đúng cấu trúc | Kiểm tra lại mục 2 |
| Phys-Diff `sample()`/validate rất chậm dù chạy A100 | Đây là đặc điểm cố hữu của DDPM full-chain sampling (1000 bước liên tiếp theo đúng công thức paper), không phải lỗi | Đã tối ưu cache `ConditionalEncoder` (giảm ~10x so với bản chưa tối ưu) — nếu vẫn cần nhanh hơn nữa, cân nhắc giảm `--val_freq` cho ít lần đánh giá hơn trong lúc train, chỉ đánh giá đầy đủ ở `--test_at_end` |
| `assert model_type in MODEL_TYPES` khi eval | Gõ nhầm `--model_type`, hoặc dùng bản `evaluate_full.py` cũ chưa hỗ trợ `mgtcf`/`physdiff` | Đảm bảo dùng đúng bản `evaluate_full.py`/`evaluate_multi_model.py` đã cập nhật hỗ trợ 7 kiến trúc |