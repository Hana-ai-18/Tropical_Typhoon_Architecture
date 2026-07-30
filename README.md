# MGTCF & Phys-Diff — Hướng dẫn cài đặt và chạy trên server

Hướng dẫn này chỉ cho 2 baseline mới (**MGTCF** — multi-generator GAN,
**Phys-Diff** — latent diffusion + PIGA). Nếu bạn cũng cần chạy 5 model
kia (LSTM/GRU/RNN/ST-Trans/FM), quy trình cài đặt/dataset giống hệt bên
dưới — chỉ khác script train cuối cùng.

---

## 1. Cài đặt môi trường

```bash
# Tạo virtual environment (khuyến nghị, tránh xung đột với package hệ thống)
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # nếu server là Windows

# Cài thư viện
pip install -r requirements.txt
```

**Yêu cầu Python ≥ 3.9** (code dùng cú pháp built-in generic `tuple[...]`,
không chạy được trên Python 3.8 trở xuống — kiểm tra bằng `python3 --version`).

### Nếu server có GPU NVIDIA

`requirements.txt` cài bản `torch` CPU-only mặc định qua PyPI. Để dùng GPU,
cài đúng bản torch khớp driver CUDA của server (xem hướng dẫn chi tiết
trong comment của `requirements.txt`):

```bash
nvidia-smi                        # xem CUDA version server đang có
# Ví dụ với CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra GPU đã nhận đúng chưa:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

---

## 2. Chuẩn bị dataset trên server

Dataset gốc của bạn (`Dataset_TC.zip`) đang nằm **cục bộ trên máy Windows**
(`C:\Users\Zenbook\Downloads\archive\Dataset_TC.zip`) — server **không thể
tự tải** đường dẫn này, bạn cần tự upload lên server bằng 1 trong các cách
sau.

### Cách 1 — `scp` từ máy Windows (PowerShell/WSL) lên server

```powershell
scp C:\Users\Zenbook\Downloads\archive\Dataset_TC.zip user@your-server-ip:/home/user/
```

### Cách 2 — Upload qua giao diện web (Kaggle/Colab/Jupyter server)

Dùng nút Upload của giao diện, hoặc kéo-thả file `.zip` vào file browser
nếu server chạy Jupyter/Kaggle Notebook.

### Cách 3 — Qua cloud storage trung gian (nếu server không có SSH trực tiếp)

Upload `Dataset_TC.zip` lên Google Drive/Dropbox trước, lấy link chia sẻ,
rồi trên server:

```bash
# Ví dụ với gdown (Google Drive)
pip install gdown
gdown "<link-file-google-drive>" -O Dataset_TC.zip
```

### Giải nén và kiểm tra cấu trúc

```bash
unzip Dataset_TC.zip -d Dataset_TC/
cd Dataset_TC
ls
```

Cấu trúc **bắt buộc** để `--dataset_root` nhận diện đúng (script tự động
tìm thư mục chứa `Data1d/` — xem `_find_tcnd_root()` trong
`Model/data/loader_training.py`):

```
Dataset_TC/
├── Data1d/
│   ├── train/     <- các file .txt track sau khi prepare_dataset.py chia
│   ├── val/
│   └── test/
├── Data3d/
│   └── <year>/<storm_name>/*.npy
└── Env_data/
    └── <year>/<storm_name>/*.npy
```

**Nếu dataset bạn tải về CHƯA được chia `train/val/test`** (chỉ có file
`.txt` phẳng trong `Data1d/`), cần chạy `prepare_dataset.py` trước (xem
lại hướng dẫn ở phần trước của dự án — file này không nằm trong phạm vi
README này vì đã có sẵn từ trước):

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

Đảm bảo cây thư mục sau khi giải nén code khớp đúng để `import Model.xxx`
hoạt động (chạy script từ thư mục gốc project, không phải từ bên trong
`Model/`):

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

Nếu thiếu file `__init__.py` trong `Model/` hoặc `Model/data/`, tạo file
rỗng:

```bash
touch Model/__init__.py Model/data/__init__.py
```

---

## 4. Chạy train

### 4.1. Test nhanh (1 epoch, kiểm tra pipeline chạy được trước khi train dài)

```bash
python train_mgtcf.py \
    --dataset_root Dataset_TC \
    --output_dir runs/mgtcf_test \
    --num_epochs 1 --val_freq 1 \
    --batch_size 8 --gpu_num 0

python train_physdiff.py \
    --dataset_root Dataset_TC \
    --output_dir runs/physdiff_test \
    --num_epochs 1 --val_freq 1 --n_sample_steps 10 \
    --batch_size 8 --gpu_num 0
```

Nếu cả 2 lệnh chạy hết 1 epoch không lỗi (in ra dòng `[VAL ep0] ADE=...`),
pipeline đã đúng — có thể chạy train đầy đủ.

### 4.2. Train đầy đủ, 3 seed (giống 5 model kia)

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

Chạy nền (không mất tiến trình nếu mất kết nối SSH), dùng `nohup` hoặc
`tmux`/`screen`:

```bash
# Cách 1: nohup
nohup python train_mgtcf.py --dataset_root Dataset_TC --output_dir runs/mgtcf_seed0 --seed 0 --gpu_num 0 --test_at_end > log_mgtcf_seed0.txt 2>&1 &

# Cách 2: tmux (khuyến nghị — dễ theo dõi log trực tiếp, attach lại sau)
tmux new -s mgtcf_train
python train_mgtcf.py --dataset_root Dataset_TC --output_dir runs/mgtcf_seed0 --seed 0 --gpu_num 0 --test_at_end
# Ctrl+B rồi D để detach, "tmux attach -t mgtcf_train" để xem lại
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
| `ModuleNotFoundError: No module named 'Model'` | Chạy script không đúng từ `project_root/` | `cd` về đúng thư mục gốc trước khi chạy `python train_mgtcf.py` |
| `RuntimeError: CUDA out of memory` | `--batch_size` quá lớn so với GPU | Giảm `--batch_size` (thử 32/16/8), hoặc thêm `--gpu_num` đúng GPU còn trống nếu server nhiều GPU |
| `FileNotFoundError` liên quan `Data1d`/`Data3d` | `--dataset_root` trỏ sai, hoặc dataset chưa giải nén đúng cấu trúc | Kiểm tra lại mục 2, đảm bảo `Data1d/train`, `Data1d/val`, `Data1d/test` tồn tại |
| Training rất chậm dù có GPU | `torch` cài bản CPU-only dù server có GPU | Kiểm tra lại mục 1 — cài đúng bản torch có CUDA |
| `assert model_type in MODEL_TYPES` khi eval | Gõ nhầm `--model_type` hoặc thiếu `mgtcf`/`physdiff` trong `evaluate_full.py`'s `MODEL_TYPES` | Đảm bảo dùng đúng bản `evaluate_full.py` đã cập nhật hỗ trợ 7 kiến trúc |