"""
n_only_calibration_sweep.py — Sweep RÚT GỌN, chỉ theo trục N (K cố định
ở giá trị lớn nhất, mặc định K=30), dùng RIÊNG để lấy đúng
spread_skill_ratio cho Bước A của select_kn.py.

TẠI SAO SCRIPT NÀY TỒN TẠI (không sửa/chạy lại k_n_joint_sweep cũ):
  - select_kn.py Bước A (select_n_by_calibration) chỉ dùng field
    "spread_skill_ratio", GỘP QUA MỌI K cho mỗi N -- nghĩa là kết quả
    Bước A không phụ thuộc vào việc bạn có quét đủ 5 giá trị K hay
    không. Do đó không cần chạy lại toàn bộ lưới 5(K) x 9(N) = 45 cell
    như trước -- chỉ cần 9 cell (9 giá trị N, K cố định = giá trị lớn
    nhất bạn có, để candidate pool đủ lớn cho spread ổn định) x 3 seed
    = 27 lần, thay vì 135 lần.
  - File k_n_sweep_multiseed_fm_val.json ĐÃ CÓ (từ lần chạy trước) vẫn
    được giữ nguyên, dùng cho Bước B (chọn K theo ADE, không cần
    spread_skill_ratio) -- không mất bất kỳ kết quả cũ nào.
  - merge_calibration_sweep() ở cuối file gộp 2 nguồn (ADE cũ theo K,N
    + spread_skill_ratio mới theo N) thành 1 file JSON select_kn.py có
    thể đọc thẳng, không cần sửa gì thêm ở select_kn.py.

CÁCH DÙNG (ví dụ, thay checkpoint path cho đúng máy bạn):
    python n_only_calibration_sweep.py \
        --checkpoints /kaggle/input/.../best_model_fm_seed0_new_ver.pth \
                      /kaggle/input/.../best_model_fm_seed1_new_ver.pth \
                      /kaggle/input/.../best_model_fm_seed2_new_ver.pth \
        --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
        --split val \
        --n_values 1 4 8 10 12 16 20 25 30 \
        --k_fixed 30 \
        --old_kn_json /kaggle/working/eval_full_VAL/k_n_sweep_multiseed_fm_val.json \
        --out /kaggle/working/eval_full_VAL/k_n_sweep_multiseed_fm_val_WITH_RATIO.json

Sau đó chạy select_kn.py NHƯ CŨ, chỉ đổi --sweep_json sang file
"..._WITH_RATIO.json" mới này.
"""
import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from Model.data.loader_training import data_loader
from Model.flow_matching_model import TCFlowMatching, EMAModel, _norm_to_deg, _haversine_deg

try:
    # tái dùng loader/build/infer-seed logic đã có trong evaluate_full.py,
    # thay vì viết lại — tránh một bản logic thứ hai có thể lệch dần
    # theo thời gian so với bản gốc.
    from evaluate_full import (
        _build_model, _infer_model_type_from_checkpoint, move,
    )
except ImportError as e:
    raise ImportError(
        "n_only_calibration_sweep.py phải được đặt CẠNH evaluate_full.py "
        "(đã patch, có spread_skill_ratio) để tái dùng _build_model/"
        "_infer_model_type_from_checkpoint/move. Không tự viết lại các "
        "hàm này ở đây để tránh 2 bản logic load-checkpoint lệch nhau."
    ) from e


def _infer_seed(checkpoint_path: str, ck: dict) -> str:
    if isinstance(ck, dict) and "seed" in ck:
        return str(ck["seed"])
    import re
    m = re.search(r"seed[_-]?(\d+)", checkpoint_path)
    return m.group(1) if m else "unknown"


@torch.no_grad()
def n_only_sweep(model, loader, device, n_values: List[int], k_fixed: int,
                  use_tta: bool = True, n_tta: int = 5,
                  use_curvature_score: bool = True) -> Dict:
    """
    Với K cố định = k_fixed (mặc định 30, khớp giá trị K lớn nhất đã
    dùng trong sweep cũ để candidate pool đủ lớn cho spread ổn định),
    quét qua từng N trong n_values, tính spread_skill_ratio ĐÚNG CHUẨN
    (spread(tau)/RMSE(tau), trung bình qua 12 horizon) -- logic giống
    hệt phần đã patch trong evaluate_full.py's k_n_joint_sweep(), chỉ
    khác là bỏ hẳn trục K để không lặp lại công việc không cần thiết.
    """
    model.eval()
    raw = model.module if hasattr(model, "module") else model
    orig_steps = getattr(raw, "n_inference_steps", 1)
    tta_scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]

    results = {}
    for n_steps in n_values:
        print(f"  [N={n_steps}] K_fixed={k_fixed} ...")
        t0 = time.time()
        by_lt_spread = defaultdict(list)
        by_lt_ens_sqerr = defaultdict(list)

        for batch in loader:
            bl = move(list(batch), device)
            gt = bl[1]
            try:
                raw.n_inference_steps = n_steps
                if use_tta:
                    obs = bl[0]
                    anchor = obs[-1:, :, :2].detach()
                    all_t = None
                    for sc in tta_scales:
                        obs_s = obs.clone()
                        obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                        bl_s = list(bl); bl_s[0] = obs_s
                        _, _, at = model.sample(bl_s, num_ensemble=k_fixed,
                                                 use_curvature_score=use_curvature_score)
                        if abs(sc - 1.0) < 1e-6:
                            all_t = at  # spread chỉ lấy từ pass KHÔNG scale, khớp evaluate_full.py cũ
                else:
                    _, _, all_t = model.sample(bl, num_ensemble=k_fixed,
                                                use_curvature_score=use_curvature_score)
                raw.n_inference_steps = orig_steps
            except Exception as e:
                raw.n_inference_steps = orig_steps
                print(f"    Error at N={n_steps}: {e}")
                continue

            if all_t is None or not torch.is_tensor(all_t) or all_t.shape[0] < 2:
                continue
            Kb = all_t.shape[0]
            T_all = min(all_t.shape[1], gt.shape[0])
            all_t_deg = _norm_to_deg(all_t[:, :T_all, :, :2])   # [K, T_all, B, 2]
            gd_all = _norm_to_deg(gt[:T_all, :, :2])            # [T_all, B, 2]

            for tt in range(T_all):
                lt = tt + 1
                step_k = all_t_deg[:, tt]                        # [K, B, 2]
                idx1 = torch.randperm(Kb)[:min(Kb, 10)]
                idx2 = torch.randperm(Kb)[:min(Kb, 10)]
                for a, b in zip(idx1.tolist(), idx2.tolist()):
                    if a != b:
                        dab = _haversine_deg(step_k[a:a+1], step_k[b:b+1]).squeeze(0)
                        by_lt_spread[lt].append(float(dab.mean()))
                ens_mean = step_k.mean(0, keepdim=True)
                se = (_haversine_deg(ens_mean, gd_all[tt:tt+1]) ** 2)
                by_lt_ens_sqerr[lt].append(float(se.mean()))

        ss_ratio_by_lt = {}
        for lt in sorted(set(by_lt_spread.keys()) | set(by_lt_ens_sqerr.keys())):
            sp = float(np.mean(by_lt_spread[lt])) if by_lt_spread.get(lt) else float("nan")
            mse = float(np.mean(by_lt_ens_sqerr[lt])) if by_lt_ens_sqerr.get(lt) else float("nan")
            rmse = np.sqrt(mse) if mse == mse else float("nan")
            ss_ratio_by_lt[lt] = sp / (rmse + 1e-6) if rmse == rmse else float("nan")
        valid_ratios = [v for v in ss_ratio_by_lt.values() if v == v]
        ratio_mean = float(np.mean(valid_ratios)) if valid_ratios else float("nan")

        elapsed = time.time() - t0
        results[n_steps] = {
            "N": n_steps, "K_fixed": k_fixed,
            "spread_skill_ratio": ratio_mean,
            "spread_skill_ratio_by_lead_time": ss_ratio_by_lt,
            "time_s": elapsed,
        }
        print(f"    spread_skill_ratio={ratio_mean:.3f}  t={elapsed:.1f}s")

    return results


def run_multi_seed(checkpoints: List[str], dataset_root: str, split: str,
                    n_values: List[int], k_fixed: int, device, args) -> Dict:
    per_seed = {}
    for ckpt_path in checkpoints:
        print(f"\n{'='*70}\nLoading checkpoint: {ckpt_path}\n{'='*70}")
        ck = torch.load(ckpt_path, map_location="cpu")
        model_cfg = ck.get("model_cfg") or {}
        resolved_type = _infer_model_type_from_checkpoint(ck, "fm")
        model = _build_model(resolved_type, model_cfg, device)
        state = ck.get("model", ck)
        model.load_state_dict(state, strict=False)
        if not args.no_ema and ck.get("ema"):
            try:
                ema = EMAModel(model)
                for k, v in ck["ema"].items():
                    if k in ema.shadow:
                        ema.shadow[k].copy_(v.to(device))
                print(f"  EMA loaded ({len(ema.shadow)} params)")
            except Exception as e:
                print(f"  \u26a0 EMA failed: {e}")

        seed = _infer_seed(ckpt_path, ck)
        print(f"  seed={seed}  epoch={ck.get('epoch', '?')}")

        import argparse as _ap
        _loader_args = _ap.Namespace(
            dataset_root=dataset_root, obs_len=8, pred_len=12,
            batch_size=64, num_workers=2, test_year=None, skip=1,
            min_ped=1, threshold=0.002,
        )
        _, loader = data_loader(_loader_args, {"root": dataset_root, "type": split},
                                 test=(split != "train"))
        print(f"  Data: {len(loader)} batches")

        model.eval()
        per_seed[seed] = n_only_sweep(model, loader, device, n_values, k_fixed,
                                       use_tta=args.use_tta, n_tta=args.n_tta,
                                       use_curvature_score=args.use_curvature_score)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_seeds = len(per_seed)
    print(f"\nGop qua {n_seeds} seed: {list(per_seed.keys())}")
    merged = {}
    for n_steps in n_values:
        vals = [per_seed[s][n_steps]["spread_skill_ratio"] for s in per_seed
                if n_steps in per_seed[s]
                and per_seed[s][n_steps]["spread_skill_ratio"] == per_seed[s][n_steps]["spread_skill_ratio"]]
        per_lt = defaultdict(list)
        for s in per_seed:
            if n_steps not in per_seed[s]:
                continue
            for lt, v in per_seed[s][n_steps]["spread_skill_ratio_by_lead_time"].items():
                if v == v:
                    per_lt[lt].append(v)
        merged[n_steps] = {
            "N": n_steps, "K_fixed": k_fixed,
            "spread_skill_ratio": float(np.mean(vals)) if vals else float("nan"),
            "spread_skill_ratio_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "spread_skill_ratio_by_lead_time": {lt: float(np.mean(v)) for lt, v in per_lt.items()},
            "n_seeds": n_seeds,
        }
    return merged


def merge_calibration_sweep(old_kn_json_path: str, ratio_by_n: Dict, out_path: str):
    """
    Gop file k_n_sweep_multiseed_fm_val.json CU (co ADE/ATE/CTE theo tung
    (K,N), dung cho Buoc B) voi ratio_by_n MOI (spread_skill_ratio theo
    tung N, K co dinh, dung cho Buoc A). Khong sua doi gia tri cu, chi
    THEM field "spread_skill_ratio" vao moi cell (K,N) da co, lay tu gia
    tri cua dung N tuong ung (K cu bi bo qua boi Buoc A, nhu giai thich
    o dau file, nen viec "dung chung 1 ratio cho moi K cua cung 1 N" la
    dung voi logic select_kn.py, khong phai xap xi).
    """
    with open(old_kn_json_path) as f:
        old = json.load(f)
    sweep = old["k_n_sweep"] if "k_n_sweep" in old else old

    n_missing = []
    for key, cell in sweep.items():
        n_val = cell.get("N")
        if n_val is None:
            k_str, n_str = key.split(",")
            n_val = int(n_str)
        if n_val in ratio_by_n:
            cell["spread_skill_ratio"] = ratio_by_n[n_val]["spread_skill_ratio"]
            cell["spread_skill_ratio_by_lead_time"] = ratio_by_n[n_val]["spread_skill_ratio_by_lead_time"]
        else:
            n_missing.append(n_val)

    if n_missing:
        print(f"  \u26a0 CANH BAO: cac gia tri N sau co trong file cu nhung "
              f"KHONG co trong sweep moi (thieu spread_skill_ratio o cell "
              f"tuong ung): {sorted(set(n_missing))}. Kiem tra lai --n_values "
              f"co khop voi file cu khong.")

    out = {"k_n_sweep": sweep}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Da luu file gop -> {out_path}")
    print(f"  Dung file nay lam --sweep_json cho select_kn.py (khong doi gi "
          f"khac o select_kn.py).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_values", type=int, nargs="+",
                     default=[1, 4, 8, 10, 12, 16, 20, 25, 30])
    ap.add_argument("--k_fixed", type=int, default=30,
                     help="K co dinh dung cho n-only sweep. Nen dat bang "
                          "K lon nhat da dung trong sweep K,N cu, de "
                          "candidate pool du lon cho spread on dinh.")
    ap.add_argument("--old_kn_json", required=True,
                     help="File k_n_sweep_multiseed_fm_val.json DA CO tu "
                          "lan chay truoc (khong bi xoa/doi gi).")
    ap.add_argument("--out", required=True,
                     help="File JSON moi, da gop, dung lam --sweep_json "
                          "cho select_kn.py.")
    ap.add_argument("--use_tta", action="store_true", default=True)
    ap.add_argument("--no_tta", dest="use_tta", action="store_false")
    ap.add_argument("--n_tta", type=int, default=5)
    ap.add_argument("--use_curvature_score", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_true", default=False)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ratio_by_n = run_multi_seed(args.checkpoints, args.dataset_root, args.split,
                                 args.n_values, args.k_fixed, device, args)
    merge_calibration_sweep(args.old_kn_json, ratio_by_n, args.out)


if __name__ == "__main__":
    main()