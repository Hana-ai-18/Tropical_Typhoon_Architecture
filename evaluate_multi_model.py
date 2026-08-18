"""
evaluate_multi_model.py
=========================
Runs test-set evaluation for RNN / GRU / LSTM / ST-Trans / FM / MMSTN /
Phys-Diff / TC-Diffuser checkpoints, using ONE SHARED ATE/CTE formula for
all models.

WHY A SEPARATE SCRIPT, NOT REUSING EACH MODEL'S OWN METRIC CODE
-----------------------------------------------------------------
paper_baseline_model.py (used by RNN/GRU/LSTM/ST-Trans) has its own
_ate_cte_tensors() with a DIFFERENT convention than flow_matching_model.py's
_ate_cte_full() (used by FM):
  - flat-earth approx (111*cos(lat)) vs haversine — small effect (~0.2%)
  - reference heading computed OUTGOING from each point (gt[t]->gt[t+1])
    vs INCOMING to each point (gt[t-1]->gt[t]) — this is NOT a small
    effect: verified numerically on a synthetic turning trajectory, the
    two conventions can even disagree on the SIGN of CTE at a given step.
Comparing FM's ATE/CTE (computed one way) against RNN/GRU/LSTM/ST-Trans's
ATE/CTE (computed the other way) would not be a fair like-for-like
comparison — any difference could be an artifact of the formula, not the
model. This script computes ADE/ATE/CTE for ALL EIGHT models using the
SAME function (_ate_cte_full, _haversine_deg, _forward_azimuth, imported
from flow_matching_model.py — the version with the off-by-one fix already
verified in evaluate_full.py), so Table-10-style comparisons are sound.
MMSTN (Model/mmstn_model.py), Phys-Diff (Model/phys_diff_model.py), and
TC-Diffuser (Model/tc_diffuser_model.py) all use this exact same shared
formula too — no separate ATE/CTE code of their own, so they slot into the
comparison with the same guarantee. All three also share PaperEncoder
(FNO3D+Mamba+Env_net) with every non-FM baseline, for a fair encoder-level
comparison — see each model file's own docstring for details.

NOTE ON TC-DIFFUSER'S SAMPLING: unlike the other seven models (single-
sample or FM's own ensemble-averaged ADE), TC-Diffuser reports ADE using
best-of-K sampling by ORIGINAL-repo design (see load_tc_diffuser's
docstring) — this is not directly comparable to the others without
accounting for that difference; the JSON output does not currently encode
this distinction per-record, so track it separately when reading results.

USAGE
-----
python evaluate_multi_model.py \
    --dataset_root <root> \
    --fm_checkpoint runs/fm_seed42/best_model.pth \
    --st_trans_checkpoint runs/st_trans/best_model.pth \
    --lstm_checkpoint runs/lstm/best_model.pth \
    --gru_checkpoint runs/gru/best_model.pth \
    --rnn_checkpoint runs/rnn/best_model.pth \
    --mmstn_checkpoint runs/mmstn/best_model.pth \
    --physdiff_checkpoint runs/phys_diff/best_model.pth \
    --tcdiffuser_checkpoint runs/tc_diffuser/best_model.pth \
    --output_dir eval_multi/

Any checkpoint arg can be omitted to skip that model. Produces one
combined JSON of per-window records (one row per (model, storm, window))
that generate_comparison_table.py consumes directly for the Table-10-style
statistical significance tables.
"""
from __future__ import annotations
import sys, os, argparse, json, random
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
)
from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
from Model.st_trans_model import STTrans
# MMSTN (Social-GAN style), Phys-Diff (PIGA-augmented DDPM), and TC-Diffuser
# (velocity-space DDPM) all share PaperEncoder (FNO3D+Mamba+Env_net) with
# every other baseline in this comparison -- see each model file's own
# docstring for what was kept faithful to its original repo vs. adapted.
# All three are generative/stochastic models (like FM), so they use the
# "ensemble" sampling path in evaluate_one_model() below, though each has
# its OWN sampling knob (MMSTN: single sample by original design; Phys-Diff:
# DDIM sample_steps; TC-Diffuser: best_k) rather than accepting a runtime
# num_ensemble override the way FM's ddim_steps does -- see is_ensemble_model
# and the --mmstn_*/--physdiff_*/--tcdiffuser_* eval-time override args in
# main() for how each is configured explicitly instead of silently ignored.
from Model.Mmstn_model import MMSTN
from Model.physdiff_model import PhysDiff
from Model.tc_diffuser_model import TCDiffuser


def _infer_seed(checkpoint_path: str, ck: dict) -> str:
    """
    Best-effort seed extraction: prefer an explicit 'seed' key saved in
    the checkpoint (train_flowmatching.py / train_st_trans.py /
    train_paper_baseline.py all save this per the project's multi-seed
    convention). Falls back to parsing "_seed<N>" from the path/dirname
    (the convention those same scripts use for output_dir), and finally
    "unknown" if neither is found — records with seed="unknown" will
    NOT be treated as genuinely pooled multi-seed data by
    generate_paper_table.py (it warns explicitly in that case).
    """
    if isinstance(ck, dict) and "seed" in ck:
        return str(ck["seed"])
    import re
    m = re.search(r"seed[_-]?(\d+)", checkpoint_path)
    if m:
        return m.group(1)
    return "unknown"


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
    """
    SAME formula as evaluate_full.py's _ate_cte_full (off-by-one already
    fixed there — see that file's own comments for the fix history).
    Returns [T-1, B] each; index k holds the error at ORIGINAL step k+1.

    [DESIGN DECISION, confirmed with user] Used for ALL models in this
    script (FM, ST-Trans, LSTM, GRU, RNN, MMSTN, Phys-Diff, TC-Diffuser),
    not just FM — even though only FM's own training/eval code
    (flow_matching_model.py / train_flowmatching.py) actually uses this
    formula; every other model imports and trains with
    paper_baseline_model.py's own _ate_cte_tensors instead (a different,
    flat-plane-projection formula with a different heading convention).
    This is intentional: using ONE shared formula for every model in
    this comparison script ensures FM-vs-baseline ATE/CTE differences
    reflect genuine model quality, not which formula happened to be
    used — see this file's module docstring for the full rationale.
    Consequence: ATE/CTE reported here for the 7 non-FM models will NOT
    match what each model's own train_*.py printed right after
    training — that mismatch is expected and correct for this script's
    purpose (fair cross-model comparison), not a bug.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T - 1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


def load_fm(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
              f"defaults (correct only if trained with default architecture).")
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ FM load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        if missing:
            print(f"    missing keys: {missing}")
        if unexpected:
            print(f"    unexpected keys: {unexpected}")

    # [FIX-EMA-MISMATCH] train_flowmatching.py's own validation/eval loop
    # ALWAYS calls ema.apply_to(model) before scoring whenever an EMA
    # shadow exists (see evaluate()'s `if ema is not None: bk = ema.apply_to(model)`,
    # called at every eval during training). But ck["model"] saved by _save()
    # is the RAW, non-EMA state_dict — ema.shadow is written to a SEPARATE
    # ck["ema"] key that this loader previously never read. That mismatch
    # is the primary reason numbers printed right after training (EMA
    # weights) differ from this script's FM numbers (raw weights) even on
    # the exact same checkpoint file. Applying it here restores parity.
    #
    # is_swa checkpoints are a DIFFERENT, mutually-exclusive averaging
    # scheme (SWAHandler.save_avg_state's ck["model"] IS ALREADY the SWA
    # running average) -- do not also apply ema on top of that.
    is_swa = ck.get("is_swa", False)
    if ck.get("ema") and not is_swa:
        sd = model.state_dict()
        applied, skipped = 0, 0
        for k, v in ck["ema"].items():
            if k in sd:
                sd[k].copy_(v.to(device))
                applied += 1
            else:
                skipped += 1
        print(f"  ✓ Applied EMA shadow weights ({applied} tensors"
              f"{f', {skipped} skipped (not in current model)' if skipped else ''}) "
              f"— matches training-time eval convention.")
    elif is_swa:
        print(f"  ℹ Checkpoint is an SWA average (is_swa=True) — "
              f"ck['model'] IS the SWA running average, no separate EMA applied.")
    else:
        print(f"  ⚠ No 'ema' key found in checkpoint — evaluating RAW weights. "
              f"If this checkpoint was saved mid-training with use_ema=True, "
              f"this will NOT match the ADE/ATE printed during training.")

    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_paper_baseline(checkpoint: str, model_type: str, device,
                         hidden_dim: int = 256, n_layers: int = 3,
                         obs_len: int = 8, pred_len: int = 12,
                         unet_in_ch: int = 13, dropout: float = 0.20):
    """
    [UPDATED] train_paper_baseline.py now saves a full model_cfg dict in
    newer checkpoints (added alongside --seed support). If present, it is
    used directly and the CLI-default args above are ignored — this is
    the reliable path. If absent (checkpoint trained before that fix),
    falls back to the CLI-default-matching values passed in here, with a
    warning, same caveat as FM's missing-model_cfg case.
    """
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", model_type)
    if saved_type != model_type:
        # [FIX-SILENT-MISLABEL] This was previously a quiet ⚠, easy to miss
        # in a long log, and the code proceeded anyway to force-build a
        # {model_type} architecture (e.g. PaperGRUHead) and load an RNN/LSTM
        # checkpoint's state_dict into it with strict=False. Because
        # PaperGRUHead/PaperRNNHead/PaperLSTMHead share the same
        # nn.ModuleList("cells")/fc naming pattern, a meaningful FRACTION of
        # keys can share names even though the underlying cell type
        # (GRUCell vs RNNCell vs LSTMCell) differs -- so load_state_dict can
        # load MOST weights without a large missing/unexpected count,
        # producing a "Frankenstein" model (wrong architecture, partially
        # mismatched weights) that still gives PLAUSIBLE-LOOKING (not
        # obviously garbage) ADE/ATE/CTE numbers instead of an obvious
        # failure. This is exactly how a checkpoint path accidentally
        # pointed at the WRONG --xxx_checkpoints flag (e.g. RNN checkpoints
        # passed to --gru_checkpoints) can silently produce a record
        # labeled "GRU" in the output JSON that is actually RNN data run
        # through GRU-shaped (but wrong-celltype) weights -- confirmed as
        # the root cause of a real observed mislabeling incident. Now
        # raises instead of silently continuing: the checkpoint path is
        # almost certainly wrong, and continuing would silently corrupt
        # the model-name labeling of every downstream record/table.
        raise ValueError(
            f"🛑 Checkpoint's saved model_type='{saved_type}' does NOT match "
            f"the requested '{model_type}' (checkpoint path: {checkpoint}). "
            f"This almost always means the wrong file was passed to "
            f"--{model_type}_checkpoints (e.g. an RNN checkpoint accidentally "
            f"pointed to by --gru_checkpoints) -- continuing would silently "
            f"build a '{model_type}' architecture and load '{saved_type}' "
            f"weights into it via strict=False, producing a real-looking but "
            f"WRONG result mislabeled as '{model_type}' in the output JSON. "
            f"Fix the --{model_type}_checkpoints path(s) to point at the "
            f"correct {model_type.upper()} checkpoint file(s) and re-run."
        )

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PaperBaseline(**model_cfg).to(device)
    else:
        print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — "
              f"using CLI-default-matching args (hidden_dim={hidden_dim}, "
              f"n_layers={n_layers}, dropout={dropout}). Only correct if "
              f"trained with train_paper_baseline.py's own defaults.")
        model = PaperBaseline(model_type=model_type, pred_len=pred_len,
                               obs_len=obs_len, hidden_dim=hidden_dim,
                               n_layers=n_layers, unet_in_ch=unet_in_ch,
                               dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ {model_type.upper()} load_state_dict: {len(missing)} "
              f"missing, {len(unexpected)} unexpected keys")
        # [FIX-MISSING-MISMATCH-WARNING] load_st_trans/load_mmstn/
        # load_phys_diff/load_tc_diffuser all warn loudly (🛑) when a
        # large fraction of parameter keys mismatch -- signaling the
        # loaded weights are essentially random-init, not the trained
        # checkpoint, so any ADE/ATE/CTE from them is meaningless. This
        # function (used for GRU/RNN/LSTM) was missing that same check
        # entirely, only printing a quiet ⚠ with a raw key count that's
        # easy to miss in a long log -- exactly the failure mode behind
        # an ADE~1260km result (a checkpoint from a mismatched dataset/
        # path silently loading as near-random weights instead of
        # erroring out or warning clearly). Added for parity with every
        # other loader in this file.
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of {model_type.upper()}'s parameter keys did not "
                  f"match this checkpoint's state_dict -- loaded weights are almost "
                  f"entirely random-initialized. Any ADE/ATE/CTE reported below is "
                  f"NOT a valid evaluation -- verify the checkpoint path/architecture "
                  f"(hidden_dim/n_layers/dropout matching what this checkpoint was "
                  f"actually trained with) before trusting it.")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_st_trans(checkpoint: str, device,
                   obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                   d_model: int = 64, nhead: int = 4, num_enc_layers: int = 1,
                   num_dec_layers: int = 3, dim_ff: int = 512, dropout: float = 0.1):
    """
    [UPDATED] Same model_cfg pattern as load_paper_baseline. Note STTrans
    (non_ar) and STTransAR have DIFFERENT constructor signatures (AR has
    no num_dec_layers) — the saved model_cfg already accounts for this
    (see train_st_trans.py's checkpoint-save branch), so **kwargs here
    naturally works for either. This loader only instantiates STTrans
    (non-AR) — STTransAR support would need its own loader if you also
    want to evaluate that variant.

    [FIX-STTRANS-MODEL-CFG] Checkpoints saved by an OLDER version of
    train_st_trans.py (before the "STTrans vs STTransAR have different
    ctor signatures" fix was added there) can have a model_cfg dict that
    includes extra keys STTrans.__init__() does not accept -- observed in
    practice: 'model_type' (TypeError: STTrans.__init__() got an
    unexpected keyword argument 'model_type'). Current train_st_trans.py
    saves model_type as a SEPARATE top-level checkpoint key (ck['model_type']),
    not inside model_cfg, so a clean checkpoint should not hit this -- but
    since we cannot control what checkpoint file the user points this at
    (e.g. one trained before that convention existed), filter model_cfg
    down to exactly STTrans's known-valid non_ar constructor keys before
    calling STTrans(**model_cfg), rather than trusting the checkpoint's
    model_cfg blindly. Any dropped key is reported so a real config
    mismatch (not just a harmless stale extra key) is still visible.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg")
    if model_cfg:
        _VALID_STTRANS_KEYS = {
            "obs_len", "pred_len", "unet_in_ch", "d_model", "nhead",
            "num_enc_layers", "num_dec_layers", "dim_ff", "dropout",
        }
        _dropped = {k: v for k, v in model_cfg.items() if k not in _VALID_STTRANS_KEYS}
        if _dropped:
            # [FIX-STTRANS-WRONG-CKPT] 'hidden_dim'/'n_layers' are
            # PaperBaseline-specific (RNN/GRU/LSTM), never valid STTrans
            # ctor args under ANY version of train_st_trans.py -- their
            # presence means this file is almost certainly a PaperBaseline
            # checkpoint pointed at by the wrong --st_trans_checkpoints
            # path/filename, NOT a genuinely-older-but-valid STTrans
            # checkpoint. Silently dropping and proceeding in that case
            # would build a randomly-initialized STTrans and load a
            # state_dict for a totally different architecture (near-total
            # missing/unexpected key mismatch), producing a model that
            # LOOKS like it loaded (no crash) but is untrained garbage --
            # exactly the ADE~6000km symptom this warning exists to catch
            # before it's mistaken for a legitimate eval result.
            _paper_baseline_keys = _dropped.keys() & {"model_type", "hidden_dim", "n_layers"}
            if _paper_baseline_keys:
                print(f"  🛑 ST-Trans checkpoint's model_cfg contains "
                      f"{sorted(_paper_baseline_keys)} — these are PaperBaseline "
                      f"(RNN/GRU/LSTM) architecture keys, NEVER valid for STTrans. "
                      f"This file is very likely NOT an ST-Trans checkpoint at all "
                      f"(wrong path in --st_trans_checkpoints?). Proceeding anyway "
                      f"per current call, but expect load_state_dict to report a "
                      f"large missing/unexpected key count below if so — if that "
                      f"happens, the resulting ADE number is meaningless (near-random "
                      f"weights), not a real evaluation of ST-Trans. Verify the "
                      f"checkpoint path before trusting any metric from this run.")
            else:
                print(f"  ⚠ ST-Trans checkpoint's model_cfg has keys STTrans.__init__() "
                      f"doesn't accept — dropping before construction: {list(_dropped.keys())}. "
                      f"(Likely saved by an older train_st_trans.py; verify this is really "
                      f"an STTrans non_ar checkpoint, not e.g. STTransAR, if unsure.)")
        model_cfg = {k: v for k, v in model_cfg.items() if k in _VALID_STTRANS_KEYS}
        model = STTrans(**model_cfg).to(device)
    else:
        print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using "
              f"CLI-default-matching args (d_model={d_model}, nhead={nhead}, "
              f"num_dec_layers={num_dec_layers}). Only correct if trained "
              f"with train_st_trans.py's own defaults.")
        model = STTrans(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                         d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
                         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
                         dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ ST-Trans load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        # [FIX-STTRANS-WRONG-CKPT] A handful of missing/unexpected keys can
        # legitimately happen (e.g. a minor architecture tweak between
        # checkpoint versions). A LARGE fraction mismatching means the
        # state_dict almost certainly belongs to a different architecture
        # entirely -- most of the model is left at its random init, so any
        # ADE/ATE/CTE computed from it is meaningless, not a real result.
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of STTrans's parameter keys did not "
                  f"match this checkpoint's state_dict -- this is FAR too many for a "
                  f"minor version difference. The loaded weights are almost entirely "
                  f"random-initialized, not the trained checkpoint. Any metric reported "
                  f"below (ADE/ATE/CTE) is NOT a valid evaluation of ST-Trans -- stop and "
                  f"verify the checkpoint path/architecture before using these numbers.")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_mmstn(checkpoint: str, device,
               embedding_dim: int = 32, encoder_h_dim_g: int = 64,
               decoder_h_dim_g: int = 64, encoder_h_dim_d: int = 128,
               mlp_dim: int = 128, num_layers: int = 1, noise_dim: int = 16,
               noise_type: str = "gaussian", dropout: float = 0.0,
               best_k: int = 6, l2_loss_weight: float = 1.0,
               obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13):
    """
    Load MMSTN checkpoint (train_mmstn.py, Model/mmstn_model.py::MMSTN).
    Social-GAN-style baseline (Generator+Discriminator), shares PaperEncoder
    with every other baseline in this comparison. Same model_cfg/CLI-default
    fallback pattern as load_st_trans/load_paper_baseline.

    IMPORTANT TYPE FIX: train_mmstn.py saves model_cfg["noise_dim"] as a
    plain int (args.noise_dim), but MMSTN.__init__ expects noise_dim as a
    TUPLE (Tuple[int, ...] = (16,) by default) -- it does `noise_dim[0]`
    internally, which raises "int is not subscriptable" if passed a bare
    int. This loader wraps noise_dim into a 1-tuple before construction,
    whether it comes from a saved model_cfg or the CLI-default fallback.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", "MMSTN")
    if saved_type.lower() != "mmstn":
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' is not "
              f"'MMSTN' — verify this checkpoint path is really the MMSTN one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model_cfg = dict(model_cfg)  # don't mutate the checkpoint's own dict
        if "noise_dim" in model_cfg and not isinstance(model_cfg["noise_dim"], (tuple, list)):
            model_cfg["noise_dim"] = (model_cfg["noise_dim"],)
        elif "noise_dim" in model_cfg:
            model_cfg["noise_dim"] = tuple(model_cfg["noise_dim"])
        model = MMSTN(**model_cfg).to(device)
    else:
        print(f"  ⚠ MMSTN checkpoint has no model_cfg — using CLI-default"
              f"-matching args (embedding_dim={embedding_dim}, "
              f"encoder_h_dim_g={encoder_h_dim_g}, best_k={best_k}). Only "
              f"correct if trained with train_mmstn.py's own defaults.")
        model = MMSTN(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                       embedding_dim=embedding_dim, encoder_h_dim_g=encoder_h_dim_g,
                       decoder_h_dim_g=decoder_h_dim_g, encoder_h_dim_d=encoder_h_dim_d,
                       mlp_dim=mlp_dim, num_layers=num_layers,
                       noise_dim=(noise_dim,), noise_type=noise_type,
                       dropout=dropout, best_k=best_k,
                       l2_loss_weight=l2_loss_weight).to(device)

    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ MMSTN load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of MMSTN's parameter keys did not "
                  f"match this checkpoint's state_dict -- loaded weights are almost "
                  f"entirely random-initialized. Any ADE/ATE/CTE reported below is "
                  f"NOT a valid evaluation -- verify the checkpoint path/architecture "
                  f"before trusting it.")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_phys_diff(checkpoint: str, device,
                    d_model: int = 128, d_embedding: int = 64,
                    enc_layers: int = 3, enc_heads: int = 4, enc_ff: int = 256,
                    enc_dropout: float = 0.1, dec_layers: int = 3, dec_heads: int = 4,
                    dec_ff: int = 256, dec_dropout: float = 0.1, d_sub: int = 16,
                    gate_mlp_dims=(64, 16, 1), num_timesteps: int = 1000,
                    beta_schedule: str = "cosine", beta_start: float = 0.0001,
                    beta_end: float = 0.02, sample_steps: int = 50,
                    coord_loss_weight: float = 1.0, diffusion_loss_weight: float = 1.0,
                    obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                    eval_sample_steps: Optional[int] = None):
    """
    Load Phys-Diff checkpoint (train_phys_diff.py, Model/phys_diff_model.py
    ::PhysDiff). PIGA-augmented DDPM, shares PaperEncoder with every other
    baseline. Same model_cfg/CLI-default fallback pattern as load_st_trans.

    eval_sample_steps: if given, OVERRIDES model.sample_steps after loading
    (the DDIM-strided reverse-step count used by model.sample()). The value
    saved in the checkpoint's model_cfg is whatever was used for cheap
    per-epoch validation during training (--sample_steps, default 50) --
    passing a larger value here (e.g. via --physdiff_eval_sample_steps)
    trades eval time for a more accurate (lower-variance) final sample,
    exactly like train_phys_diff.py's own --sample_steps controls training-
    time validation cost. This does NOT change what was trained, only how
    many reverse diffusion steps are used to produce the final sample.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", "PhysDiff")
    if saved_type.lower() != "physdiff":
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' is not "
              f"'PhysDiff' — verify this checkpoint path is really the Phys-Diff one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PhysDiff(**model_cfg).to(device)
    else:
        print(f"  ⚠ Phys-Diff checkpoint has no model_cfg — using CLI-default"
              f"-matching args (d_model={d_model}, d_embedding={d_embedding}). "
              f"Only correct if trained with train_phys_diff.py's own defaults.")
        model = PhysDiff(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                          d_model=d_model, d_embedding=d_embedding,
                          enc_layers=enc_layers, enc_heads=enc_heads, enc_ff=enc_ff,
                          enc_dropout=enc_dropout, dec_layers=dec_layers,
                          dec_heads=dec_heads, dec_ff=dec_ff, dec_dropout=dec_dropout,
                          d_sub=d_sub, gate_mlp_dims=tuple(gate_mlp_dims),
                          num_timesteps=num_timesteps, beta_schedule=beta_schedule,
                          beta_start=beta_start, beta_end=beta_end,
                          sample_steps=sample_steps,
                          coord_loss_weight=coord_loss_weight,
                          diffusion_loss_weight=diffusion_loss_weight).to(device)

    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ Phys-Diff load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of Phys-Diff's parameter keys did not "
                  f"match this checkpoint's state_dict -- loaded weights are almost "
                  f"entirely random-initialized. Any ADE/ATE/CTE reported below is NOT "
                  f"a valid evaluation -- verify the checkpoint path/architecture "
                  f"before trusting it.")
    model.eval()

    if eval_sample_steps is not None and eval_sample_steps != model.sample_steps:
        print(f"  ℹ Phys-Diff: overriding sample_steps {model.sample_steps} "
              f"(checkpoint's training-time value) → {eval_sample_steps} "
              f"(--physdiff_eval_sample_steps) for this evaluation.")
        model.sample_steps = eval_sample_steps

    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_tc_diffuser(checkpoint: str, device,
                      context_dim: int = 256, tf_layer: int = 3,
                      num_steps: int = 100, beta_1: float = 1e-4, beta_T: float = 5e-2,
                      var_mode: str = "linear", dt: float = 1.0, best_k: int = 6,
                      sample_steps_stride: int = 100,
                      obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                      eval_best_k: Optional[int] = None):
    """
    Load TC-Diffuser checkpoint (train_tc_diffuser.py,
    Model/tc_diffuser_model.py::TCDiffuser). Velocity-space DDPM, shares
    PaperEncoder with every other baseline. Same model_cfg/CLI-default
    fallback pattern as load_st_trans.

    IMPORTANT: unlike the other baselines, TC-Diffuser's own evaluate()
    (see train_tc_diffuser.py) uses best-of-K sampling by ORIGINAL-repo
    design (picks, among best_k independent reverse samples, the one
    closest to ground truth), NOT a single-sample ADE -- this is not
    directly comparable to the other baselines' single-sample ADE. This
    loader does not change that; it is surfaced here as a warning and via
    the "sampling" metadata this script's caller should note when reading
    results for TC-Diffuser specifically (see also the sampling_note field
    in the returned tuple's seed string is NOT used for this -- check
    train_tc_diffuser.py's own metrics.csv "sampling" column for the
    authoritative record of this during training).

    eval_best_k: if given, OVERRIDES model.best_k after loading (the
    number of independent reverse samples generated at model.sample()
    time, the "K" in best-of-K). The value saved in the checkpoint's
    model_cfg is whatever was used during training/validation (--best_k,
    default 6, matching the original repo's own train() convention) --
    passing a larger value here (e.g. via --tcdiffuser_eval_best_k) trades
    eval time for a better best-of-K selection on the final evaluation.
    sample_steps_stride is NOT overridden here since (per
    Model/tc_diffuser_model.py's own runtime guard) it must remain either
    1 or num_steps for the 'ddpm' sampling branch to be mathematically
    valid -- changing it independently of num_steps at eval time risks
    silently reintroducing the strided-sampling bug documented in that
    model file's DiffusionTraj.sample() docstring.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", "TCDiffuser")
    if saved_type.lower() not in ("tcdiffuser", "tc-diffuser", "tc_diffuser"):
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' is not "
              f"'TCDiffuser' — verify this checkpoint path is really the TC-Diffuser one.")
    if ck.get("sampling"):
        print(f"  ℹ Checkpoint records sampling='{ck['sampling']}' from training "
              f"(best-of-K, per the original repo's own design -- see docstring above; "
              f"NOT directly comparable to the other baselines' single-sample ADE).")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = TCDiffuser(**model_cfg).to(device)
    else:
        print(f"  ⚠ TC-Diffuser checkpoint has no model_cfg — using CLI-default"
              f"-matching args (context_dim={context_dim}, tf_layer={tf_layer}, "
              f"best_k={best_k}). Only correct if trained with "
              f"train_tc_diffuser.py's own defaults.")
        model = TCDiffuser(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                            context_dim=context_dim, tf_layer=tf_layer,
                            num_steps=num_steps, beta_T=beta_T, beta_1=beta_1,
                            var_mode=var_mode, dt=dt, best_k=best_k,
                            sample_steps_stride=sample_steps_stride).to(device)

    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ TC-Diffuser load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of TC-Diffuser's parameter keys did not "
                  f"match this checkpoint's state_dict -- loaded weights are almost "
                  f"entirely random-initialized. Any ADE/ATE/CTE reported below is NOT "
                  f"a valid evaluation -- verify the checkpoint path/architecture "
                  f"before trusting it.")
    model.eval()

    if eval_best_k is not None and eval_best_k != model.best_k:
        print(f"  ℹ TC-Diffuser: overriding best_k {model.best_k} "
              f"(checkpoint's training-time value) → {eval_best_k} "
              f"(--tcdiffuser_eval_best_k) for this evaluation.")
        model.best_k = eval_best_k

    seed = _infer_seed(checkpoint, ck)
    return model, seed


@torch.no_grad()
def evaluate_one_model(model, loader, device, model_name: str,
                        seed: str = "unknown",
                        n_ensemble: int = 20,
                        ddim_steps: Optional[int] = None,
                        use_curvature_score: bool = False) -> List[Dict]:
    """
    Returns a list of PER-LEAD-TIME records:
      {"model": name, "seed": seed, "storm": storm_key, "window": idx,
       "lead_time": t, "ade": .., "ate": .., "cte": .., "obs_speed": ..}
    One record per (storm, window, lead_time) triple — matches the
    paper's Table 10 pairing granularity (140 windows x 16 lead-times =
    2240 matched pairs, i.e. paired PER FORECAST STEP, not averaged over
    the whole trajectory first).

    lead_time convention (1-indexed, 1..T; T=pred_len, e.g. 1=6h...12=72h
    when T=12): this is the SAME convention as generate_paper_report.py's
    HORIZON_LEAD_TIMES = {"6h":1,...,"72h":12}. ADE (d) has a value for
    EVERY lead_time 1..T, for every model.

    ATE/CTE use ate_cte_full (spherical forward-azimuth/haversine, from
    flow_matching_model.py) for ALL models here, by deliberate design —
    see this file's module docstring and ate_cte_full's own docstring
    for the rationale (one shared formula so FM-vs-baseline comparisons
    aren't an artifact of which formula was used). This formula needs a
    PRIOR point, so it is undefined at lead_time=1 (6h) -- None there
    for every model -- and defined for lead_time 2..T. This intentionally
    does NOT match each non-FM model's own training-time ATE/CTE (which
    uses paper_baseline_model.py's different _ate_cte_tensors formula) —
    that mismatch is expected here, not a bug.

    [FIX] An earlier version of this loop bounded lead_time by ate/cte's
    shorter range (T-1 instead of T) for ADE too, which silently dropped
    the LAST lead_time (T, i.e. 72h when T=12), and additionally
    mislabeled lead_time=1 as if it were the first step (6h) when it was
    actually the second step (12h, 0-indexed step 1) — both fixed: ADE
    now covers the full 1..T range, and lead_time=1 genuinely is the
    first predicted step.
    """
    records = []
    # is_ensemble_model: FM, MMSTN, Phys-Diff, and TC-Diffuser are all
    # generative/stochastic models (unlike the deterministic LSTM/GRU/RNN/
    # ST-Trans baselines), so they get routed away from the plain
    # num_ensemble=1 branch below. HOWEVER, only FM's sample() actually
    # accepts a runtime num_ensemble/ddim_steps override -- MMSTN, Phys-Diff,
    # and TC-Diffuser each have their OWN fixed sampling behavior configured
    # at model-construction/load time (MMSTN: single sample by original
    # design; Phys-Diff: model.sample_steps DDIM steps; TC-Diffuser:
    # model.best_k best-of-K), which load_mmstn/load_phys_diff/
    # load_tc_diffuser's eval_* args let you override BEFORE calling this
    # function (see --physdiff_eval_sample_steps / --tcdiffuser_eval_best_k
    # in main()) rather than through this loop's --n_ensemble. Passing
    # num_ensemble=n_ensemble to their .sample() calls would be silently
    # ignored (they accept **kwargs), which could mislead someone into
    # thinking --n_ensemble controls them -- so those three are called with
    # NO num_ensemble argument at all, making it explicit in the code that
    # their sampling config comes from the loaded model object, not this loop.
    is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")
    is_mmstn = isinstance(model, MMSTN)
    is_phys_diff = isinstance(model, PhysDiff)
    is_tc_diffuser = isinstance(model, TCDiffuser)
    is_ensemble_model = is_fm or is_mmstn or is_phys_diff or is_tc_diffuser

    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        obs = bl[0]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None

        try:
            if is_fm:
                # [FIX-ODE-STEPS-MISMATCH] Previously ignored any ddim_steps
                # override entirely and always used the checkpoint's own
                # self.n_inference_steps. Now CLI-configurable via --ddim_steps
                # to match evaluate_full.py's convention (None = defer to
                # checkpoint's trained value, same as before if not passed).
                # [ADD-CURVATURE-SCORE] use_curvature_score is a pure
                # inference-time re-ranking option (5th physics-score
                # component, checks whole-path turning rate vs step-0
                # direction only) -- the model's own docstring confirms it
                # needs no retraining and can be A/B tested on any existing
                # checkpoint. Previously this script always left it at the
                # sample() default (False); now CLI-configurable via
                # --use_curvature_score so it can actually be tried.
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble,
                                           ddim_steps=ddim_steps,
                                           use_curvature_score=use_curvature_score)
            elif is_mmstn or is_phys_diff or is_tc_diffuser:
                # No num_ensemble passed -- see is_ensemble_model comment
                # above. Each model's sample() uses its own pre-configured
                # sampling behavior (best_k / sample_steps / single-sample).
                pred, _, _ = model.sample(bl)
            else:
                pred, _, _ = model.sample(bl, num_ensemble=1)
        except Exception as e:
            print(f"  [{model_name}] batch {bi}: sample error: {e}")
            continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        d  = _haversine_deg(pd, gd)                  # [T, B] -- steps 0..T-1 (0=6h ... T-1=72h when T=12)
        # [DESIGN DECISION, confirmed with user] This script deliberately
        # uses ONE SHARED formula (ate_cte_full, spherical forward-azimuth/
        # haversine from flow_matching_model.py) for ALL models, including
        # ST-Trans/LSTM/GRU/RNN/MMSTN/Phys-Diff/TC-Diffuser -- even though
        # each of those imports and trains/self-evaluates with a DIFFERENT
        # formula (paper_baseline_model.py's _ate_cte_tensors, flat-plane
        # projection). This is intentional, per the original script's own
        # documented rationale (see this file's module docstring): using
        # each model's own formula would make FM-vs-baseline ATE/CTE
        # comparisons unsound, since any difference could be an artifact
        # of which formula was used rather than a genuine model quality
        # difference. A prior version of this script routed non-FM models
        # to their own _ate_cte_tensors formula instead (for a different
        # goal -- matching each model's own training-time printed numbers)
        # -- that was reverted per explicit user decision to prioritize
        # fair cross-model comparison over matching training-time numbers.
        # Consequence: ATE/CTE printed here for 7 of 8 models will NOT
        # match what each model's own train_*.py run_test_evaluation()
        # printed right after training -- that is expected and correct
        # for this script's purpose, not a bug.
        ate, cte = ate_cte_full(pd, gd)               # [T-1, B] -- ate[k] = error at step k+1 (0-indexed); undefined at step0=0 (=6h) for ALL models
        T_valid = ate.shape[0]                        # = T-1

        obs_deg = _norm_to_deg(obs[:, :, :2])
        if obs_deg.shape[0] >= 2:
            step_km = _haversine_deg(obs_deg[:-1], obs_deg[1:])
            obs_speed = step_km.mean(0) / 6.0
        else:
            obs_speed = torch.zeros(obs.shape[1], device=device)

        B = obs.shape[1]
        for b in range(B):
            if tyid_list is not None and b < len(tyid_list) and \
               isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
                info = tyid_list[b]
                storm_key = f"{info['old'][1]}_{info['old'][0]}"
            else:
                storm_key = f"UNKNOWN_batch{bi}"
            for step0 in range(T):           # step0 = 0-indexed step, 0..T-1
                lead_time = step0 + 1        # 1-indexed, 1..T (1=6h...T=72h)
                # [FIX] Bug thật đã tìm ra: trước đây vòng lặp chạy
                # `for i in range(T_valid)` (T_valid = T-1), với
                # lead_time = i+1 (i=0..T_valid-1 => lead_time=1..T_valid=1..T-1).
                # Với T=12, lead_time chỉ chạy 1..11 -- KHÔNG BAO GIỜ đạt 12.
                # generate_paper_report.py's HORIZON_LEAD_TIMES tra "72h"->12,
                # nên luôn ra n=0 ở 72h (khớp đúng hiện tượng đã quan sát).
                # Đồng thời "lead_time=1" trước đây thực chất ứng 0-indexed
                # step 1 (=12h theo evaluate_full.py's HORIZONS convention),
                # KHÔNG PHẢI 6h -- tên horizon "6h" ở nơi đọc dữ liệu cũng bị
                # lệch 1 bước so với dữ liệu thật.
                #
                # Sửa: lead_time giờ là 1-indexed THẬT trên toàn bộ T bước
                # (lead_time = step_0indexed + 1, chạy 1..T, tức 1=6h...T=72h
                # khi T=12) -- khớp đúng HORIZON_LEAD_TIMES = {"6h":1,...,
                # "72h":12} sau khi sửa ở generate_paper_report.py.
                # ADE (d) có đủ giá trị cho MỌI lead_time 1..T.
                # ATE/CTE (ate/cte) chỉ có giá trị cho lead_time 2..T (không
                # định nghĩa được ở lead_time=1/6h, vì cần bước trước đó để
                # biết hướng đi) -- ghi None thay vì bỏ hẳn record. Áp dụng
                # ĐỒNG NHẤT cho MỌI model (kể cả 7 model không phải FM),
                # vì toàn bộ script này CHỦ ĐÍCH dùng chung 1 công thức
                # ate_cte_full cho mọi model -- xem [DESIGN DECISION]
                # comment ở nơi gọi ate_cte_full phía trên.
                has_atecte = step0 >= 1      # ate/cte defined for step0=1..T-1
                ate_i = step0 - 1            # ate/cte array index when has_atecte
                records.append({
                    "model":     model_name,
                    "seed":      seed,
                    "storm":     storm_key,
                    "window":    b,
                    "lead_time": lead_time,
                    "ade":       float(d[step0, b]),
                    "ate":       float(ate[ate_i, b].abs()) if has_atecte else None,
                    "cte":       float(cte[ate_i, b].abs()) if has_atecte else None,
                    "obs_speed": float(obs_speed[b]),
                })
    return records


# [FIX-DETERMINISM] Mirrors evaluate_full.py / visual_evaluate_mode.py's
# set_seed(): this script never seeded RNGs before model.sample()'s
# K-candidate torch.randn(...) draw, so repeated runs (and comparisons
# against visualize's fixed-seed output) were not reproducible for FM.
def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42,
                   help="[FIX-DETERMINISM] RNG seed applied before any model.sample() call; "
                        "matches visual_evaluate_mode.py's fixed seed(42) convention.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--output_dir", default="eval_multi")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--ddim_steps", type=int, default=None,
                   help="[FIX-ODE-STEPS-MISMATCH] Number of ODE integration steps for "
                        "FM's model.sample(). Was previously not exposed here at all "
                        "(evaluate_one_model always used the checkpoint's own trained "
                        "n_inference_steps, silently ignoring any intended override). "
                        "Default None matches evaluate_full.py's --ddim_steps convention: "
                        "defer to the checkpoint's own value unless explicitly set.")
    p.add_argument("--use_curvature_score", action="store_true", default=False,
                   help="[ADD-CURVATURE-SCORE] Enable FM's 5th physics-score "
                        "re-ranking component (whole-path turning-rate match vs "
                        "observed storm, instead of only checking step-0 direction). "
                        "Pure inference-time change on FM's sample() -- confirmed by "
                        "the model's own docstring to need no retraining, directly "
                        "A/B-testable on any existing checkpoint. Has no effect on "
                        "non-FM models. Default off, matching sample()'s own default.")
    p.add_argument("--test_year", type=int, default=None)

    p.add_argument("--fm_checkpoints",       nargs="+", default=None,
                   help="One or more FM checkpoint paths, one per seed")
    p.add_argument("--st_trans_checkpoints", nargs="+", default=None,
                   help="One or more ST-Trans checkpoint paths, one per seed")
    p.add_argument("--lstm_checkpoints",     nargs="+", default=None,
                   help="One or more LSTM checkpoint paths, one per seed")
    p.add_argument("--gru_checkpoints",      nargs="+", default=None,
                   help="One or more GRU checkpoint paths, one per seed")
    p.add_argument("--rnn_checkpoints",      nargs="+", default=None,
                   help="One or more RNN checkpoint paths, one per seed")
    # MMSTN / Phys-Diff / TC-Diffuser checkpoint args, same multi/legacy-single convention
    p.add_argument("--mmstn_checkpoints",       nargs="+", default=None,
                   help="One or more MMSTN checkpoint paths, one per seed")
    p.add_argument("--physdiff_checkpoints",    nargs="+", default=None,
                   help="One or more Phys-Diff checkpoint paths, one per seed")
    p.add_argument("--tcdiffuser_checkpoints",  nargs="+", default=None,
                   help="One or more TC-Diffuser checkpoint paths, one per seed")
    # Backward-compat singular aliases (old single-checkpoint usage still works)
    p.add_argument("--fm_checkpoint",       default=None, help="[legacy] single checkpoint, use --fm_checkpoints instead")
    p.add_argument("--st_trans_checkpoint", default=None, help="[legacy] single checkpoint")
    p.add_argument("--lstm_checkpoint",     default=None, help="[legacy] single checkpoint")
    p.add_argument("--gru_checkpoint",      default=None, help="[legacy] single checkpoint")
    p.add_argument("--rnn_checkpoint",      default=None, help="[legacy] single checkpoint")
    p.add_argument("--mmstn_checkpoint",       default=None, help="[legacy] single checkpoint")
    p.add_argument("--physdiff_checkpoint",    default=None, help="[legacy] single checkpoint")
    p.add_argument("--tcdiffuser_checkpoint",  default=None, help="[legacy] single checkpoint")

    p.add_argument("--paper_hidden_dim", type=int, default=256)
    p.add_argument("--paper_n_layers",   type=int, default=3)
    p.add_argument("--paper_dropout",    type=float, default=0.20)
    p.add_argument("--st_d_model",        type=int, default=64)
    p.add_argument("--st_nhead",          type=int, default=4)
    p.add_argument("--st_num_enc_layers", type=int, default=1)
    p.add_argument("--st_num_dec_layers", type=int, default=3)
    p.add_argument("--st_dim_ff",         type=int, default=512)
    p.add_argument("--st_dropout",        type=float, default=0.1)

    # MMSTN architecture args (fallback only used if checkpoint lacks model_cfg;
    # defaults match train_mmstn.py's own CLI defaults)
    p.add_argument("--mmstn_embedding_dim",    type=int,   default=32)
    p.add_argument("--mmstn_encoder_h_dim_g",  type=int,   default=64)
    p.add_argument("--mmstn_decoder_h_dim_g",  type=int,   default=64)
    p.add_argument("--mmstn_encoder_h_dim_d",  type=int,   default=128)
    p.add_argument("--mmstn_mlp_dim",          type=int,   default=128)
    p.add_argument("--mmstn_noise_dim",        type=int,   default=16)
    p.add_argument("--mmstn_best_k",           type=int,   default=6)

    # Phys-Diff architecture args (fallback only used if checkpoint lacks model_cfg;
    # defaults match train_phys_diff.py's own CLI defaults)
    p.add_argument("--physdiff_d_model",       type=int,   default=128)
    p.add_argument("--physdiff_d_embedding",   type=int,   default=64)
    p.add_argument("--physdiff_enc_layers",    type=int,   default=3)
    p.add_argument("--physdiff_dec_layers",    type=int,   default=3)
    p.add_argument("--physdiff_num_timesteps", type=int,   default=1000)
    p.add_argument("--physdiff_eval_sample_steps", type=int, default=None,
                   help="Overrides model.sample_steps (DDIM reverse-step count) "
                        "AFTER loading, for a more accurate final evaluation than "
                        "whatever --sample_steps was used for cheap per-epoch "
                        "validation during training (train_phys_diff.py default: 50). "
                        "None = use the checkpoint's own trained value unchanged.")

    # TC-Diffuser architecture args (fallback only used if checkpoint lacks model_cfg;
    # defaults match train_tc_diffuser.py's own CLI defaults)
    p.add_argument("--tcdiffuser_context_dim", type=int,   default=256)
    p.add_argument("--tcdiffuser_tf_layer",    type=int,   default=3)
    p.add_argument("--tcdiffuser_num_steps",   type=int,   default=100)
    p.add_argument("--tcdiffuser_best_k",      type=int,   default=6)
    p.add_argument("--tcdiffuser_eval_best_k", type=int, default=None,
                   help="Overrides model.best_k (number of independent best-of-K "
                        "reverse samples) AFTER loading, for the final evaluation. "
                        "None = use the checkpoint's own trained value unchanged. "
                        "NOTE: TC-Diffuser's ADE here uses best-of-K sampling by "
                        "ORIGINAL-repo design, unlike the other baselines' single-"
                        "sample ADE -- not directly comparable without accounting "
                        "for this; see load_tc_diffuser's docstring.")

    args = p.parse_args()
    set_seed(args.seed)   # [FIX-DETERMINISM] must run before any model.sample() call below
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"  Loading test data...")
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root = args.dataset_root,
        obs_len      = 8,
        pred_len     = 12,
        batch_size   = 64,
        num_workers  = 2,
        test_year    = args.test_year,
        skip         = 1,
        min_ped      = 1,
        threshold    = 0.002,
    )
    _, loader = data_loader(_loader_args,
                             {"root": args.dataset_root, "type": args.split},
                             test=(args.split != "train"))
    print(f"  Data: {len(loader)} batches")

    def _collect(multi, single):
        """Merge --xxx_checkpoints (list) and legacy --xxx_checkpoint (str) into one list."""
        paths = list(multi) if multi else []
        if single and single not in paths:
            paths.append(single)
        return paths

    jobs = []  # (display_name, kind, checkpoint_path)
    for display_name, kind, multi, single in [
        ("FM",          "fm",         args.fm_checkpoints,         args.fm_checkpoint),
        ("ST-Trans",    "st_trans",   args.st_trans_checkpoints,   args.st_trans_checkpoint),
        ("LSTM",        "lstm",       args.lstm_checkpoints,       args.lstm_checkpoint),
        ("GRU",         "gru",        args.gru_checkpoints,        args.gru_checkpoint),
        ("RNN",         "rnn",        args.rnn_checkpoints,        args.rnn_checkpoint),
        ("MMSTN",       "mmstn",      args.mmstn_checkpoints,      args.mmstn_checkpoint),
        ("Phys-Diff",   "physdiff",   args.physdiff_checkpoints,   args.physdiff_checkpoint),
        ("TC-Diffuser", "tcdiffuser", args.tcdiffuser_checkpoints, args.tcdiffuser_checkpoint),
    ]:
        for ckpt_path in _collect(multi, single):
            jobs.append((display_name, kind, ckpt_path))

    if not jobs:
        print("  No checkpoints given — nothing to do.")
        return

    all_records = []
    for display_name, kind, ckpt_path in jobs:
        print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
        if kind == "fm":
            model, seed = load_fm(ckpt_path, device)
        elif kind == "st_trans":
            model, seed = load_st_trans(ckpt_path, device,
                                   d_model=args.st_d_model, nhead=args.st_nhead,
                                   num_enc_layers=args.st_num_enc_layers,
                                   num_dec_layers=args.st_num_dec_layers,
                                   dim_ff=args.st_dim_ff, dropout=args.st_dropout)
        elif kind == "mmstn":
            model, seed = load_mmstn(ckpt_path, device,
                                   embedding_dim=args.mmstn_embedding_dim,
                                   encoder_h_dim_g=args.mmstn_encoder_h_dim_g,
                                   decoder_h_dim_g=args.mmstn_decoder_h_dim_g,
                                   encoder_h_dim_d=args.mmstn_encoder_h_dim_d,
                                   mlp_dim=args.mmstn_mlp_dim,
                                   noise_dim=args.mmstn_noise_dim,
                                   best_k=args.mmstn_best_k)
        elif kind == "physdiff":
            model, seed = load_phys_diff(ckpt_path, device,
                                   d_model=args.physdiff_d_model,
                                   d_embedding=args.physdiff_d_embedding,
                                   enc_layers=args.physdiff_enc_layers,
                                   dec_layers=args.physdiff_dec_layers,
                                   num_timesteps=args.physdiff_num_timesteps,
                                   eval_sample_steps=args.physdiff_eval_sample_steps)
        elif kind == "tcdiffuser":
            model, seed = load_tc_diffuser(ckpt_path, device,
                                   context_dim=args.tcdiffuser_context_dim,
                                   tf_layer=args.tcdiffuser_tf_layer,
                                   num_steps=args.tcdiffuser_num_steps,
                                   best_k=args.tcdiffuser_best_k,
                                   eval_best_k=args.tcdiffuser_eval_best_k)
        else:
            model, seed = load_paper_baseline(ckpt_path, kind, device,
                                         hidden_dim=args.paper_hidden_dim,
                                         n_layers=args.paper_n_layers,
                                         dropout=args.paper_dropout)

        n_params = sum(pm.numel() for pm in model.parameters())
        print(f"  {display_name} (seed={seed}): {n_params:,} params")

        # [FIX-DETERMINISM-PER-MODEL] set_seed(args.seed) was previously
        # called ONCE at the very top of main(), before this loop even
        # started. That meant only the FIRST stochastic model to run
        # (whichever one happens to be first in `jobs`) got a clean,
        # reproducible RNG state -- every stochastic model after it
        # (MMSTN, Phys-Diff, TC-Diffuser, or a later FM checkpoint) inherited
        # whatever RNG state was LEFT OVER after all earlier models'
        # torch.randn(...) draws inside their own .sample() calls. Since
        # deterministic models (ST-Trans/LSTM/GRU/RNN, once in eval() mode
        # so their Dropout layers are inactive) consume no RNG, this mostly
        # hit MMSTN specifically: its results silently depended on exactly
        # how many FM checkpoints/candidates were evaluated before it in
        # THIS run -- adding, removing, or reordering --fm_checkpoints (or
        # --physdiff_checkpoints / --tcdiffuser_checkpoints) between two
        # otherwise-identical runs was enough to change MMSTN's reported
        # ADE/ATE/CTE even though its checkpoint never changed. Re-seeding
        # right before EACH model's evaluate_one_model() call makes every
        # model -- stochastic or not -- start sampling from the SAME fixed
        # RNG state every time, independent of the rest of the job list.
        set_seed(args.seed)
        recs = evaluate_one_model(model, loader, device, display_name,
                                   seed=seed, n_ensemble=args.n_ensemble,
                                   ddim_steps=args.ddim_steps,
                                   use_curvature_score=args.use_curvature_score)
        all_records.extend(recs)

        # [FIX] ate/cte là None ở lead_time=1 (6h) theo convention đã sửa
        # (xem evaluate_one_model's docstring) — np.mean crash nếu None
        # lẫn trong list. Lọc trước khi tính, giống mọi chỗ khác trong
        # generate_paper_report.py đã áp dụng cùng bộ lọc này.
        ade = np.mean([r["ade"] for r in recs if r["ade"] is not None])
        ate_vals = [r["ate"] for r in recs if r["ate"] is not None]
        cte_vals = [r["cte"] for r in recs if r["cte"] is not None]
        ate = np.mean(ate_vals) if ate_vals else float("nan")
        cte = np.mean(cte_vals) if cte_vals else float("nan")
        print(f"  {display_name} seed={seed}: n={len(recs)}  ADE={ade:.2f}  "
              f"ATE={ate:.2f}  CTE={cte:.2f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, f"multi_model_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n  Saved {len(all_records)} records → {out_path}")
    print(f"  Run generate_comparison_table.py --records {out_path} "
          f"to produce the Table-10-style significance table.")


if __name__ == "__main__":
    main()