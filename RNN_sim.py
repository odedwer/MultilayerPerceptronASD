# RNN_sim.py
import contextlib
from torch.amp import GradScaler, autocast
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RNNConfig, default_bias_means
from datasets import make_sparse_hmm, rewire_transitions, DelayedCopyHMM, make_mc_words_as_hmm
from models import LSTMWithGateBias, RNNWithGateBias
import utils
import seaborn as sns
import os, json, hashlib, time
from dataclasses import asdict
import itertools
import argparse
import sys


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_to_serializable(cfg) -> dict:
    """Convert config to a JSON-serializable dict, replacing class refs with names."""
    d = cfg.__dict__.copy()
    for k, v in d.items():
        if isinstance(v, type):
            d[k] = v.__name__
    return d


def cfg_to_ordered_json(cfg) -> str:
    """Stable JSON string for hashing (excludes run-specific fields)."""
    d = _cfg_to_serializable(cfg)
    d = {k: v for k, v in d.items() if k not in ("name", "results_dir", "run_id", "cmd", "desc")}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def fingerprint_cfg(cfg) -> str:
    return hashlib.sha256(cfg_to_ordered_json(cfg).encode()).hexdigest()[:12]


def run_dir_for_cfg(cfg) -> str:
    base = getattr(cfg, "results_dir", "runs")
    sub = f"{fingerprint_cfg(cfg)}__{cfg.name}"
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_hmm_matrices(T_train, E_train, T_test, E_test, cfg: RNNConfig, save_path=None):
    """2×2 grid: train vs test HMM transition & emission matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.heatmap(T_train, ax=axes[0, 0], cmap="viridis", vmin=0, vmax=1, cbar=False)
    axes[0, 0].set_title("A. Train: State→State Transitions")
    sns.heatmap(E_train, ax=axes[0, 1], cmap="viridis", vmin=0, vmax=1, cbar=False)
    axes[0, 1].set_title("B. Train: State→Token Emissions")
    sns.heatmap(T_test, ax=axes[1, 0], cmap="viridis", vmin=0, vmax=1, cbar=False)
    axes[1, 0].set_title("C. Test: State→State Transitions")
    sns.heatmap(E_test, ax=axes[1, 1], cmap="viridis", vmin=0, vmax=1, cbar=False)
    axes[1, 1].set_title("D. Test: State→Token Emissions")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Next state / Token")
            ax.set_ylabel("Current state")
    fig.suptitle(
        f"HMM Structure: {cfg.name}  M={cfg.M_states}  K={cfg.K_symbols}  flip={cfg.flip_prob}",
        fontsize=14,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def save_run(run_dir, cfg, model, history, final_metrics):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_cfg_to_serializable(cfg), f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "history.npz"), **history)
    np.savez_compressed(
        os.path.join(run_dir, "final_metrics.npz"),
        **{k: np.array(v) for k, v in final_metrics.items()},
    )
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))


def load_run_if_exists(cfg):
    """Return (run_dir, history, final_metrics, model_path) or Nones on cache miss."""
    rd = run_dir_for_cfg(cfg)
    hist_path = os.path.join(rd, "history.npz")
    if not os.path.exists(hist_path):
        return rd, None, None, None
    try:
        data = np.load(hist_path, allow_pickle=True)
        history = {k: data[k].tolist() for k in data.files}
        fm_path = os.path.join(rd, "final_metrics.npz")
        fm = dict(np.load(fm_path, allow_pickle=True)) if os.path.exists(fm_path) else {}
        model_path = os.path.join(rd, "model.pt")
        return rd, history, fm, model_path
    except Exception as e:
        print(f"[warn] Cache load failed: {e}")
        return rd, None, None, None


# ---------------------------------------------------------------------------
# Dimensionality
# ---------------------------------------------------------------------------

def compute_dimensionality(H, eps=1e-12):
    """
    PCA-based dimensionality metrics on hidden states.
    H: torch Tensor (N, H_dim) — already subsampled if needed.
    Returns: (pcs80, pcs90, pcs95, eff_rank)
    """
    if H.ndim != 2:
        H = H.reshape(-1, H.shape[-1])
    H = H.float()
    H_centered = H - H.mean(0, keepdim=True)
    cov = (H_centered.T @ H_centered) / max(H_centered.shape[0] - 1, 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()
        eigvals = np.clip(np.real(eigvals), eps, None)
        eigvals = np.sort(eigvals)[::-1]
        total_var = eigvals.sum()
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    if total_var < eps:
        return np.nan, np.nan, np.nan, np.nan
    cumvar = np.cumsum(eigvals) / total_var
    pcs80 = int(np.searchsorted(cumvar, 0.80)) + 1
    pcs90 = int(np.searchsorted(cumvar, 0.90)) + 1
    pcs95 = int(np.searchsorted(cumvar, 0.95)) + 1
    eff_rank = float(np.exp(-np.sum((eigvals / total_var) * np.log(eigvals / total_var + eps))))
    return pcs80, pcs90, pcs95, eff_rank


def compute_pca(H_for_pca, cfg):
    """
    Compute per-phase PCA metrics.
    H_for_pca: list of (B, T, H_dim) float32 CPU tensors.
    Returns dict of metric_name → value.
    """
    H_all = torch.cat(H_for_pca, 0).float()  # (N, T, H_dim)
    L, D = cfg.L_input, cfg.D_delay
    T_total = H_all.shape[1]
    phase_indices = {
        "in":  (0,         L),
        "dl":  (L,         L + D),
        "out": (L + D + 1, T_total),
    }
    out = {}
    for ph, (start, end) in phase_indices.items():
        if end <= start or end > T_total:
            for metric in ("pcs80", "pcs90", "pcs95", "eff"):
                out[f"{metric}_{ph}"] = np.nan
            continue
        H_phase = H_all[:, start:end, :].reshape(-1, H_all.shape[-1])
        if H_phase.shape[0] > cfg.pca_sample_size:
            idx = torch.randperm(H_phase.shape[0])[: cfg.pca_sample_size]
            H_phase = H_phase[idx]
        pcs80, pcs90, pcs95, eff_rank = compute_dimensionality(H_phase)
        out[f"pcs80_{ph}"] = pcs80
        out[f"pcs90_{ph}"] = pcs90
        out[f"pcs95_{ph}"] = pcs95
        out[f"eff_{ph}"] = eff_rank
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_probe_acc(X, Y, out_dim, device, l2=1e-3):
    """
    Ridge classifier (closed-form) returning a boolean correct-prediction tensor.
    X: (N, H_dim) float  |  Y: (N,) long  |  out_dim: number of classes
    """
    Y = Y.long()
    T = torch.zeros(X.size(0), out_dim, device=device)
    T[torch.arange(X.size(0)), Y] = 1.0
    mu = X.mean(0, keepdim=True)
    X_c = X - mu
    ones = torch.ones(X_c.size(0), 1, device=device)
    X_aug = torch.cat([X_c, ones], 1)
    Hdim = X_c.size(1)
    I = torch.eye(Hdim + 1, device=device)
    I[-1, -1] = 0.0
    try:
        W = torch.linalg.solve(X_aug.T @ X_aug + l2 * I, X_aug.T @ T)
    except Exception:
        return torch.zeros(X.size(0), dtype=torch.bool, device=device)
    preds = (X_aug @ W).argmax(-1)
    return preds == Y


def compute_hmm_probe(H_repro_all, Z_labels_all, cfg):
    """
    Linear probe: decode HMM state from repro-phase hidden states.
    Returns (avg_acc_over_positions, max_acc_over_positions).
    """
    if not H_repro_all:
        return np.nan, np.nan
    H = torch.cat(H_repro_all, 0).to(cfg.device)
    Z = torch.cat(Z_labels_all, 0).to(cfg.device)
    L, M = cfg.L_input, cfg.M_states
    N = H.shape[0]
    if N < L:
        return np.nan, np.nan
    N_seqs = N // L
    H = H[: N_seqs * L]
    Z = Z[: N_seqs * L]
    if H.size(0) > cfg.probe_sample_size:
        idx = torch.randperm(H.size(0))[: cfg.probe_sample_size]
        H, Z = H[idx], Z[idx]
        N_seqs = H.shape[0] // L
        H = H[: N_seqs * L]
        Z = Z[: N_seqs * L]
    correct = get_probe_acc(H.float(), Z, M, cfg.device)
    correct_by_seq = correct.view(N_seqs, L).float()
    correct_by_time = correct_by_seq.mean(0)
    return correct_by_time.mean().item(), correct_by_time.max().item()


def compute_token_probe(H_all, X_all, cfg):
    """
    Linear probe: decode input token identity from hidden states.
    Used for both input-phase and repro-phase probing.
    Returns accuracy (float) or nan.
    """
    if not H_all:
        return np.nan
    H = torch.cat(H_all, 0).to(cfg.device).float()
    X = torch.cat(X_all, 0).to(cfg.device)
    valid = X >= 0
    if valid.sum() < cfg.K_symbols * 2:
        return np.nan
    H, X = H[valid], X[valid]
    if H.size(0) > cfg.probe_sample_size:
        idx = torch.randperm(H.size(0))[: cfg.probe_sample_size]
        H, X = H[idx], X[idx]
    correct = get_probe_acc(H, X, cfg.K_symbols, cfg.device)
    return correct.float().mean().item()


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _lag1_autocorr_batch(H_phase: np.ndarray) -> float:
    """
    Vectorised lag-1 autocorrelation for a batch of sequences.
    H_phase: (B, T, H_dim).  Returns scalar mean over batch and hidden dim.
    """
    if H_phase.shape[1] < 2:
        return np.nan
    X = H_phase - H_phase.mean(axis=1, keepdims=True)          # (B, T, H)
    num = (X[:, :-1, :] * X[:, 1:, :]).mean(axis=1)            # (B, H)
    denom = (X[:, :-1, :].std(axis=1) * X[:, 1:, :].std(axis=1) + 1e-12)
    return float(np.nanmean(num / denom))


# ---------------------------------------------------------------------------
# Jacobian SR helper
# ---------------------------------------------------------------------------

def _estimate_jacobian_sr(model, val_loader, cfg) -> float:
    """Estimate Jacobian spectral radius using the first sample from val_loader."""
    try:
        sample_x = next(iter(val_loader))[0][:1, :2]  # (1, 2) token indices
        return utils.jacobian_spectral_radius(model, sample_x, device=cfg.device)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Data accumulation
# ---------------------------------------------------------------------------

def save_epoch_details(
    H, H_for_pca,
    H_repro_all, Z_labels_all,
    H_in_all, X_in_all,
    H_repro_token_all, X_repro_all,
    autocorr_accum,
    cfg, delay_mask, X_true,
):
    """
    Accumulate per-batch data needed for end-of-epoch metric computation.
    All hidden states stored as float32 CPU tensors.
    """
    L, D = cfg.L_input, cfg.D_delay
    H_cpu = H.detach().cpu().float()  # (B, T, H_dim)

    # ---- HMM probe: repro-phase H vs input-phase HMM states ----
    dm_cpu = delay_mask.cpu()
    if dm_cpu.any():
        H_repro_all.append(H_cpu[dm_cpu])
        Z_labels_all.append(X_true[:, :L].flatten().detach().cpu())  # reuse true tokens for alignment
    # NOTE: Z_labels_all is used by compute_hmm_probe which receives Z (HMM states), not X_true.
    # The actual HMM labels are passed as Z from the caller.

    # ---- Token probe (input phase) ----
    if L <= H_cpu.shape[1]:
        H_in = H_cpu[:, :L, :].reshape(-1, H_cpu.shape[-1])    # (B*L, H)
        X_in = X_true[:, :L].flatten().detach().cpu()           # (B*L,)
        valid = X_in >= 0
        if valid.any():
            H_in_all.append(H_in[valid])
            X_in_all.append(X_in[valid])

    # ---- Token probe (repro phase) ----
    if dm_cpu.any():
        H_repro_token_all.append(H_cpu[dm_cpu])
        X_repro_all.append(X_true[:, :L].flatten().detach().cpu())

    # ---- Per-phase lag-1 autocorrelation ----
    H_np = H_cpu.numpy()
    phase_indices = {
        "input":  (0,         L),
        "delay":  (L,         L + D),
        "output": (L + D + 1, H_np.shape[1]),
    }
    for phase, (start, end) in phase_indices.items():
        if 0 <= start < end <= H_np.shape[1]:
            autocorr_accum[phase].append(_lag1_autocorr_batch(H_np[:, start:end, :]))

    # ---- PCA accumulation (full H) ----
    H_for_pca.append(H_cpu)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.K_symbols)
    for X, Y, _, _, delay_mask in loader:
        X, Y, delay_mask = X.to(cfg.device), Y.to(cfg.device), delay_mask.to(cfg.device)
        out, _ = model(X)
        loss = criterion(out[delay_mask], Y[delay_mask])
        total_loss += loss.item() * X.size(0)
        preds = out.argmax(-1)
        total_correct += (preds[delay_mask] == Y[delay_mask]).float().sum().item()
        total += delay_mask.sum().item()
    return {"loss": total_loss / len(loader.dataset), "acc": total_correct / max(total, 1)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, test_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.K_symbols)

    use_amp = cfg.device.startswith("cuda")
    scaler = GradScaler() if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else contextlib.nullcontext()

    history = {k: [] for k in [
        # Task performance
        "train_loss", "val_loss", "test_loss",
        "train_acc",  "val_acc",  "test_acc",
        # Optimization
        "grad_norm",
        # Information probes
        "probe_hmm_avg", "probe_hmm_max",
        "probe_token_in", "probe_token_out",
        # Stability
        "jacobian_sr",
        # Geometry — effective rank per phase
        "eff_in",  "eff_dl",  "eff_out",
        # Geometry — explained variance thresholds per phase
        "pcs80_in", "pcs90_in", "pcs95_in",
        "pcs80_dl", "pcs90_dl", "pcs95_dl",
        "pcs80_out","pcs90_out","pcs95_out",
        # Temporal dynamics — lag-1 autocorrelation per phase
        "autocorr_in", "autocorr_dl", "autocorr_out",
    ]}

    for epoch in range(cfg.epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        grad_norm_accum = 0.0
        n_batches = 0

        # Accumulators reset each epoch
        H_repro_all,       Z_labels_all       = [], []
        H_in_all,          X_in_all           = [], []
        H_repro_token_all, X_repro_all        = [], []
        H_for_pca                              = []
        autocorr_accum = {"input": [], "delay": [], "output": []}

        for X, Y, Z, X_true, delay_mask in train_loader:
            X          = X.to(cfg.device)
            Y          = Y.to(cfg.device)
            Z          = Z.to(cfg.device)
            X_true     = X_true.to(cfg.device)
            delay_mask = delay_mask.to(cfg.device)

            optimizer.zero_grad()
            with amp_ctx():
                out, H = model(X)
                loss = criterion(out[delay_mask], Y[delay_mask])

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)          # unscale before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            grad_norm_accum += grad_norm.item()
            n_batches       += 1

            preds = out.argmax(-1)
            total_correct += (preds[delay_mask] == Y[delay_mask]).sum().item()
            total         += delay_mask.sum().item()
            total_loss    += loss.item() * X.size(0)

            save_epoch_details(
                H, H_for_pca,
                H_repro_all, Z_labels_all,
                H_in_all, X_in_all,
                H_repro_token_all, X_repro_all,
                autocorr_accum,
                cfg, delay_mask, X_true,
            )

        # --- Aggregate training stats ---
        train_loss     = total_loss / len(train_loader.dataset)
        train_acc      = total_correct / max(total, 1)
        mean_grad_norm = grad_norm_accum / max(n_batches, 1)

        # --- Probes ---
        # HMM probe uses Z (HMM state labels), not X_true.
        # Re-collect proper Z labels: compute_hmm_probe expects HMM-state labels.
        probe_hmm_avg, probe_hmm_max = _compute_hmm_probe_with_z(
            H_repro_all, train_loader, cfg
        )
        probe_token_in  = compute_token_probe(H_in_all,          X_in_all,    cfg)
        probe_token_out = compute_token_probe(H_repro_token_all, X_repro_all, cfg)

        # --- PCA ---
        pca = compute_pca(H_for_pca, cfg)

        # --- Autocorrelation (mean over batches) ---
        autocorr = {
            ph: float(np.nanmean(vals)) if vals else np.nan
            for ph, vals in autocorr_accum.items()
        }

        # --- Validation / test ---
        val_metrics  = evaluate(model, val_loader,  cfg)
        test_metrics = evaluate(model, test_loader, cfg)

        # --- Jacobian spectral radius ---
        sr = _estimate_jacobian_sr(model, val_loader, cfg)

        # --- Store ---
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])
        history["grad_norm"].append(mean_grad_norm)
        history["probe_hmm_avg"].append(probe_hmm_avg)
        history["probe_hmm_max"].append(probe_hmm_max)
        history["probe_token_in"].append(probe_token_in)
        history["probe_token_out"].append(probe_token_out)
        history["jacobian_sr"].append(sr)
        history["autocorr_in"].append(autocorr["input"])
        history["autocorr_dl"].append(autocorr["delay"])
        history["autocorr_out"].append(autocorr["output"])
        for k, v in pca.items():
            history[k].append(v)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss={train_loss:.3f} Acc={train_acc:.3f} | "
            f"Val={val_metrics['acc']:.3f} Test={test_metrics['acc']:.3f} | "
            f"HMM={probe_hmm_avg:.3f}/{probe_hmm_max:.3f} "
            f"TokIn={probe_token_in:.3f} TokOut={probe_token_out:.3f} | "
            f"SR={sr:.3f} GradNorm={mean_grad_norm:.3f}"
        )

    return history


def _compute_hmm_probe_with_z(H_repro_all, train_loader, cfg):
    """
    HMM probe that correctly uses Z (HMM state) labels.
    We re-accumulate Z labels from the loader's Z field (3rd element),
    but since we've already accumulated H_repro_all during training,
    we need to pair them. This helper reuses the existing compute_hmm_probe
    logic but fixes the label source.

    In practice, save_epoch_details puts X_true[:, :L] into Z_labels_all as
    a placeholder; the real HMM labels were dropped. To avoid a second pass,
    we use the token labels as a proxy for now and note that compute_hmm_probe
    needs Z from the dataset — we pass H_repro_all with a rebuilt Z accumulator.

    The simplest fix without a second loader pass: accumulate Z_repro in
    save_epoch_details (see notes). For now this function is intentionally
    left to use the accumulated HMM-state labels from the epoch's data.
    """
    # NOTE: Z_labels_all is re-built in train_model from Z (HMM states).
    # This function signature exists for symmetry; the actual computation
    # is done inside train_model via compute_hmm_probe() directly.
    return np.nan, np.nan  # placeholder — see train_model for actual call


# ---------------------------------------------------------------------------
# The actual HMM probe accumulation is fixed in train_model:
# We need Z_labels_all to come from Z, not X_true.
# Patch save_epoch_details to accept Z separately.
# ---------------------------------------------------------------------------

def _train_model_fixed(model, train_loader, val_loader, test_loader, cfg):
    """
    Corrected train_model that properly accumulates Z (HMM state) labels.
    Replaces train_model above.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.K_symbols)

    use_amp = cfg.device.startswith("cuda")
    scaler = GradScaler() if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else contextlib.nullcontext()

    history = {k: [] for k in [
        "train_loss", "val_loss", "test_loss",
        "train_acc",  "val_acc",  "test_acc",
        "grad_norm",
        "probe_hmm_avg", "probe_hmm_max",
        "probe_token_in", "probe_token_out",
        "jacobian_sr",
        "eff_in",  "eff_dl",  "eff_out",
        "pcs80_in", "pcs90_in", "pcs95_in",
        "pcs80_dl", "pcs90_dl", "pcs95_dl",
        "pcs80_out","pcs90_out","pcs95_out",
        "autocorr_in", "autocorr_dl", "autocorr_out",
    ]}

    for epoch in range(cfg.epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        grad_norm_accum = 0.0
        n_batches = 0

        H_repro_all,       Z_hmm_labels_all   = [], []   # for HMM probe
        H_in_all,          X_in_all           = [], []   # for token probe (input phase)
        H_repro_token_all, X_repro_all        = [], []   # for token probe (repro phase)
        H_for_pca                              = []
        autocorr_accum = {"input": [], "delay": [], "output": []}

        L, D = cfg.L_input, cfg.D_delay

        for X, Y, Z, X_true, delay_mask in train_loader:
            X          = X.to(cfg.device)
            Y          = Y.to(cfg.device)
            Z          = Z.to(cfg.device)
            X_true     = X_true.to(cfg.device)
            delay_mask = delay_mask.to(cfg.device)

            optimizer.zero_grad()
            with amp_ctx():
                out, H = model(X)
                loss = criterion(out[delay_mask], Y[delay_mask])

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            grad_norm_accum += grad_norm.item()
            n_batches       += 1

            preds = out.argmax(-1)
            total_correct += (preds[delay_mask] == Y[delay_mask]).sum().item()
            total         += delay_mask.sum().item()
            total_loss    += loss.item() * X.size(0)

            H_cpu    = H.detach().cpu().float()        # (B, T, H_dim)
            dm_cpu   = delay_mask.cpu()
            H_np     = H_cpu.numpy()

            # HMM probe: repro-phase H vs input-phase HMM states (Z)
            if dm_cpu.any():
                H_repro_all.append(H_cpu[dm_cpu])
                # Z[:, :L] holds HMM states for input phase; -1 elsewhere
                z_in = Z[:, :L].flatten().detach().cpu()
                Z_hmm_labels_all.append(z_in)

            # Token probe — input phase
            if L <= H_cpu.shape[1]:
                H_in_flat = H_cpu[:, :L, :].reshape(-1, H_cpu.shape[-1])
                X_in_flat = X_true[:, :L].flatten().detach().cpu()
                valid = X_in_flat >= 0
                if valid.any():
                    H_in_all.append(H_in_flat[valid])
                    X_in_all.append(X_in_flat[valid])

            # Token probe — repro phase
            if dm_cpu.any():
                H_repro_token_all.append(H_cpu[dm_cpu])
                X_repro_all.append(X_true[:, :L].flatten().detach().cpu())

            # Per-phase autocorrelation
            phase_indices = {
                "input":  (0,         L),
                "delay":  (L,         L + D),
                "output": (L + D + 1, H_np.shape[1]),
            }
            for phase, (start, end) in phase_indices.items():
                if 0 <= start < end <= H_np.shape[1]:
                    autocorr_accum[phase].append(
                        _lag1_autocorr_batch(H_np[:, start:end, :])
                    )

            H_for_pca.append(H_cpu)

        # --- Epoch-level aggregates ---
        train_loss     = total_loss / len(train_loader.dataset)
        train_acc      = total_correct / max(total, 1)
        mean_grad_norm = grad_norm_accum / max(n_batches, 1)

        probe_hmm_avg, probe_hmm_max = compute_hmm_probe(H_repro_all, Z_hmm_labels_all, cfg)
        probe_token_in  = compute_token_probe(H_in_all,          X_in_all,    cfg)
        probe_token_out = compute_token_probe(H_repro_token_all, X_repro_all, cfg)

        pca = compute_pca(H_for_pca, cfg)

        autocorr = {
            ph: float(np.nanmean(vals)) if vals else np.nan
            for ph, vals in autocorr_accum.items()
        }

        val_metrics  = evaluate(model, val_loader,  cfg)
        test_metrics = evaluate(model, test_loader, cfg)
        sr = _estimate_jacobian_sr(model, val_loader, cfg)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])
        history["grad_norm"].append(mean_grad_norm)
        history["probe_hmm_avg"].append(probe_hmm_avg)
        history["probe_hmm_max"].append(probe_hmm_max)
        history["probe_token_in"].append(probe_token_in)
        history["probe_token_out"].append(probe_token_out)
        history["jacobian_sr"].append(sr)
        history["autocorr_in"].append(autocorr["input"])
        history["autocorr_dl"].append(autocorr["delay"])
        history["autocorr_out"].append(autocorr["output"])
        for k, v in pca.items():
            history[k].append(v)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss={train_loss:.3f} Acc={train_acc:.3f} | "
            f"Val={val_metrics['acc']:.3f} Test={test_metrics['acc']:.3f} | "
            f"HMM={probe_hmm_avg:.3f}/{probe_hmm_max:.3f} "
            f"TokIn={probe_token_in:.3f} TokOut={probe_token_out:.3f} | "
            f"SR={sr:.3f} GradNorm={mean_grad_norm:.3f}"
        )

    return history


# Use the fixed version as the public API
train_model = _train_model_fixed


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(cfg):
    """
    Train (or load cached) a single model.  Returns (run_dir, history, final_metrics, model).
    """
    run_dir, history, fm, model_path = load_run_if_exists(cfg)
    if history is not None:
        print(f"[cache hit] {cfg.name} from {run_dir}")
        model = cfg.model(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(cfg.device)
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        return run_dir, history, fm, model

    print(f"[run] Training {cfg.name}  global_bias_std={cfg.global_bias_std}  on {cfg.device}")
    cfg.bias_means = default_bias_means()
    rng      = np.random.RandomState(cfg.seed)
    rng_test = np.random.RandomState(cfg.seed + 7)

    if cfg.data.lower() == "hmm":
        T, E         = make_sparse_hmm(cfg.M_states, cfg.K_symbols, cfg.s_transitions, cfg.s_emissions, rng)
        T_test, E_test = make_sparse_hmm(cfg.M_states, cfg.K_symbols, cfg.s_transitions, cfg.s_emissions, rng_test)
    elif cfg.data == "words":
        T, E         = make_mc_words_as_hmm(cfg.M_states, cfg.word_len, rng, cfg.symbol_noise_prob)
        T_test, E_test = make_mc_words_as_hmm(cfg.M_states, cfg.word_len, rng_test, cfg.symbol_noise_prob)
    else:
        raise ValueError(f"Unknown data type: {cfg.data}")

    plot_hmm_matrices(T, E, T_test, E_test, cfg, save_path=os.path.join(run_dir, "hmm_matrices.svg"))

    train = DelayedCopyHMM(cfg.n_train, T,      E,      cfg, rng)
    val   = DelayedCopyHMM(cfg.n_val,   T,      E,      cfg, rng)
    test  = DelayedCopyHMM(cfg.n_test,  T_test, E_test, cfg, rng)

    num_workers = 4 if cfg.device.startswith("cuda") else 0
    pin_memory  = cfg.device.startswith("cuda")
    kw = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True,  **kw)
    val_loader   = torch.utils.data.DataLoader(val,   batch_size=cfg.batch_size, shuffle=False, **kw)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=cfg.batch_size, shuffle=False, **kw)

    model   = cfg.model(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(cfg.device)
    history = train_model(model, train_loader, val_loader, test_loader, cfg)
    fm      = evaluate(model, test_loader, cfg)

    save_run(run_dir, cfg, model, history, fm)
    print(f"[saved] {run_dir}")
    return run_dir, history, fm, model


# ---------------------------------------------------------------------------
# Comparison runner + plotting
# ---------------------------------------------------------------------------

def run_comparison(cfg_def, cfg_low, cfg_high):
    """
    Train Low-var and High-var models on the same HMM data, cache results,
    and produce a 3×3 diagnostic summary plot.
    """
    colors = {"Low": utils.NT_COLOR, "High": utils.ASD_COLOR}
    PHASE_DEFS = [
        ("in",  "Input",  "-"),
        ("dl",  "Delay",  "--"),
        ("out", "Output", ":"),
    ]

    plt.rcParams.update({
        "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.frameon": False, "legend.fontsize": 9,
    })

    # --- Build shared dataset ---
    rng      = np.random.RandomState(cfg_def.seed)
    rng_test = np.random.RandomState(cfg_def.seed + 7)

    if cfg_def.data.lower() == "hmm":
        T, E         = make_sparse_hmm(cfg_def.M_states, cfg_def.K_symbols,
                                       cfg_def.s_transitions, cfg_def.s_emissions, rng)
        T_test, E_test = make_sparse_hmm(cfg_def.M_states, cfg_def.K_symbols,
                                         cfg_def.s_transitions, cfg_def.s_emissions, rng_test)
    elif cfg_def.data == "words":
        T, E         = make_mc_words_as_hmm(cfg_def.M_states, cfg_def.word_len, rng,
                                             cfg_def.symbol_noise_prob)
        T_test, E_test = make_mc_words_as_hmm(cfg_def.M_states, cfg_def.word_len, rng_test,
                                               cfg_def.symbol_noise_prob)
    else:
        raise ValueError(f"Unknown data type: {cfg_def.data}")

    train = DelayedCopyHMM(cfg_def.n_train, T,      E,      cfg_def, rng)
    val   = DelayedCopyHMM(cfg_def.n_val,   T,      E,      cfg_def, rng)
    test  = DelayedCopyHMM(cfg_def.n_test,  T_test, E_test, cfg_def, rng)

    num_workers = 4 if cfg_def.device.startswith("cuda") else 0
    pin_memory  = cfg_def.device.startswith("cuda")
    kw = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg_def.batch_size, shuffle=True,  **kw)
    val_loader   = torch.utils.data.DataLoader(val,   batch_size=cfg_def.batch_size, shuffle=False, **kw)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=cfg_def.batch_size, shuffle=False, **kw)

    # --- Train or load each model ---
    results, model_paths = {}, {}
    for label, cfg in [("Low", cfg_low), ("High", cfg_high)]:
        cfg_hash = hashlib.sha1(cfg_to_ordered_json(cfg).encode()).hexdigest()[:10]
        run_dir  = os.path.join("runs", f"{label}_{cfg_hash}")
        model_paths[label] = run_dir
        os.makedirs(run_dir, exist_ok=True)

        hist_path = os.path.join(run_dir, "history.npz")
        if os.path.exists(hist_path):
            print(f"\n[cache] Reusing {label} ({cfg_hash})")
            data = np.load(hist_path, allow_pickle=True)
            results[label] = {k: data[k].tolist() for k in data.files}
            continue

        model   = cfg.model(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(cfg.device)
        history = train_model(model, train_loader, val_loader, test_loader, cfg)
        results[label] = history

        np.savez_compressed(hist_path, **history)
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(_cfg_to_serializable(cfg), f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- 3×3 Summary plot ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    epochs = np.arange(1, cfg_def.epochs + 1)

    def get(label, key):
        """Safely retrieve a metric list; return list of NaNs on missing key."""
        return results[label].get(key, [np.nan] * cfg_def.epochs)

    # A — Loss
    ax = axes[0, 0]
    for label in results:
        ax.plot(epochs, get(label, "train_loss"), "-",  color=colors[label], label=f"{label} Train")
        ax.plot(epochs, get(label, "val_loss"),   "--", color=colors[label], label=f"{label} Val",   alpha=0.7)
    ax.set_title("A. Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy")
    ax.legend(fontsize=8)

    # B — Accuracy
    ax = axes[0, 1]
    for label in results:
        ax.plot(epochs, get(label, "train_acc"), "-",  color=colors[label], label=f"{label} Train")
        ax.plot(epochs, get(label, "val_acc"),   "--", color=colors[label], label=f"{label} Val",  alpha=0.7)
        ax.plot(epochs, get(label, "test_acc"),  ":",  color=colors[label], label=f"{label} Test", alpha=0.7)
    ax.set_title("B. Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

    # C — Gradient norm
    ax = axes[0, 2]
    for label in results:
        ax.plot(epochs, get(label, "grad_norm"), color=colors[label], label=label)
    ax.set_title("C. Gradient Norm")
    ax.set_xlabel("Epoch"); ax.set_ylabel("‖∇‖ (pre-clip)")
    ax.legend()

    # D — HMM probe
    ax = axes[1, 0]
    for label in results:
        ax.plot(epochs, get(label, "probe_hmm_avg"), "-",  color=colors[label], lw=2, label=f"{label} Avg")
        ax.plot(epochs, get(label, "probe_hmm_max"), ":",  color=colors[label], lw=1.5, label=f"{label} Max")
    ax.set_title("D. HMM State Probe (Repro Phase)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

    # E — Token probe
    ax = axes[1, 1]
    for label in results:
        ax.plot(epochs, get(label, "probe_token_in"),  "-",  color=colors[label], lw=2, label=f"{label} Input")
        ax.plot(epochs, get(label, "probe_token_out"), "--", color=colors[label], lw=1.5, label=f"{label} Repro")
    ax.set_title("E. Token Identity Probe")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

    # F — Jacobian spectral radius
    ax = axes[1, 2]
    for label in results:
        ax.plot(epochs, get(label, "jacobian_sr"), color=colors[label], label=label)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5, label="ρ = 1")
    ax.set_title("F. Jacobian Spectral Radius  ρ(∂h/∂h)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("ρ")
    ax.legend(fontsize=8)

    # G — Effective rank per phase
    ax = axes[2, 0]
    for label in results:
        for ph, phase_name, ls in PHASE_DEFS:
            ax.plot(epochs, get(label, f"eff_{ph}"),
                    ls=ls, color=colors[label], lw=1.5,
                    label=f"{label} {phase_name}")
    ax.set_title("G. Effective Rank (per phase)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Eff. rank")
    _deduplicated_legend(ax, ncol=2, fontsize=7)

    # H — PCs for 95 % explained variance
    ax = axes[2, 1]
    for label in results:
        for ph, phase_name, ls in PHASE_DEFS:
            ax.plot(epochs, get(label, f"pcs95_{ph}"),
                    ls=ls, color=colors[label], lw=1.5,
                    label=f"{label} {phase_name}")
    ax.set_title("H. PCs for 95 % Explained Variance")
    ax.set_xlabel("Epoch"); ax.set_ylabel("# PCs")
    _deduplicated_legend(ax, ncol=2, fontsize=7)

    # I — Lag-1 autocorrelation
    ax = axes[2, 2]
    for label in results:
        for ph, phase_name, ls in PHASE_DEFS:
            ax.plot(epochs, get(label, f"autocorr_{ph}"),
                    ls=ls, color=colors[label], lw=1.5,
                    label=f"{label} {phase_name}")
    ax.set_title("I. Lag-1 Autocorrelation (per phase)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("r(1)")
    _deduplicated_legend(ax, ncol=2, fontsize=7)

    for ax_row in axes:
        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Low Var (teal) vs High Var (red) — M={cfg_def.M_states} K={cfg_def.K_symbols} "
        f"L={cfg_def.L_input} D={cfg_def.D_delay}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    # --- Save ---
    timestamp   = int(time.time())
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, f"summary_grid_{timestamp}.svg")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.show()

    plot_hmm_matrices(T, E, T_test, E_test, cfg_def,
                      save_path=os.path.join(results_dir, f"hmm_matrices_{timestamp}.svg"))
    cfg_low.dump(os.path.join(results_dir, f"cfg_low_{timestamp}.json"))
    cfg_high.dump(os.path.join(results_dir, f"cfg_high_{timestamp}.json"))

    index = {
        "summary": summary_path,
        "runs": [{"id": lbl, "path": p, "color": colors[lbl]} for lbl, p in model_paths.items()],
    }
    with open(os.path.join(results_dir, f"index_{timestamp}.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[saved] {summary_path}")
    return results


def _deduplicated_legend(ax, **kwargs):
    """Add a legend without duplicate labels (one per unique label)."""
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    default_hidden_dim = 128
    parser = argparse.ArgumentParser(description="RNN HMM Delayed Copy Experiment")
    parser.add_argument("--data",              type=str,   default="hmm")
    parser.add_argument("--M_states",          type=int,   default=3)
    parser.add_argument("--K_symbols",         type=int,   default=9)
    parser.add_argument("--word_len",          type=int,   default=3)
    parser.add_argument("--symbol_noise_prob", type=float, default=0.0)
    parser.add_argument("--L_input",           type=int,   default=10)
    parser.add_argument("--D_delay",           type=int,   default=10)
    parser.add_argument("--s_transitions",     type=int,   default=2)
    parser.add_argument("--s_emissions",       type=int,   default=3)
    parser.add_argument("--model_type",        type=str,   default="LSTM")
    parser.add_argument("--dr_gates",          type=str,   default="input,forget,cell,output")
    parser.add_argument("--low_dr_std",        type=float, default=1.0)
    parser.add_argument("--high_dr_std",       type=float, default=10.0)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--hidden_size",       type=int,   default=default_hidden_dim)
    parser.add_argument("--n_train",           type=int,   default=15000)
    parser.add_argument("--desc",              type=str,   default="")
    parser.add_argument("--n_seeds",           type=int,   default=5)
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--force_seed",        type=int,   default=-1)
    args = parser.parse_args()
    print(args)

    torch.set_float32_matmul_precision("medium")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    seeds = [2, 16, 83, 7, 99, 42][: args.n_seeds]
    if args.force_seed >= 0:
        seeds = [args.force_seed] * args.n_seeds

    model_cls = LSTMWithGateBias if args.model_type.lower() == "lstm" else RNNWithGateBias

    param_combinations = list(itertools.product(
        [args.M_states], [args.K_symbols], [args.L_input], [args.D_delay],
        [args.s_transitions], [args.s_emissions], [0.0], [False], seeds,
    ))

    print(f"Running {len(param_combinations)} combinations …\n")
    for M, K, L, D, s_t, s_e, flip, fr, seed in param_combinations:
        cfg_def = RNNConfig(
            data=args.data,
            word_len=args.word_len,
            symbol_noise_prob=args.symbol_noise_prob,
            M_states=M, K_symbols=K, L_input=L, D_delay=D,
            s_transitions=s_t, s_emissions=s_e,
            flip_prob=flip, ood_rewire_frac=1.0,
            name="default_bias",
            epochs=args.epochs,
            freeze_all_biases=fr,
            hidden_size=args.hidden_size,
            model=model_cls,
            lr=args.lr,
            seed=seed,
            gates_dr=tuple(args.dr_gates.split(",")),
            n_train=args.n_train,
            desc=args.desc,
            cmd=" ".join(sys.argv),
        )
        cfg_low  = cfg_def.replace(global_bias_std=args.low_dr_std,  name="low_bias_std")
        cfg_high = cfg_def.replace(global_bias_std=args.high_dr_std, name="high_bias_std")
        run_comparison(cfg_def, cfg_low, cfg_high)
