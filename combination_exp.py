#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colored-Shapes compositional task under Low vs High bias-variance initializations in MLPs.

Key features:
- Synthetic task with factors: x, y (nuisance), size s, one-hot COLOR(3), SHAPE(3).
- Train with selected color×shape pairs held out -> OOD-CG.
- Threshold shift at test -> OOD-TH.
- Compare LOW vs HIGH bias initialization variance.
- Linear probes for COLOR, SHAPE, R1, R2, y.
- 3D joint embeddings (shared axes) per layer:
    * Saved as interactive HTML.
    * Also saved into a single multi-page PDF report with split views:
      LOW (left) and HIGH (right), both with 'o' markers, colored by (color×shape) combo.
- Optional GPU classical MDS (set use_mds=True).
- All Matplotlib figures are appended to one PDF: figures/mlp_composition/mlp_report.pdf
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, pairwise_distances
from scipy.special import expit
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")  # headless backend for saving
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

# Optional Plotly for interactive HTML
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ----------------------------
# Global styling / constants
# ----------------------------
LOW_MARKER = 'o'
HIGH_MARKER = 'x'
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]  # 9 colors for 3x3 combos

COLORS = ["red", "green", "blue"]         # 3
SHAPES = ["circle", "square", "triangle"] # 3
# Marker/symbol mapping by SHAPE index
SHAPE_MARKERS_MPL   = ['o', 's', '^']              # circle, square, triangle (matplotlib)
SHAPE_SYMBOLS_PLOTLY = ['circle', 'square', 'diamond']  # (plotly)

def one_hot(idx, n):
    v = np.zeros(n, dtype=np.float32); v[idx] = 1.0; return v

# ----------------------------
# Data generation
# ----------------------------
def generate_samples(
    n,
    tau1=0.7, tau2=0.3,
    holdout_pairs=None,
    x_range=(-1, 1),
    size_range=(0.0, 1.0),
    rng=None,
    input_noise_std=0.0,
    label_flip_prob=0.0
):
    """
    Generate samples with fields: [x, y, s, onehot_color(3), onehot_shape(3)] and label y.
    Rule:
      R1: (shape in {square,triangle}) XOR (color in {red,green})
      R2: if x>=0 -> s > tau1  else s < tau2
      y  = R1 AND R2
    Optionally add gaussian noise to (x,y,s) and uniform label flips.
    """
    if rng is None:
        rng = np.random.default_rng()

    xs, ys_cont, sizes, colors_oh, shapes_oh = [], [], [], [], []
    R1_list, R2_list, Y_list, pairs = [], [], [], []

    while len(xs) < n:
        x = rng.uniform(*x_range)
        y = rng.uniform(*x_range)   # nuisance factor
        s = rng.uniform(*size_range)

        color_idx = rng.integers(0, len(COLORS))
        shape_idx = rng.integers(0, len(SHAPES))
        pair = (color_idx, shape_idx)
        if holdout_pairs and pair in holdout_pairs:
            continue

        shape_is_sq_or_tr = 1 if (shape_idx in [1, 2]) else 0
        color_is_r_or_g   = 1 if (color_idx in [0, 1]) else 0
        R1 = shape_is_sq_or_tr ^ color_is_r_or_g
        R2 = 1 if ((x >= 0 and s > tau1) or (x < 0 and s < tau2)) else 0
        y_label = 1 if (R1 and R2) else 0

        if input_noise_std > 0.0:
            x += rng.normal(0, input_noise_std)
            y += rng.normal(0, input_noise_std)
            s = np.clip(s + rng.normal(0, input_noise_std), 0.0, 1.0)

        if label_flip_prob > 0.0 and rng.random() < label_flip_prob:
            y_label = 1 - y_label

        xs.append(np.float32(x)); ys_cont.append(np.float32(y)); sizes.append(np.float32(s))
        colors_oh.append(one_hot(color_idx, 3)); shapes_oh.append(one_hot(shape_idx, 3))
        R1_list.append(R1); R2_list.append(R2); Y_list.append(y_label); pairs.append(pair)

    X = np.column_stack([np.array(xs, dtype=np.float32),
                         np.array(ys_cont, dtype=np.float32),
                         np.array(sizes, dtype=np.float32)])
    X = np.concatenate([X, np.stack(colors_oh), np.stack(shapes_oh)], axis=1)  # [n, 9]
    return X, np.array(Y_list, dtype=np.float32), np.array(R1_list, dtype=np.float32), np.array(R2_list, dtype=np.float32), np.array(pairs, dtype=np.int64)

def build_canonical_probe(n_per_combo=25, tau1=0.7, tau2=0.3, rng=None):
    """
    Balanced probe grid: 3 colors × 3 shapes × 2 x-bins × 3 size-bins.
    """
    if rng is None:
        rng = np.random.default_rng()
    xs, ys, ss, cols, shps, R1s, R2s, Ys, combos = [], [], [], [], [], [], [], [], []
    x_bins = [(-1.0, 0.0), (0.0, 1.0)]
    s_bins = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]
    for c in range(3):
        for h in range(3):
            for xb in x_bins:
                for sb in s_bins:
                    for _ in range(n_per_combo):
                        x = rng.uniform(*xb); y = rng.uniform(-1.0, 1.0); s = rng.uniform(*sb)
                        shape_is_sq_or_tr = 1 if (h in [1,2]) else 0
                        color_is_r_or_g   = 1 if (c in [0,1]) else 0
                        R1 = shape_is_sq_or_tr ^ color_is_r_or_g
                        R2 = 1 if ((x >= 0 and s > tau1) or (x < 0 and s < tau2)) else 0
                        Y  = 1 if (R1 and R2) else 0
                        xs.append(x); ys.append(y); ss.append(s)
                        cols.append(one_hot(c, 3)); shps.append(one_hot(h, 3))
                        R1s.append(R1); R2s.append(R2); Ys.append(Y); combos.append((c, h, xb, sb))
    X = np.column_stack([np.array(xs, dtype=np.float32),
                         np.array(ys, dtype=np.float32),
                         np.array(ss, dtype=np.float32)])
    X = np.concatenate([X, np.stack(cols), np.stack(shps)], axis=1)
    return {"X": X, "R1": np.array(R1s, dtype=np.float32), "R2": np.array(R2s, dtype=np.float32),
            "y": np.array(Ys, dtype=np.float32), "combos": combos}

def make_splits(
    n_train=20000, n_val=4000, n_test=8000,
    holdout_pairs=[(1,2), (2,1)],  # (color_idx, shape_idx)
    tau1=0.7, tau2=0.3,
    tau1_shift=0.8, tau2_shift=0.2,
    seed=123,
    input_noise_std=0.0,
    label_flip_prob=0.0
):
    rng = np.random.default_rng(seed)
    Xtr, ytr, R1tr, R2tr, _ = generate_samples(n_train, tau1, tau2, holdout_pairs, rng=rng,
                                               input_noise_std=input_noise_std, label_flip_prob=label_flip_prob)
    Xva, yva, R1va, R2va, _ = generate_samples(n_val, tau1, tau2, holdout_pairs, rng=rng,
                                               input_noise_std=input_noise_std, label_flip_prob=label_flip_prob)
    Xte, yte, R1te, R2te, pairs_te = generate_samples(n_test, tau1, tau2, holdout_pairs=None, rng=rng,
                                                      input_noise_std=input_noise_std, label_flip_prob=label_flip_prob)
    mask_cg = np.array([tuple(p) in holdout_pairs for p in pairs_te])
    X_cg, y_cg, R1_cg, R2_cg = Xte[mask_cg], yte[mask_cg], R1te[mask_cg], R2te[mask_cg]
    if len(X_cg) < 1000:
        need = max(1000, n_test) - len(X_cg)
        X_lst, y_lst, R1_lst, R2_lst = [X_cg], [y_cg], [R1_cg], [R2_cg]
        while need > 0:
            Xtmp, ytmp, R1tmp, R2tmp, ptmp = generate_samples(min(need, 5000), tau1, tau2, holdout_pairs=None, rng=rng,
                                                               input_noise_std=input_noise_std, label_flip_prob=label_flip_prob)
            keep = np.array([tuple(p) in holdout_pairs for p in ptmp])
            X_lst.append(Xtmp[keep]); y_lst.append(ytmp[keep]); R1_lst.append(R1tmp[keep]); R2_lst.append(R2tmp[keep])
            need -= keep.sum()
        X_cg = np.vstack(X_lst); y_cg = np.concatenate(y_lst); R1_cg = np.concatenate(R1_lst); R2_cg = np.concatenate(R2_lst)

    # OOD threshold-shift (kept clean)
    Xth, yth, R1th, R2th, _ = generate_samples(n_test, tau1_shift, tau2_shift, holdout_pairs=None, rng=rng,
                                               input_noise_std=0.0, label_flip_prob=0.0)

    probe = build_canonical_probe(rng=rng)
    return {
        "train": (Xtr, ytr, R1tr, R2tr),
        "val":   (Xva, yva, R1va, R2va),
        "test":  (Xte, yte, R1te, R2te),
        "ood_cg": (X_cg, y_cg, R1_cg, R2_cg),
        "ood_th": (Xth, yth, R1th, R2th),
        "probe": probe
    }

# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=9, hidden=256, depth=4, out_dim=1,
                 nonlin="tanh", layernorm=False, bias_std=0.01, weight_std=None, seed=123):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        dims = [in_dim] + [hidden]*depth + [out_dim]
        for i in range(len(dims)-1):
            lin = nn.Linear(dims[i], dims[i+1], bias=True)
            if weight_std is not None:
                nn.init.normal_(lin.weight, mean=0.0, std=weight_std, generator=g)
            else:
                nn.init.kaiming_normal_(lin.weight, nonlinearity='tanh' if nonlin=='tanh' else 'relu', generator=g)
            nn.init.normal_(lin.bias, mean=0.0, std=bias_std, generator=g)
            layers.append(lin)
            if i < len(dims)-2:
                if nonlin == "tanh":
                    layers.append(nn.Tanh())
                elif nonlin == "gelu":
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def set_bias_trainable(model: nn.Module, trainable: bool):
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.requires_grad = trainable

# ----------------------------
# Train / eval
# ----------------------------
def torchify(X, y):
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.reshape(-1, 1).astype(np.float32))

def make_loader(X, y, batch=256, shuffle=True):
    X_t, y_t = torchify(X, y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=shuffle)

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, freeze_bias_epochs=0, device="cpu", desc=""):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    if freeze_bias_epochs > 0:
        set_bias_trainable(model, False)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if ep == freeze_bias_epochs and freeze_bias_epochs > 0:
            set_bias_trainable(model, True)
        tl = 0.0
        for xb, yb in tqdm(train_loader, desc=f"{desc} Epoch {ep+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb); loss = crit(logits, yb)
            loss.backward(); opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_loader.dataset)
        vl, vacc = evaluate_model(model, val_loader, device=device, crit=crit)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(vacc)
    t1 = time.time()
    print(f"[{desc}] Training finished in {t1 - t0:.1f}s")
    return hist

@torch.no_grad()
def evaluate_model(model, loader, device="cpu", crit=None):
    model.eval()
    tot_loss, y_true, y_pred = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if crit is not None:
            loss = crit(logits, yb)
            tot_loss += loss.item() * xb.size(0)
        probs = torch.sigmoid(logits)
        yhat = (probs >= 0.5).float()
        y_true.append(yb.cpu().numpy()); y_pred.append(yhat.cpu().numpy())
    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred).ravel()
    acc = (y_true == y_pred).mean()
    return (tot_loss / len(loader.dataset), acc) if crit is not None else acc

# ----------------------------
# Activations & Probes
# ----------------------------
def get_layer_modules(model):
    return [m for m in model.net if isinstance(m, (nn.Tanh, nn.ReLU, nn.GELU))]

def collect_activations(model, X, device="cpu"):
    model.eval(); X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    hooks, acts = [], []
    def hook_fn(_m, _i, out): acts.append(out.detach().cpu().numpy())
    for m in get_layer_modules(model):
        hooks.append(m.register_forward_hook(hook_fn))
    with torch.no_grad():
        _ = model(X_t)
    for h in hooks: h.remove()
    return acts

def fit_probe_classification(Z, y, C=1.0, max_iter=1000):
    Zs = StandardScaler().fit_transform(Z)
    clf = LogisticRegression(C=C, max_iter=max_iter)  # multinomial default (no FutureWarning)
    clf.fit(Zs, y)
    return accuracy_score(y, clf.predict(Zs))

def build_feature_labels(X, R1, R2, y):
    color = X[:, 3:6].argmax(axis=1)
    shape = X[:, 6:9].argmax(axis=1)
    return color, shape, R1.astype(int), R2.astype(int), y.astype(int)

def make_combo_labels(color_idx, shape_idx):
    """
    Map each sample to a unique combo id in [0..8] for 3x3 color×shape.
    Returns:
      combo_labels: int array shape [n]
      combo_names:  list of length 9 in canonical order: color-major then shape
    """
    combo = (color_idx * 3 + shape_idx).astype(int)
    combo_names = [f"{c}×{s}" for c in COLORS for s in SHAPES]
    return combo, combo_names

# ----------------------------
# GPU classical MDS helpers (optional)
# ----------------------------
def _to_device(x_np, device):
    return torch.as_tensor(x_np, dtype=torch.float32, device=device)

@torch.no_grad()
def pairwise_sqdist_torch(X, metric="euclidean", device="cuda"):
    """
    X: [n, d] -> returns squared pairwise distances D2 [n,n] on device.
    metric: "euclidean" | "correlation"
    """
    X = _to_device(X, device)
    n, d = X.shape

    if metric == "euclidean":
        xx = (X * X).sum(dim=1, keepdim=True)
        D2 = xx + xx.t() - 2.0 * (X @ X.t())
        D2.clamp_(min=0.0)
        return D2
    elif metric == "correlation":
        Xc = X - X.mean(dim=1, keepdim=True)
        std = Xc.std(dim=1, keepdim=True).clamp_min(1e-6)
        Z = Xc / std
        C = (Z @ Z.t()) / (d - 1)
        D2 = 2.0 * (1.0 - C).clamp(min=0.0)
        return D2
    else:
        raise ValueError(f"Unsupported metric: {metric}")

@torch.no_grad()
def classical_mds_torch_from_D2(D2, k=3, device="cuda"):
    """
    Classical MDS:
      B = -1/2 J D2 J
      Y = U_k diag(sqrt(lambda_k))
    """
    n = D2.shape[0]
    I = torch.eye(n, device=device)
    J = I - (1.0 / n) * torch.ones((n, n), device=device)
    B = -0.5 * (J @ D2 @ J)
    evals, evecs = torch.linalg.eigh(B)  # ascending
    evals, evecs = evals.flip(0), evecs.flip(1)  # descending
    evals_k = evals[:k].clamp(min=0)
    evecs_k = evecs[:, :k]
    Y = evecs_k * torch.sqrt(evals_k).clamp_min(1e-12)
    return Y.detach().cpu().numpy()

# ----------------------------
# Geometry / embedding
# ----------------------------
def joint_embedding(
    acts_low, acts_high,
    method="pca",                # "pca" | "mds" | "mds_gpu"
    n_components=3,
    random_state=0,
    mds_metric="euclidean",      # for MDS variants
    device=None                  # "cuda" | "cpu" | None -> auto
):
    """
    Shared-axes embedding of concatenated activations (fit on concat, then split).
    """
    A = np.vstack([acts_low, acts_high])
    A = StandardScaler().fit_transform(A)

    if method == "pca":
        emb = PCA(n_components=n_components, random_state=random_state).fit_transform(A)
    elif method == "mds":
        # CPU metric MDS via classical double-centering on CPU distances
        D = pairwise_distances(A, metric=("euclidean" if mds_metric == "euclidean" else "correlation"))
        # Classical MDS (eigendecomp) on CPU:
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D**2) @ H
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1][:n_components]
        w_k = np.clip(w[idx], 0, None)
        emb = V[:, idx] * np.sqrt(w_k)
    elif method == "mds_gpu":
        dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        D2 = pairwise_sqdist_torch(A, metric=mds_metric, device=dev)
        emb = classical_mds_torch_from_D2(D2, k=n_components, device=dev)
    else:
        raise ValueError(f"Unknown method: {method}")

    nL = acts_low.shape[0]
    return emb[:nL], emb[nL:]

# ----------------------------
# Calibration & Psychometrics
# ----------------------------
def expected_calibration_error(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if m.sum() == 0: continue
        acc  = (labels[m] == (probs[m] >= 0.5)).mean()
        conf = probs[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

def psychometric_size_sweep(model, color_idx=0, shape_idx=1, x_sign=+1, tau1=0.7, tau2=0.3, device="cpu", n=61):
    x = 0.1 * x_sign; y = 0.0
    ss = np.linspace(0, 1, n, dtype=np.float32)
    X = []
    for s in ss:
        v = np.array([x, y, s], dtype=np.float32)
        v = np.concatenate([v, one_hot(color_idx,3), one_hot(shape_idx,3)]).astype(np.float32)
        X.append(v)
    X = np.stack(X)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).cpu().numpy().ravel()
        probs = expit(logits)
    def f(s, k, s0): return expit(k*(s - s0))
    try:
        (k, s0), _ = curve_fit(f, ss, probs, p0=(10.0, 0.5), maxfev=5000)
    except Exception:
        k, s0 = np.nan, np.nan
    return ss, probs, k, s0

# ----------------------------
# Plot helpers -> single PDF
# ----------------------------
def fig_to_pdf(fig, pdf):
    pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_training(hist_low, hist_high):
    fig, ax = plt.subplots(1,3, figsize=(12,3.5))
    ax[0].plot(hist_low["train_loss"], marker=LOW_MARKER, label="LowBias")
    ax[0].plot(hist_high["train_loss"], marker=HIGH_MARKER, label="HighBias")
    ax[0].set_title("Train Loss"); ax[0].legend()

    ax[1].plot(hist_low["val_loss"], marker=LOW_MARKER, label="LowBias")
    ax[1].plot(hist_high["val_loss"], marker=HIGH_MARKER, label="HighBias")
    ax[1].set_title("Val Loss"); ax[1].legend()

    ax[2].plot(hist_low["val_acc"], marker=LOW_MARKER, label="LowBias")
    ax[2].plot(hist_high["val_acc"], marker=HIGH_MARKER, label="HighBias")
    ax[2].set_title("Val Acc"); ax[2].legend()
    plt.tight_layout()
    return fig

def plot_probe_curves_combined(layers, low_scores, high_scores, title="Decoding accuracy"):
    targets = ["color", "shape", "R1", "R2", "y"]
    colors = {t: PALETTE[i] for i, t in enumerate(targets)}
    x = np.asarray(layers, dtype=float); x_hi = x + 0.06  # offset High markers a bit

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    fig.suptitle(title)

    def _plot(ax, group, emphasize=False):
        for t in group:
            lw = 1.6 if emphasize else 1.2
            ms = 6 if emphasize else 5
            ax.plot(x,    low_scores[t],  marker=LOW_MARKER,  linestyle='-',  color=colors[t],
                    alpha=0.95, linewidth=lw, markersize=ms,  label=f"{t} (Low)")
            ax.plot(x_hi, high_scores[t], marker=HIGH_MARKER, linestyle='--', color=colors[t],
                    alpha=0.95, linewidth=lw, markersize=ms,  label=f"{t} (High)")
        ax.set_ylabel("Decoding accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.7); ax.legend(ncols=3, fontsize=8)

    _plot(axes[0], ["color", "shape"], emphasize=True); axes[0].set_title("Discrete features")
    _plot(axes[1], ["R1", "R2", "y"], emphasize=False); axes[1].set_title("Rules and final label"); axes[1].set_xlabel("Layer")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_joint_split3d_pdf(emb_low, emb_high, labels, label_name, layer_idx, category_names=None):
    """
    Static PDF: 1x2 3D subplots. Left: Low, Right: High.
    - Marker shape encodes SHAPE (o,s,^)
    - Color encodes COLOR (red, green, blue)
    - Legend: one entry per combo, with its marker+color
    labels: combo id in [0..8] where combo = color_idx*3 + shape_idx
    """
    uniq = np.unique(labels)

    # If not supplied, build combo names in color-major order
    if category_names is None:
        category_names = [f"{c}×{s}" for c in COLORS for s in SHAPES]

    # Axis ranges synchronized
    all_pts = np.vstack([emb_low, emb_high])
    xyz_min = all_pts.min(axis=0); xyz_max = all_pts.max(axis=0)

    fig = plt.figure(figsize=(12, 5.2))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Helper: plot one panel with per-combo style
    def _scatter_panel(ax, emb):
        for v in uniq:
            color_idx = int(v) // 3
            shape_idx = int(v) % 3
            m = (labels == v)
            col = COLORS[color_idx]
            mk  = SHAPE_MARKERS_MPL[shape_idx]
            ax.scatter(emb[m,0], emb[m,1], emb[m,2],
                       s=16, marker=mk, color=col, alpha=0.95)

    # LOW (left) and HIGH (right)
    _scatter_panel(ax1, emb_low)
    ax1.set_title(f"Layer {layer_idx} • LOW • by {label_name}")
    ax1.set_xlim(xyz_min[0], xyz_max[0]); ax1.set_ylim(xyz_min[1], xyz_max[1]); ax1.set_zlim(xyz_min[2], xyz_max[2])
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_zlabel('PC3')

    _scatter_panel(ax2, emb_high)
    ax2.set_title(f"Layer {layer_idx} • HIGH • by {label_name}")
    ax2.set_xlim(xyz_min[0], xyz_max[0]); ax2.set_ylim(xyz_min[1], xyz_max[1]); ax2.set_zlim(xyz_min[2], xyz_max[2])
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.set_zlabel('PC3')

    # Legend: one proxy handle per combo with its marker+color
    handles = []
    for v in uniq:
        color_idx = int(v) // 3
        shape_idx = int(v) % 3
        name = category_names[int(v)]
        handles.append(
            Line2D([0],[0], marker=SHAPE_MARKERS_MPL[shape_idx], linestyle='None',
                   markerfacecolor=COLORS[color_idx], markeredgecolor='none',
                   markersize=8, label=name)
        )
    fig.legend(handles=handles, loc='upper center',
               ncols=min(len(uniq), 6), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0,0,1,0.96])
    return fig



def save_joint_split3d_html(emb_low, emb_high, labels, label_name, layer_idx, save_dir, category_names=None):
    """
    Interactive HTML: 1x2 Plotly scenes. Left=Low, Right=High.
    - Marker symbol encodes SHAPE (circle, square, triangle-up)
    - Color encodes COLOR (red, green, blue)
    - Legend: one entry per combo with its marker+color (dummy traces)
    """
    if not PLOTLY_AVAILABLE:
        print("[WARN] plotly not available; skipping HTML export for 3D embeddings.")
        return

    uniq = np.unique(labels)

    # If not supplied, build combo names in color-major order
    if category_names is None:
        category_names = [f"{c}×{s}" for c in COLORS for s in SHAPES]

    all_pts = np.vstack([emb_low, emb_high])
    xyz_min = all_pts.min(axis=0); xyz_max = all_pts.max(axis=0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"LOW • by {label_name}", f"HIGH • by {label_name}")
    )

    # Panels: data traces (no legend)
    def _add_panel(emb, col_idx):
        for v in uniq:
            color_idx = int(v) // 3
            shape_idx = int(v) % 3
            m = (labels == v)
            col = COLORS[color_idx]
            sym = SHAPE_SYMBOLS_PLOTLY[shape_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=emb[m,0], y=emb[m,1], z=emb[m,2],
                    mode='markers',
                    marker=dict(size=4.5, color=col, symbol=sym),
                    showlegend=False
                ),
                row=1, col=col_idx
            )

    _add_panel(emb_low, 1)
    _add_panel(emb_high, 2)

    # Legend: dummy traces with the right symbol+color per combo (attach to scene1)
    for v in uniq:
        color_idx = int(v) // 3
        shape_idx = int(v) % 3
        name = category_names[int(v)]
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(size=8, color=COLORS[color_idx], symbol=SHAPE_SYMBOLS_PLOTLY[shape_idx]),
                name=name, showlegend=True
            ),
            row=1, col=1
        )

    scene_layout = dict(
        xaxis=dict(title='PC1', range=[xyz_min[0], xyz_max[0]]),
        yaxis=dict(title='PC2', range=[xyz_min[1], xyz_max[1]]),
        zaxis=dict(title='PC3', range=[xyz_min[2], xyz_max[2]])
    )
    fig.update_layout(
        title=f"Layer {layer_idx} • Joint 3D embedding (separate axes) • by {label_name}",
        scene=scene_layout, scene2=scene_layout,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing='constant')
    )

    suffix = "combo" if label_name.lower() == "combo" else label_name.lower()
    html_path = os.path.join(save_dir, f"layer{layer_idx:02d}_embedding_{suffix}_split.html")
    fig.write_html(html_path)


# ----------------------------
# Main experiment
# ----------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def collect_probs(model, X, device="cpu"):
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy().ravel()
        return expit(logits)

def run_experiment(
    n_train=20000, n_val=4000, n_test=8000,
    tau1=0.7, tau2=0.3, tau1_shift=0.75, tau2_shift=0.25,
    bias_std_low=0.01, bias_std_high=5.0,
    weight_std=None, nonlin="tanh", layernorm=False,
    hidden=256, depth=4, epochs=25, lr=1e-3, batch=512,
    freeze_bias_epochs=0, device="cpu",
    use_mds=False, random_state=0, mds_metric="euclidean",
    input_noise_std=0.0, label_flip_prob=0.0,
    save_dir="figures/mlp_composition"
):
    ensure_dir(save_dir)
    pdf_path = os.path.join(save_dir, "mlp_report.pdf")
    print("Building splits...")
    splits = make_splits(n_train, n_val, n_test,
                         holdout_pairs=[(1,2), (2,1)],
                         tau1=tau1, tau2=tau2,
                         tau1_shift=tau1_shift, tau2_shift=tau2_shift,
                         seed=1234,
                         input_noise_std=input_noise_std,
                         label_flip_prob=label_flip_prob)
    Xtr, ytr, R1tr, R2tr = splits["train"]
    Xva, yva, R1va, R2va = splits["val"]
    Xte, yte, R1te, R2te = splits["test"]
    Xcg, ycg, R1cg, R2cg = splits["ood_cg"]
    Xth, yth, R1th, R2th = splits["ood_th"]
    probe = splits["probe"]

    # DataLoaders
    train_loader = make_loader(Xtr, ytr, batch=batch, shuffle=True)
    val_loader   = make_loader(Xva, yva, batch=batch, shuffle=False)
    test_loader  = make_loader(Xte, yte, batch=batch, shuffle=False)
    cg_loader    = make_loader(Xcg, ycg, batch=batch, shuffle=False)
    th_loader    = make_loader(Xth, yth, batch=batch, shuffle=False)

    # Models
    m_low = MLP(in_dim=9, hidden=hidden, depth=depth, out_dim=1,
                nonlin=nonlin, layernorm=layernorm,
                bias_std=bias_std_low, weight_std=weight_std, seed=123)
    m_high = MLP(in_dim=9, hidden=hidden, depth=depth, out_dim=1,
                 nonlin=nonlin, layernorm=layernorm,
                 bias_std=bias_std_high, weight_std=weight_std, seed=123)

    # Train
    hist_low = train_model(m_low,  train_loader, val_loader, epochs=epochs, lr=lr,
                           freeze_bias_epochs=freeze_bias_epochs, device=device, desc="LowBias")
    hist_high = train_model(m_high, train_loader, val_loader, epochs=epochs, lr=lr,
                            freeze_bias_epochs=freeze_bias_epochs, device=device, desc="HighBias")

    # Eval
    crit = nn.BCEWithLogitsLoss()
    print("Evaluating...")
    vloss_low, vacc_low   = evaluate_model(m_low,  val_loader,  device=device, crit=crit)
    vloss_high, vacc_high = evaluate_model(m_high, val_loader,  device=device, crit=crit)
    tloss_low, tacc_low   = evaluate_model(m_low,  test_loader, device=device, crit=crit)
    tloss_high, tacc_high = evaluate_model(m_high, test_loader, device=device, crit=crit)
    cgacc_low             = evaluate_model(m_low,  cg_loader,   device=device, crit=None)
    cgacc_high            = evaluate_model(m_high, cg_loader,   device=device, crit=None)
    thacc_low             = evaluate_model(m_low,  th_loader,   device=device, crit=None)
    thacc_high            = evaluate_model(m_high, th_loader,   device=device, crit=None)
    print(f"VAL    acc: low={vacc_low:.3f} | high={vacc_high:.3f}")
    print(f"TEST   acc: low={tacc_low:.3f} | high={tacc_high:.3f}")
    print(f"OOD-CG acc: low={cgacc_low:.3f} | high={cgacc_high:.3f}")
    print(f"OOD-TH acc: low={thacc_low:.3f} | high={thacc_high:.3f}")

    with PdfPages(pdf_path) as pdf:
        # 1) Training curves
        fig = plot_training(hist_low, hist_high); fig_to_pdf(fig, pdf)

        # 2) OOD bars
        labels3 = ["ID", "OOD-CG", "OOD-TH"]
        x = np.arange(len(labels3)); w = 0.35
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(x - w/2, [tacc_low,  cgacc_low,  thacc_low],  width=w, label="Low",  color=PALETTE[0])
        ax.bar(x + w/2, [tacc_high, cgacc_high, thacc_high], width=w, label="High", color=PALETTE[1])
        ax.set_xticks(x); ax.set_xticklabels(labels3); ax.set_ylim(0,1.05)
        ax.set_ylabel("Accuracy"); ax.set_title("Generalization performance"); ax.legend()
        plt.tight_layout(); fig_to_pdf(fig, pdf)

        # 3) Probes + 4) Embeddings (per layer)
        Xp = probe["X"]; R1p = probe["R1"]; R2p = probe["R2"]; yp = probe["y"]
        color_p, shape_p, R1_lbl, R2_lbl, y_lbl = build_feature_labels(Xp, R1p, R2p, yp)
        combo_labels, combo_names = make_combo_labels(color_p, shape_p)

        acts_low  = collect_activations(m_low,  Xp, device=device)
        acts_high = collect_activations(m_high, Xp, device=device)

        layers = list(range(len(acts_low)))
        probe_scores_low  = {"color": [], "shape": [], "R1": [], "R2": [], "y": []}
        probe_scores_high = {"color": [], "shape": [], "R1": [], "R2": [], "y": []}

        for L in layers:
            ZL_low, ZL_high = acts_low[L], acts_high[L]
            probe_scores_low["color"].append(fit_probe_classification(ZL_low,  color_p))
            probe_scores_high["color"].append(fit_probe_classification(ZL_high, color_p))
            probe_scores_low["shape"].append(fit_probe_classification(ZL_low,  shape_p))
            probe_scores_high["shape"].append(fit_probe_classification(ZL_high, shape_p))
            probe_scores_low["R1"].append(fit_probe_classification(ZL_low,  R1_lbl))
            probe_scores_high["R1"].append(fit_probe_classification(ZL_high, R1_lbl))
            probe_scores_low["R2"].append(fit_probe_classification(ZL_low,  R2_lbl))
            probe_scores_high["R2"].append(fit_probe_classification(ZL_high, R2_lbl))
            probe_scores_low["y"].append(fit_probe_classification(ZL_low,  y_lbl))
            probe_scores_high["y"].append(fit_probe_classification(ZL_high, y_lbl))

            # Shared 3D embedding (choose PCA or GPU MDS)
            ZL_low_s  = StandardScaler().fit_transform(ZL_low)
            ZL_high_s = StandardScaler().fit_transform(ZL_high)
            emb_low, emb_high = joint_embedding(
                ZL_low_s, ZL_high_s,
                method=("mds_gpu" if use_mds else "pca"),
                n_components=3,
                random_state=random_state,
                mds_metric=mds_metric,
                device=("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Save interactive split HTML colored by combo
            save_joint_split3d_html(emb_low, emb_high, combo_labels, "COMBO", L, save_dir, category_names=combo_names)
            # Also push static split PDF page
            fig = plot_joint_split3d_pdf(emb_low, emb_high, combo_labels, "COMBO", L, category_names=combo_names)
            fig_to_pdf(fig, pdf)

            # RSA Low vs High (print only)
            from sklearn.metrics import pairwise_distances
            rdm_low  = pairwise_distances(ZL_low_s,  metric="correlation")
            rdm_high = pairwise_distances(ZL_high_s, metric="correlation")
            iu = np.triu_indices_from(rdm_low, k=1)
            rho, _ = spearmanr(rdm_low[iu], rdm_high[iu])
            print(f"[Layer {L}] RSA (Spearman) Low vs High RDM: {rho:.3f}")

        # Combined probe curves
        fig = plot_probe_curves_combined(layers, probe_scores_low, probe_scores_high,
                                         title="Decoding accuracy (linear probes)")
        fig_to_pdf(fig, pdf)

        # 5) Psychometrics
        ss_low,  p_low,  k_low,  s0_low  = psychometric_size_sweep(m_low,  color_idx=0, shape_idx=1, x_sign=+1, tau1=tau1, tau2=tau2, device=device)
        ss_high, p_high, k_high, s0_high = psychometric_size_sweep(m_high, color_idx=0, shape_idx=1, x_sign=+1, tau1=tau1, tau2=tau2, device=device)
        fig, ax = plt.subplots(figsize=(6.5,4))
        ax.plot(ss_low,  p_low,  marker=LOW_MARKER,  color=PALETTE[0], label=f"Low (k={k_low:.2f}, s0={s0_low:.2f})")
        ax.plot(ss_high, p_high, marker=HIGH_MARKER, color=PALETTE[1], label=f"High (k={k_high:.2f}, s0={s0_high:.2f})")
        ax.set_xlabel("size s"); ax.set_ylabel("P(y=1)"); ax.set_title("Psychometric (size sweep, red×square, x>0)")
        ax.legend(); plt.tight_layout(); fig_to_pdf(fig, pdf)

        # 6) Calibration (ID, OOD-CG, OOD-TH)
        def plot_reliability(probs_low, labels_low, probs_high, labels_high, title_suffix="Test"):
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            def reliability_diagram(ax0, probs, labels, n_bins=15, color="#1f77b4", label=""):
                bins = np.linspace(0.0, 1.0, n_bins+1); xs, ys = [], []
                for i in range(n_bins):
                    m = (probs >= bins[i]) & (probs < bins[i+1])
                    if m.sum() == 0: continue
                    xs.append(probs[m].mean()); ys.append((labels[m] == (probs[m] >= 0.5)).mean())
                ax0.plot([0,1],[0,1], '--', color="#888888", linewidth=1)
                ax0.plot(xs, ys, marker='o', color=color, label=label)
                ax0.set_xlabel("Confidence"); ax0.set_ylabel("Accuracy")
                if label: ax0.legend()
            reliability_diagram(ax[0], probs_low,  labels_low,  color=PALETTE[0], label="Low")
            reliability_diagram(ax[0], probs_high, labels_high, color=PALETTE[1], label="High")
            ax[0].set_title(f"Reliability ({title_suffix})")
            def ece(probs, labels, n_bins=15):
                bins = np.linspace(0.0, 1.0, n_bins+1); e = 0.0
                for i in range(n_bins):
                    m = (probs >= bins[i]) & (probs < bins[i+1])
                    if m.sum() == 0: continue
                    acc  = (labels[m] == (probs[m] >= 0.5)).mean()
                    conf = probs[m].mean()
                    e += (m.mean()) * abs(acc - conf)
                return float(e)
            ece_low, ece_high = ece(probs_low, labels_low), ece(probs_high, labels_high)
            ax[1].bar([0,1], [ece_low, ece_high], color=[PALETTE[0], PALETTE[1]])
            ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Low","High"])
            ax[1].set_ylabel("ECE"); ax[1].set_title(f"Calibration error ({title_suffix})")
            plt.tight_layout()
            return fig

        probs_low_test  = collect_probs(m_low,  Xte, device=device)
        probs_high_test = collect_probs(m_high, Xte, device=device)
        fig = plot_reliability(probs_low_test,  yte.astype(int), probs_high_test, yte.astype(int), title_suffix="ID Test"); fig_to_pdf(fig, pdf)

        probs_low_cg  = collect_probs(m_low,  Xcg, device=device)
        probs_high_cg = collect_probs(m_high, Xcg, device=device)
        fig = plot_reliability(probs_low_cg,  ycg.astype(int), probs_high_cg, ycg.astype(int), title_suffix="OOD-CG"); fig_to_pdf(fig, pdf)

        probs_low_th  = collect_probs(m_low,  Xth, device=device)
        probs_high_th = collect_probs(m_high, Xth, device=device)
        fig = plot_reliability(probs_low_th,  yth.astype(int), probs_high_th, yth.astype(int), title_suffix="OOD-TH"); fig_to_pdf(fig, pdf)

    print(f"Saved multi-page report: {os.path.abspath(pdf_path)}")
    print(f"Interactive 3D HTMLs are in: {os.path.abspath(save_dir)}")
    print("Done.")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    run_experiment(
        n_train=20000, n_val=4000, n_test=8000,
        tau1=0.7, tau2=0.3, tau1_shift=0.75, tau2_shift=0.25,
        bias_std_low=0.01, bias_std_high=5.0,
        weight_std=None,
        nonlin="tanh", layernorm=False,
        hidden=256, depth=4,
        epochs=25, lr=1e-3, batch=512,
        freeze_bias_epochs=1,
        device="cuda",                 # or "cuda"
        use_mds=True,                # set True to enable GPU classical MDS
        random_state=0,
        mds_metric="euclidean",
        input_noise_std=0.10,
        label_flip_prob=0.10,
        save_dir="figures/mlp_composition"
    )
