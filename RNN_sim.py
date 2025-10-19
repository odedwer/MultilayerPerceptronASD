# RNN_sim.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from config import RNNConfig, default_bias_means
from datasets import make_sparse_hmm, rewire_transitions, DelayedCopyHMM
from models import LSTMWithGateBias
import utils
import seaborn as sns
import os, json, hashlib, time
from dataclasses import asdict
import itertools


def cfg_to_ordered_json(cfg) -> str:
    """Convert config dataclass to stable JSON string for hashing."""
    d = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
    d = {k: v for k, v in d.items() if k not in ("name", "results_dir")}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def fingerprint_cfg(cfg) -> str:
    """Hash configuration for unique run directory."""
    s = cfg_to_ordered_json(cfg)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def run_dir_for_cfg(cfg) -> str:
    base = getattr(cfg, "results_dir", "runs")
    sub = f"{fingerprint_cfg(cfg)}__{cfg.name}"
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


# --- use npz files for arrays ---
def save_run(run_dir, cfg, model, train_losses, train_metrics, final_metrics):
    os.makedirs(run_dir, exist_ok=True)

    # Save config (JSON)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Save metrics (NPZ)
    np.savez_compressed(os.path.join(run_dir, "metrics.npz"),
                        train_losses=train_losses,
                        final_metrics=final_metrics,
                        train_metrics=train_metrics)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))


def load_run_if_exists(cfg):
    """Check for existing run and load from npz if available."""
    rd = run_dir_for_cfg(cfg)
    cfg_path = os.path.join(rd, "config.json")
    if not os.path.exists(cfg_path):
        return rd, None, None, None, None

    with open(cfg_path, "r") as f:
        stored = f.read()
    if stored != cfg_to_ordered_json(cfg):
        return rd, None, None, None, None

    try:
        losses = np.load(os.path.join(rd, "train_losses.npz"))["arr_0"].tolist()
        train_metrics = np.load(os.path.join(rd, "train_metrics.npz"), allow_pickle=True)["train_metrics"].tolist()
        fm = dict(np.load(os.path.join(rd, "final_metrics.npz"), allow_pickle=True))
        model_path = os.path.join(rd, "model.pt")
        return rd, losses, train_metrics, fm, model_path
    except Exception as e:
        print(f"[warn] Failed to load cached results: {e}")
        return rd, None, None, None, None


def compute_dimensionality(H, eps=1e-12):
    """
    Compute effective dimensionality metrics for a 2D array of hidden states (samples x features).
    Returns: pcs80, pcs90, pcs95, eff_rank
    """
    if H.ndim != 2:
        H = H.reshape(-1, H.shape[-1])

    H_centered = H - H.mean(0, keepdim=True)
    cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)

    eigvals = torch.linalg.eigvalsh(cov.float()).cpu().numpy()
    eigvals = np.clip(np.real(eigvals), eps, None)
    eigvals = np.sort(eigvals)[::-1]
    total_var = eigvals.sum()

    if total_var < eps:
        return np.nan, np.nan, np.nan, np.nan

    pcs80 = np.searchsorted(np.cumsum(eigvals) / total_var, 0.80) + 1
    pcs90 = np.searchsorted(np.cumsum(eigvals) / total_var, 0.90) + 1
    pcs95 = np.searchsorted(np.cumsum(eigvals) / total_var, 0.95) + 1
    eff_rank = np.exp(-np.sum((eigvals/total_var) * np.log((eigvals/total_var) + eps)))
    return pcs80, pcs90, pcs95, eff_rank

# ---------- Linear Probe ----------
def train_linear_probe(H, y, out_dim, cfg, l2=1e-3):
    """Closed-form ridge classifier with bias & centering (used for all probes)."""
    device = cfg.device
    mask = y >= 0
    X = H[mask].to(device)
    Y = y[mask].to(device)
    if X.numel() == 0:
        return float("nan")

    N = X.size(0)
    torch.manual_seed(cfg.seed)
    idx = torch.randperm(N, device=device)
    ntr = int(0.8 * N)
    Xtr, Xval = X[idx[:ntr]], X[idx[ntr:]]
    Ytr, Yval = Y[idx[:ntr]], Y[idx[ntr:]]

    # one-hot targets
    T = torch.zeros((Xtr.size(0), out_dim), device=device)
    T[torch.arange(Xtr.size(0), device=device), Ytr] = 1.0

    mu = Xtr.mean(0, keepdim=True)
    Xtr_c, Xval_c = Xtr - mu, Xval - mu
    ones_tr = torch.ones(Xtr_c.size(0), 1, device=device)
    ones_val = torch.ones(Xval_c.size(0), 1, device=device)
    Xtr_aug = torch.cat([Xtr_c, ones_tr], 1)
    Xval_aug = torch.cat([Xval_c, ones_val], 1)

    Hdim = Xtr_c.size(1)
    I = torch.eye(Hdim + 1, device=device)
    I[-1, -1] = 0.0
    W = torch.linalg.solve(Xtr_aug.T @ Xtr_aug + l2 * I, Xtr_aug.T @ T)

    pred = (Xval_aug @ W).argmax(-1)
    return (pred == Yval).float().mean().item()
# ---------- Train & Evaluate ----------
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def probe_per_class(H_mem, labels_2d, n_classes):
    """
    For each sequence, take one memory vector and all unique labels (states or tokens)
    that appeared during the input phase, then probe across all such pairs.
    """
    H_list, y_list = [], []
    H_mem_np = H_mem.detach().cpu().numpy()
    labels_np = labels_2d.detach().cpu().numpy()

    for i in range(H_mem_np.shape[0]):  # iterate over sequences
        unique_labels = np.unique(labels_np[i][labels_np[i] >= 0])
        for label in unique_labels:
            H_list.append(H_mem_np[i])
            y_list.append(label)

    if len(y_list) == 0:
        return np.nan

    H_stack = np.stack(H_list)
    y_stack = np.array(y_list)

    scores = []
    for label in np.unique(y_stack):
        mask = (y_stack == label)
        if mask.sum() < 5:
            continue
        clf = LogisticRegression(max_iter=500, n_jobs=-1)
        scaler = StandardScaler()
        H = scaler.fit_transform(H_stack)
        clf.fit(H, (y_stack == label).astype(int))
        pred = clf.predict(H)
        scores.append(f1_score((y_stack == label).astype(int), pred))

    return float(np.mean(scores)) if scores else np.nan

# ---------- Train ----------
def train_model(model, train_loader, val_loader, test_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    history = {k: [] for k in [
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
        "probe_hmm", "probe_token", "probe_baseline",
        "pcs80", "pcs90", "pcs95", "eff_rank"
    ]}

    for epoch in range(cfg.epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        # Each batch returns 5 items, including y_delay
        X_all, H_all, Z_all, Y_all, delay_masks = [], [], [], [],[]

        for X, Y, Z, _, delay_mask in train_loader:
            X, Y, Z, delay_mask = X.to(cfg.device), Y.to(cfg.device), Z.to(cfg.device), delay_mask.to(cfg.device)

            optimizer.zero_grad()
            out, H = model(X)

            # Cross-entropy over all timesteps
            loss = criterion(out.transpose(1, 2), Y)
            loss.backward()
            optimizer.step()

            # Accuracy computation
            preds = out.argmax(-1)
            mask = ((Y >= 0) & (Y<cfg.K_symbols))  # all valid time steps
            total_correct += (preds[mask] == Y[mask]).sum().item()
            total += mask.sum().item()  # ✅ ensure nonzero
            total_loss += loss.item() * X.size(0)

            # Save for probes
            H_all.append(H.detach())
            # Z_all.append(Z.detach().cpu())
            # Y_all.append(Y.detach().cpu())
            # X_all.append(X.detach().cpu())
            # delay_masks.append(delay_mask.detach().cpu())

        # aggregate
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / total if total > 0 else 0.0  # ✅ safe guard

        # Flatten over batches
        H_all = torch.cat([a.cpu() for a in H_all], 0)
        # Z_all = torch.cat(Z_all, 0)
        # Y_all = torch.cat(Y_all, 0)
        # X_all = torch.cat(X_all, 0)
        # delay_masks = torch.cat(delay_masks, 0)  # (batch, time)

        # H_mem = H_all[:, cfg.L_input + cfg.D_delay - 1]
        # ---- Extract input-phase labels ----
        # Z_input = Z_all[:, :cfg.L_input]
        # X_input = X_all[:, :cfg.L_input]

        # ---- Compute per-class decoding with unique label aggregation ----
        # probe_hmm = probe_per_class(H_mem, Z_input, cfg.M_states)
        # probe_token = probe_per_class(H_mem, X_input, cfg.K_symbols)

        # ---- Baseline: token → HMM structure ----
        # X_unique_vectors, Z_unique_labels = [], []
        # X_np, Z_np = X_input.cpu().numpy(), Z_input.cpu().numpy()
        # for i in range(X_np.shape[0]):
        #     unique_tokens = np.unique(X_np[i][X_np[i] >= 0])
        #     for token in unique_tokens:
        #         X_unique_vectors.append(np.eye(cfg.K_symbols)[token])
        #         Z_unique_labels.append(np.argmax(np.bincount(Z_np[i][Z_np[i] >= 0])))
        # if len(Z_unique_labels) > 0:
        #     onehot_X = np.stack(X_unique_vectors)
        #     y_base = np.array(Z_unique_labels)
        #     scores = []
        #     for label in np.unique(y_base):
        #         mask = (y_base == label)
        #         if mask.sum() < 5:
        #             continue
        #         clf = LogisticRegression(max_iter=500, n_jobs=-1)
        #         clf.fit(onehot_X, (y_base == label).astype(int))
        #         pred = clf.predict(onehot_X)
        #         scores.append(f1_score((y_base == label).astype(int), pred))
        #     probe_baseline = float(np.mean(scores)) if scores else np.nan
        # else:
        #     probe_baseline = np.nan

        # --- Compute PCA metrics (effective dimensionality) per phase ---
        H_np = H_all.detach().cpu()

        # Define phase boundaries
        L, D = cfg.L_input, cfg.D_delay
        T_total = H_np.shape[1]
        phase_indices = {
            "input": (0, L),
            "delay": (L, L + D),
            "output": (L + D + 1, T_total)
        }

        phase_dims = {}
        for phase, (start, end) in phase_indices.items():
            if end <= start or end > T_total:
                phase_dims[phase] = (np.nan, np.nan, np.nan, np.nan)
                continue
            H_phase = H_np[:, start:end, :].reshape(-1, H_np.shape[-1])
            # --- Subsample to limit computation ---
            if H_phase.shape[0] > cfg.pca_sample_size:
                idx = torch.randperm(H_phase.shape[0])[:cfg.pca_sample_size]
                H_phase = H_phase[idx]

            phase_dims[phase] = compute_dimensionality(H_phase.detach().clone())

        # unpack into variables for logging
        pcs80_in, pcs90_in, pcs95_in, eff_in = phase_dims["input"]
        pcs80_dl, pcs90_dl, pcs95_dl, eff_dl = phase_dims["delay"]
        pcs80_out, pcs90_out, pcs95_out, eff_out = phase_dims["output"]

        # --- validation and test (only accuracy & loss)
        val_metrics = evaluate(model, val_loader, cfg)
        test_metrics = evaluate(model, test_loader, cfg)
        # if (epoch) % cfg.test_eval_interval == 0:
        #     test_metrics = evaluate(model, test_loader, cfg)
        # else:
        #     test_metrics = {"loss": np.nan, "acc": np.nan}


        # --- store metrics
        for key in [
            "train_loss", "val_loss", "test_loss",
            "train_acc", "val_acc", "test_acc",
            "probe_hmm", "probe_token", "probe_baseline",
            "pcs80_in", "pcs90_in", "pcs95_in", "eff_in",
            "pcs80_dl", "pcs90_dl", "pcs95_dl", "eff_dl",
            "pcs80_out", "pcs90_out", "pcs95_out", "eff_out"
        ]:
            history.setdefault(key, [])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])
        history["probe_hmm"].append(np.nan)#probe_hmm)
        history["probe_token"].append(np.nan)#probe_token)
        history["probe_baseline"].append(np.nan)#probe_baseline)
        history["pcs80_in"].append(pcs80_in)
        history["pcs90_in"].append(pcs90_in)
        history["pcs95_in"].append(pcs95_in)
        history["eff_in"].append(eff_in)

        history["pcs80_dl"].append(pcs80_dl)
        history["pcs90_dl"].append(pcs90_dl)
        history["pcs95_dl"].append(pcs95_dl)
        history["eff_dl"].append(eff_dl)

        history["pcs80_out"].append(pcs80_out)
        history["pcs90_out"].append(pcs90_out)
        history["pcs95_out"].append(pcs95_out)
        history["eff_out"].append(eff_out)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss={train_loss:.3f} | Acc={train_acc:.3f} | "
            f"ValAcc={val_metrics['acc']:.3f} | TestAcc={test_metrics['acc']:.3f} | "
            # f"HMMProbe={probe_hmm:.3f} | TokenProbe={probe_token:.3f}"
        )

    return history


# ---------- Evaluate ----------
@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    for X, Y, _,_,_ in loader:   # assuming dataset returns (input, target, hidden_labels)
        X, Y = X.to(cfg.device), Y.to(cfg.device)
        out, _ = model(X)
        loss = criterion(out.transpose(1, 2), Y)
        total_loss += loss.item() * X.size(0)

        mask = (Y >= 0) & (Y<cfg.K_symbols)
        preds = out.argmax(-1)
        total_correct += (preds == Y).float().sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / total
    return {"loss": avg_loss, "acc": acc}



def plot_transition_matrices(T_true, T_emp, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(T_true, ax=axes[0], cmap="viridis", vmin=0, vmax=1, cbar=False)
    sns.heatmap(T_emp, ax=axes[1], cmap="viridis", vmin=0, vmax=1, cbar=False)
    axes[0].set_title("True HMM Transitions")
    axes[1].set_title("Empirical Transitions (from training set)")
    for ax in axes:
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")
    fig.suptitle(f"HMM Transition Comparison ({cfg.name})", fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------- Experiment ----------
def run_experiment(cfg):
    """
    Returns: (run_dir, train_losses, train_metrics, final_metrics, model)
    Uses a cache directory keyed by config. If results exist, loads them; otherwise trains.
    """
    run_dir, tl_cached, tm_cached, fm_cached, model_path = load_run_if_exists(cfg)
    cfg.run_id = os.path.basename(run_dir)
    if tl_cached is not None:
        print(f"[cache hit] Loaded results for {cfg.name} from {run_dir}")
        # if you need the model in-memory, instantiate and load:
        model = LSTMWithGateBias(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(cfg.device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        return run_dir, tl_cached, tm_cached, fm_cached, model

    print(f"[run] Training {cfg.name} (bias_std={cfg.input_gate_bias_std}) on {cfg.device}")
    cfg.bias_means = default_bias_means()
    rng = np.random.RandomState(cfg.seed)

    # Build data (train, val, test as before)
    T, E = make_sparse_hmm(cfg.M_states, cfg.K_symbols, cfg.s_transitions, cfg.s_emissions, rng)
    T_test = rewire_transitions(T, cfg.ood_rewire_frac, cfg.s_transitions, rng)

    n_val = max(cfg.n_val, 500)
    train = DelayedCopyHMM(cfg.n_train, T, E, cfg, rng)
    val = DelayedCopyHMM(n_val, T, E, cfg, rng)
    test = DelayedCopyHMM(cfg.n_test, T_test, E, cfg, rng)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=8,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=8,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False,
        num_workers=8,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=True
    )

    # Create & (optionally) compile model
    model = LSTMWithGateBias(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(cfg.device)
    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    # except Exception:
    #     pass

    # Train with per-epoch eval (your existing function that returns dynamics)
    train_losses, train_metrics = train_model(model, train_loader, val_loader, cfg)
    # Final OOD test metrics
    final_metrics = evaluate(model, test_loader, cfg)

    # Save all
    cfg.run_id = os.path.basename(run_dir)

    save_run(run_dir, cfg, model, train_losses, train_metrics, final_metrics)
    print(f"[saved] {run_dir}")

    return run_dir, train_losses, train_metrics, final_metrics, model


# ---------- Main ----------
def run_comparison(cfg_def, cfg_low, cfg_high):
    """
    Train and compare default, low-var, and high-var models with full diagnostics,
    automatically reusing existing results if available.
    """
    import hashlib, json, os, time
    from copy import deepcopy

    # --- Consistent color palette ---
    colors = {
        # "Default": "#F2AD00",   # Wes Anderson Darjeeling1 yellow
        "Low": utils.NT_COLOR,  # "#00A08A"
        "High": utils.ASD_COLOR # "#FF0000"
    }
    PHASE_STYLES = {
        ('in',"input"): {"ls": "-"},
        ('dl',"delay"): {"ls": "--"},
        ("out","output"): {"ls": ":"}
    }

    plt.rcParams.update({
        # "axes.prop_cycle": plt.cycler("color", [colors["Default"], colors["Low"], colors["High"]]),
        "axes.prop_cycle": plt.cycler("color", [colors["Low"], colors["High"]]),
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.frameon": False
    })

    # --- Train or load results ---
    results, model_paths = {}, {}
    # names = {"Default": cfg_def, "Low": cfg_low, "High": cfg_high}
    names = {"Low": cfg_low, "High": cfg_high}

    for label, cfg in names.items():
        # unique hash from config (sorted)
        cfg_json = json.dumps(cfg.__dict__, sort_keys=True)
        cfg_hash = hashlib.sha1(cfg_json.encode()).hexdigest()[:10]
        run_dir = os.path.join("runs", f"{label}_{cfg_hash}")
        model_paths[label] = run_dir
        os.makedirs(run_dir, exist_ok=True)

        hist_path = os.path.join(run_dir, "history.npz")
        cfg_path = os.path.join(run_dir, "config.json")
        model_path = os.path.join(run_dir, "model.pt")

        if os.path.exists(hist_path):
            print(f"\n🔁 Reusing cached results for {label} ({cfg_hash})")
            data = np.load(hist_path, allow_pickle=True)
            history = {k: data[k].tolist() for k in data.files}
            results[label] = history
            continue

        # === No cache found: train model ===
        print(f"\n🚀 Training {label} model ({cfg_hash})")
        rng = np.random.RandomState(cfg.seed)
        T, E = make_sparse_hmm(cfg.M_states, cfg.K_symbols, cfg.s_transitions, cfg.s_emissions, rng)
        T_test = rewire_transitions(T, cfg.ood_rewire_frac, cfg.s_transitions, rng)

        train = DelayedCopyHMM(cfg.n_train, T, E, cfg, rng)
        val = DelayedCopyHMM(cfg.n_val, T, E, cfg, rng)
        test = DelayedCopyHMM(cfg.n_test, T_test, E, cfg, rng)

        loaders = [
            torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=(i == 0))
            for i, ds in enumerate([train, val, test])
        ]
        train_loader, val_loader, test_loader = loaders

        model = LSTMWithGateBias(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(cfg.device)

        history = train_model(model, train_loader, val_loader, test_loader, cfg)
        results[label] = history

        # save artifacts
        np.savez_compressed(hist_path, **history)
        torch.save(model.state_dict(), model_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

    # --- Plotting ---
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.ravel()
    epochs = range(1, cfg_def.epochs + 1)

    # A. Train & Val Loss
    for label in results:
        ax[0].plot(epochs, results[label]["train_loss"], "-", color=colors[label], label=f"{label} Train")
        ax[0].plot(epochs, results[label]["val_loss"], "--", color=colors[label], label=f"{label} Val")
    ax[0].set_title("A. Training & Validation Loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()

    # B. Train & Val Accuracy
    for label in results:
        ax[1].plot(epochs, results[label]["train_acc"], "-", color=colors[label], label=f"{label} Train")
        ax[1].plot(epochs, results[label]["val_acc"], "--", color=colors[label], label=f"{label} Val")
    ax[1].set_title("B. Training & Validation Accuracy")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend()

    # C. Test Accuracy (OOD)
    for label in results:
        ax[2].plot(epochs, results[label]["test_acc"], color=colors[label], lw=2, label=label)
    ax[2].set_title("C. Test Accuracy (OOD → memorization)")
    ax[2].set_xlabel("Epoch"); ax[2].set_ylabel("Accuracy"); ax[2].legend()

    # === D: PCs for 95% Explained Variance (per phase) ===
    epochs = np.arange(len(history["pcs95_dl"]))
    for label in results:
        hist = results[label]
        for (ph,phase), style in PHASE_STYLES.items():
            style.update({"color": colors[label], "lw": 2})
            ax[3].plot(epochs, hist[f"pcs95_{ph}"], label=phase.capitalize(), **style)

    ax[3].set_title("D) #PCs for 95% Explained Variance", loc="left", weight="bold")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("# Principal Components")
    ax[3].legend(frameon=False)

    # E. HMM Probe (+ baseline)
    for label in results:
        ax[4].plot(epochs, results[label]["probe_hmm"], "-", color=colors[label], lw=2, label=f"{label} HMM")
        ax[4].plot(epochs, results[label]["probe_baseline"], "--", color=colors[label], lw=1.5, label=f"{label} baseline")
    ax[4].set_title("E. HMM Probe Accuracy (Train) + Token→State Baseline")
    ax[4].set_xlabel("Epoch"); ax[4].set_ylabel("Accuracy"); ax[4].legend()

    # F. Token Probe
    for label in results:
        ax[5].plot(epochs, results[label]["probe_token"], color=colors[label], lw=2, label=label)
    ax[5].set_title("F. Token Probe Accuracy (Train)")
    ax[5].set_xlabel("Epoch"); ax[5].set_ylabel("Accuracy"); ax[5].legend()

    fig.suptitle("Model Comparisons: Default (yellow) | Low Var (teal) | High Var (red)", fontsize=14, y=1.02)
    plt.tight_layout()

    # --- Save figure and index ---
    timestamp = int(time.time())
    summary_path = f"summary_grid_{timestamp}.svg"
    plt.savefig(os.path.join("results",summary_path), dpi=300, bbox_inches="tight")
    plt.show()

    summary_index = {
        "summary_file": summary_path,
        "runs": [
            {"id": label, "path": path, "color": colors[label]}
            for label, path in model_paths.items()
        ]
    }
    with open(os.path.join("results","summary_index.json"), "w") as f:
        json.dump(summary_index, f, indent=2)

    print(f"\n✅ Summary saved as: {summary_path}")
    print("✅ Index written to: summary_index.json")
    print("✅ Reused runs:", [label for label in results if os.path.exists(os.path.join(model_paths[label], 'history.npz'))])

    return results




if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    print(f"Training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Parameter grid
    M_states_list = [5]  # [4, 5, 6]
    K_symbols_list = [12]
    L_input_list = [10]  # [7, 11, 15]
    D_delay_list = [60]  # [10, 15, 20]
    s_transitions_list = [2]  # [2, 3]
    s_emissions_list = [2]  # [2, 3]
    flip_prob_list = [0.05]  # [0.0, 0.05, 0.2]

    # Create all combinations
    param_combinations = list(itertools.product(
        M_states_list,
        K_symbols_list,
        L_input_list,
        D_delay_list,
        s_transitions_list,
        s_emissions_list,
        flip_prob_list
    ))

    low_high_combs = [
        # (0.1, 5.0),
        (.1, 10.)
    ]

    # Run experiments for each combination
    print(f"Running {len(param_combinations) * len(low_high_combs)} parameter combinations...\n")
    # pbar = trange(len(low_high_combs) * len(param_combinations), desc="Total Progress")
    for low_bias, high_bias in low_high_combs:
        for i, (M, K, L, D, s_t, e_e, flip) in enumerate(param_combinations):
            cfg_def = RNNConfig(
                M_states=M,
                K_symbols=K,
                L_input=L,
                D_delay=D,
                s_transitions=s_t,
                s_emissions=e_e,
                flip_prob=flip,
                ood_rewire_frac=1.0,
                name="default_bias",
                epochs=75
            )
            cfg_low = cfg_def.replace(input_gate_bias_std=low_bias, name="low_bias_std")
            cfg_high = cfg_def.replace(input_gate_bias_std=high_bias, name="high_bias_std")
            run_comparison(cfg_def, cfg_low, cfg_high)
            # pbar.update(1)
