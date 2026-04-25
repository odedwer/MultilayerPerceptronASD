import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from config import RNNConfig, default_bias_means
from models import LSTMWithGateBias, RNNWithGateBias
from datasets import make_sparse_hmm, DelayedCopyHMM
import utils


class MetricsCollector:
    def __init__(self):
        self.results = []

    def add_result(self, params, metrics):
        res = {"params": params}
        res.update(metrics)
        self.results.append(res)

    def get_results(self):
        return self.results


class ParameterSweep:
    def __init__(self, config_base, param_name, param_values, device='cpu'):
        self.config_base = config_base
        self.param_name = param_name
        self.param_values = param_values
        self.device = device
        self.collector = MetricsCollector()

    def run(self, num_runs=3):
        for val in self.param_values:
            print(f"Running sweep for {self.param_name} = {val}")
            for run_id in range(num_runs):
                print(f"  Run {run_id + 1}/{num_runs}")
                cfg = self.config_base.replace(**{self.param_name: val, "seed": self.config_base.seed + run_id})
                metrics = self.run_single_experiment(cfg)
                self.collector.add_result(val, metrics)

    def run_single_experiment(self, cfg):
        """Train a model with the given config and return research metrics."""
        device = cfg.device
        rng = np.random.RandomState(cfg.seed)

        # Build HMM datasets
        T, E = make_sparse_hmm(cfg.M_states, cfg.K_symbols, cfg.s_transitions, cfg.s_emissions, rng)
        train_ds = DelayedCopyHMM(cfg.n_train, T, E, cfg, rng)
        val_ds = DelayedCopyHMM(cfg.n_val, T, E, cfg, rng)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        # Build model
        model = cfg.model(cfg.K_symbols + 2, cfg.emb_dim, cfg.hidden_size, cfg).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.K_symbols)

        # Training loop
        for epoch in range(cfg.epochs):
            model.train()
            for X, Y, Z, _, delay_mask in train_loader:
                X = X.to(device)
                Y = Y.to(device)
                delay_mask = delay_mask.to(device)

                optimizer.zero_grad()
                out, H = model(X)
                loss = criterion(out[delay_mask], Y[delay_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        # Evaluation and metric collection
        model.eval()
        all_H = []
        total_correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for X, Y, Z, _, delay_mask in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                delay_mask = delay_mask.to(device)

                out, H = model(X)
                loss = criterion(out[delay_mask], Y[delay_mask])
                total_loss += loss.item() * X.size(0)
                preds = out.argmax(-1)
                total_correct += (preds[delay_mask] == Y[delay_mask]).sum().item()
                total += delay_mask.sum().item()
                all_H.append(H.cpu().numpy())

        val_loss = total_loss / len(val_loader.dataset)
        val_acc = total_correct / total if total > 0 else 0.0

        # Research metrics on collected hidden states
        H_np = np.concatenate(all_H, axis=0)  # (N_batch, T, H_dim)
        max_tau = min(10, H_np.shape[1] - 1)
        decay = utils.memory_decay(H_np, max_tau=max_tau)
        eff_dim = utils.effective_dimensionality_extended(H_np)

        # Jacobian spectral radius using first two tokens of a validation sample
        sample_x = val_ds[0][0].unsqueeze(0)  # (1, T)
        sr = utils.jacobian_spectral_radius(model, sample_x[:, :2], device=device)

        return {
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "memory_decay_lag1": float(decay[0]) if len(decay) > 0 else float("nan"),
            "effective_dim": float(eff_dim),
            "jacobian_sr": float(sr),
        }


class Visualizer:
    @staticmethod
    def plot_sweep_results(sweep_results, param_name, metric_names=None, save_path=None):
        """
        Plot sweep results: one subplot per metric showing mean ± std across runs.

        :param sweep_results: List of dicts from MetricsCollector.get_results().
                              Each dict has 'params' key and metric keys.
        :param param_name: Name of the swept parameter (x-axis label).
        :param metric_names: List of metric keys to plot. If None, plots all metrics found.
        :param save_path: If provided, saves the figure to this path.
        """
        if not sweep_results:
            print("No results to plot.")
            return

        # Gather unique param values in order
        all_params = []
        seen = set()
        for r in sweep_results:
            p = r["params"]
            if p not in seen:
                all_params.append(p)
                seen.add(p)

        # Determine which metrics to plot
        if metric_names is None:
            metric_names = [k for k in sweep_results[0].keys() if k != "params"]

        # For each param value, collect all runs' metrics
        data = {m: {p: [] for p in all_params} for m in metric_names}
        for r in sweep_results:
            p = r["params"]
            for m in metric_names:
                if m in r:
                    data[m][p].append(r[m])

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
        axes = axes[0]

        for ax, m in zip(axes, metric_names):
            means = [np.nanmean(data[m][p]) for p in all_params]
            stds = [np.nanstd(data[m][p]) for p in all_params]
            x = np.array(all_params, dtype=float)
            ax.plot(x, means, marker='o', linewidth=2)
            ax.fill_between(x,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.3)
            ax.set_xlabel(param_name)
            ax.set_ylabel(m)
            ax.set_title(m.replace("_", " ").title())
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"Parameter Sweep: {param_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


if __name__ == "__main__":
    base_cfg = RNNConfig(
        epochs=5,
        n_train=1000,
        n_val=200,
        hidden_size=64,
        bias_means=default_bias_means(),
        global_bias_std=0.0,
    )

    sweep = ParameterSweep(
        config_base=base_cfg,
        param_name="global_bias_std",
        param_values=[0.0, 0.5, 1.0, 2.0],
        device=base_cfg.device,
    )
    sweep.run(num_runs=2)

    results = sweep.collector.get_results()
    print(f"\nCollected {len(results)} results.")

    os.makedirs("figures", exist_ok=True)
    fig = Visualizer.plot_sweep_results(
        results,
        param_name="global_bias_std",
        save_path="figures/sweep_bias_std.png",
    )
    plt.show()
