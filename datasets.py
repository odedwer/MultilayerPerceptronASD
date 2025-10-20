# hmm_task.py
import numpy as np
import torch
from torch.utils.data import Dataset
import config as cfg


def normalize_rows(mat):
    s = mat.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return mat / s


def make_sparse_hmm(M, K, s_trans, s_emit, rng):
    T = np.zeros((M, M))
    for i in range(M):
        succ = rng.choice(M, s_trans, replace=False)
        w = rng.rand(s_trans) + 1e-2
        T[i, succ] = w
    T = normalize_rows(T)

    E = np.zeros((M, K))
    for i in range(M):
        sym = rng.choice(K, s_emit, replace=False)
        w = rng.rand(s_emit) + 1e-2
        E[i, sym] = w
    E = normalize_rows(E)
    return T, E


def rewire_transitions(T, frac_rows, s_trans, rng):
    T2 = T.copy()
    n_rows = int(round(frac_rows * T.shape[0]))
    rows = rng.choice(T.shape[0], size=n_rows, replace=False)
    for i in rows:
        succ = rng.choice(T.shape[0], s_trans, replace=False)
        w = rng.rand(s_trans) + 1e-2
        T2[i, :] = 0.0
        T2[i, succ] = w
    return normalize_rows(T2)


def sample_hmm_sequence(T, E, L, rng):
    M, K = E.shape
    z, x = np.zeros(L, int), np.zeros(L, int)
    z[0] = rng.randint(M)
    x[0] = rng.choice(K, p=E[z[0]])
    for t in range(1, L):
        z[t] = rng.choice(M, p=T[z[t - 1]])
        x[t] = rng.choice(K, p=E[z[t]])
    return z, x


class DelayedCopyHMM(Dataset):
    def __init__(self, n_seq, T, E, cfg: [cfg.RNNConfig, None], rng):
        self.cfg = cfg
        self.T = T
        self.E = E
        self.rng = rng
        self.n_seq = n_seq
        self.K = cfg.K_symbols
        self.BLANK, self.GO = self.K, self.K + 1
        self.V = self.K + 2

    def __len__(self): return self.n_seq

    def __getitem__(self, idx):
        L, D, rng = self.cfg.L_input, self.cfg.D_delay, self.rng

        # --- Generate input sequence (HMM-driven) ---
        z_inp, x_inp = sample_hmm_sequence(self.T, self.E, L, rng)

        # Apply symbol flipping (structured noise)
        if getattr(self.cfg, "flip_prob", 0.0) > 0:
            n_flip = int(len(x_inp) * self.cfg.flip_prob)
            if n_flip > 0:
                flip_idx = self.rng.choice(len(x_inp), size=n_flip, replace=False)
                for i in flip_idx:
                    # sample from [0, K) and ensure it's different
                    new_val = self.rng.randint(0, self.cfg.K_symbols)
                    while new_val == x_inp[i]:
                        new_val = self.rng.randint(0, self.cfg.K_symbols)
                    x_inp[i] = new_val

        # --- Construct task phases ---
        # Phase 1: input symbols (model sees them)
        # Phase 2: delay (model sees BLANK)
        # Phase 3: GO cue (signals reproduction)
        # Phase 4: output phase (model reproduces input)

        inp = np.concatenate([
            x_inp,  # input symbols
            np.full(D, self.BLANK),  # delay period (blank)
            [self.GO],  # go cue
            np.full(L, self.BLANK)  # output phase (model generates)
        ])

        tgt = np.full_like(inp, -100)
        tgt[-L:] = x_inp  # only final L positions have target symbols

        # --- bookkeeping ---
        z_full = np.full_like(inp, -1)
        z_full[:L] = z_inp  # HMM states for input phase

        x_true = np.full_like(inp, -1)
        x_true[:L] = x_inp  # true input symbols

        # Boolean mask marking delay timesteps
        delay_mask = np.zeros_like(inp, dtype=bool)
        delay_mask[L+D+1:] = True  # mark delay period only

        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(z_full, dtype=torch.long),
            torch.tensor(x_true, dtype=torch.long),
            torch.tensor(delay_mask, dtype=torch.bool)
        )


class GaussianTask:
    """
    A simple data generator for creating data from multidimensional Gaussians with different means and same scale
    """

    def __init__(self, emb_dim, n_gaussians, locs, scales, labels=None):
        self.emb_dim = emb_dim
        self.n_gaussians = n_gaussians
        self.locs = np.array(locs)
        self.scales = scales
        self.labels = labels if labels is not None else list(range(n_gaussians))

    def create(self, n_samples):
        x = []
        y = []
        for i in range(self.n_gaussians):
            x.append(torch.randn(n_samples, self.emb_dim) * self.scales[i] + self.locs[i])
            y.append(torch.ones(n_samples) * self.labels[i])

        return torch.vstack(x), torch.hstack(y)[:, None]

    def project_data(self, x):
        if self.emb_dim == 1:
            return x
        proj_vec = np.repeat(self.locs[1], self.emb_dim).astype(float) - np.repeat(self.locs[0], self.emb_dim).astype(
            float)
        proj_vec /= np.linalg.norm(proj_vec).astype(float)  # get normalized vector for projection
        return x @ proj_vec[:, None]

    def get_centers_grid(self, n_samples):
        alphas = np.linspace(0, 1, n_samples)
        loc0 = self.locs[0][None] - 3 * self.scales[0]
        loc1 = self.locs[1][None] + 3 * self.scales[1]
        dist = loc1 - loc0
        grid_x = loc0 + alphas[:, None] * dist
        grid_x = np.tile(grid_x, (1, self.n_gaussians))
        return grid_x
