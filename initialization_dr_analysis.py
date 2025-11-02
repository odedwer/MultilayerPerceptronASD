from config import RNNConfig
from models import LSTMWithGateBias
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

RESULTS_DIR = "results/initialization_dr_analysis"
SIGMA_VALS = [0.001,0.01,0.1,1.0,10.0]
N_UNITS = 4
DATA_TYPE = "impulse"
os.makedirs(RESULTS_DIR, exist_ok=True)
for seed in [1,2,3]:
    for gates, name in zip([("cell",),("input","output"),("input","cell","forget","output")], ["cell gate","input and output gates", "all gates"]):
        cfg = RNNConfig()
        cfg.K_symbols = 6
        cfg.gates_dr = gates
        inp_npy = np.zeros((8,100), dtype=int)
        if DATA_TYPE == "impulse":
            inp_npy[:,0] = np.arange(8)
        elif DATA_TYPE == "random":
            rng = np.random.default_rng(seed)
            inp_npy = rng.integers(0,cfg.K_symbols, size=(8,100))
        elif DATA_TYPE == "constant":
            inp_npy[:, :] = np.arange(8)
        fig, axes_grid = plt.subplots(len(SIGMA_VALS),N_UNITS, figsize=(2 + 4 * N_UNITS,5*len(SIGMA_VALS)))
        plt.suptitle(f"DR in {name}", fontsize=40)
        for sigma,axes in zip(SIGMA_VALS, axes_grid):
            cfg.input_gate_bias_std = sigma
            np.random.seed(seed)
            torch.manual_seed(seed)
            rnn = LSTMWithGateBias(8,4,512,cfg)
            y,h = rnn.forward(torch.from_numpy(inp_npy))
            hn = h.detach().numpy()
            for hidx in range(N_UNITS):
                plt.sca(axes[hidx])
                plt.title(f"sigma: {sigma}, Hidden neuron #{hidx+1}")
                for i in range(hn.shape[0]):
                    plt.plot(hn[i,:,hidx])
        plt.savefig(os.path.join(RESULTS_DIR, f"init_dynamics_{name.replace(' ', '_')}_seed{seed}.svg"))
        plt.savefig(os.path.join(RESULTS_DIR,f"init_dynamics_{name.replace(' ','_')}_seed{seed}.png", dpi=400))
        plt.close()