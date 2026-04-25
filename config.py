# config.py
from dataclasses import dataclass, asdict, replace
import torch

from models import LSTMWithGateBias, RNNWithGateBias
import json


@dataclass
class RNNConfig:
    cmd: str = ""
    desc: str = "Default config"
    seed: int = 97
    data: str = "hmm"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_id: str = None

    # HMM / Task
    M_states: int = 5
    K_symbols: int = 12
    word_len: int = 4  # only used if data == "words"
    symbol_noise_prob: float = 0.0  # only used if data == "words"
    L_input: int = 12
    D_delay: int = 18
    s_transitions: int = 2
    s_emissions: int = 4
    flip_prob: float = 0.00
    ood_rewire_frac: float = 0.4
    n_train: int = 15000
    n_val: int = 500
    n_test: int = 2000

    # Model
    name: str = "LSTM"
    model: [LSTMWithGateBias, RNNWithGateBias] = LSTMWithGateBias
    emb_dim: int = 4
    use_onehot: bool = False
    hidden_size: int = 128
    num_layers: int = 1

    # Bias Initialization
    bias_means: dict = None # dict of gate names to means
    bias_std: dict = None # dict of gate names to stds
    global_bias_mean: float = 0.0
    global_bias_std: float = 0.0

    input_gate_bias_std: float = None
    input_gate_bias_mean: float = 0.0
    gates_dr: tuple = ("input", "forget", "cell", "output") # apply dynamic range to all gates
    freeze_all_biases: bool = False
    freeze_input_gate_bias_only: bool = False

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    grad_clip: float = 2.0
    test_eval_interval = 10

    # Probes
    probe_epochs: int = 10
    probe_lr: float = 5e-3
    probe_batch_size: int = 256
    pca_sample_size:int = 5000
    probe_sample_size:int = 2000

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def dump(self, path):
        with open(path, 'w') as f:
            data = asdict(self)
            data = {k: (v.__name__ if callable(v) else v) for k, v in data.items()}
            json.dump(data, f, indent=4)


def default_bias_means():
    return {"input": 0.0, "forget": 1.0, "cell": 0.0, "output": 0.0}
