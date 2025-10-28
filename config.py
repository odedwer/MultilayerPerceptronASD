# config.py
from dataclasses import dataclass, asdict, replace
import torch


@dataclass
class RNNConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_id: str = None

    # HMM / Task
    M_states: int = 5
    K_symbols: int = 12
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
    emb_dim: int = 16
    use_onehot: bool = False
    hidden_size: int = 128
    num_layers: int = 1
    bias_means: dict = None
    input_gate_bias_std: float = None
    freeze_all_biases: bool = False
    freeze_input_gate_bias_only: bool = False

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    grad_clip: float = 5.0
    test_eval_interval = 10

    # Probes
    probe_epochs: int = 10
    probe_lr: float = 5e-3
    probe_batch_size: int = 256
    pca_sample_size:int = 5000
    probe_sample_size:int = 2000

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def default_bias_means():
    return {"i": 0.0, "f": 1.0, "g": 0.0, "o": 0.0}
