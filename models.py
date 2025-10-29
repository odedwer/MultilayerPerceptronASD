import torch
import torch.nn as nn
nn.LSTM

# Define the MLP model
class MLP(nn.Module):
    """
    A simple MLP model with ReLU activation and Sigmoid output, that can be reinitialized and set to specific bias scale
    """

    def __init__(self, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale):
        super(MLP, self).__init__()
        self._layers = nn.Sequential()
        self._layers.add_module('fc1', nn.Linear(input_size, hidden_size, bias=True))
        self._layers.add_module('activation_func1', nn.Tanh())
        self.w_scale = w_scale
        self.b_scale = b_scale
        for i in range(n_hidden):
            self._layers.add_module(f'fc{i + 2}', nn.Linear(hidden_size, hidden_size, bias=True))
            self._layers.add_module(f'activation_func{i + 2}', nn.Tanh())
        self._layers.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=True))
        self._layers.add_module('sigmoid', nn.Sigmoid())
        self._handles = []
        # self.reinitialize()

    def set_activations_hook(self, activations):
        def hook_generator(name, activations):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()

            return hook

        self._handles = []
        for name, m in self._layers.named_modules():
            self._handles.append(m.register_forward_hook(hook_generator(name, activations)))

    def remove_activations_hook(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def get_out_activation(self):
        return self._layers[-1]

    def forward(self, x):
        return self._layers(x)

    def reinitialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, self.w_scale)
                nn.init.normal_(m.bias, 0, self.b_scale)


# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class LSTMWithGateBias(nn.Module):
    """
    LSTM model that uses an embedding layer for symbol inputs.

    - Input tokens are integer indices: 0..K_symbols-1 (symbols), K_symbols (blank), K_symbols+1 (go)
    - Uses nn.Embedding instead of one-hot encodings.
    - Only the input-gate bias is initialized differently (mean/std from cfg).
    - All other weights and biases remain at PyTorch defaults.
    """

    def __init__(self, input_dim, emb_dim, hidden_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim  # +2 for blank and go tokens
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size

        # Embedding for all tokens (symbols + blank + go)
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Standard LSTM — no custom weight initialization except bias_ih for input gate
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=getattr(cfg, "num_layers", 1),
            batch_first=True,
        )

        # Output layer (predict over K_symbols, not including blank/go)
        self.readout = nn.Linear(hidden_size, cfg.K_symbols)

        # Custom initialization for input gate bias only
        self._init_input_gate_bias()

    # ----------------------------------------------------------------------
    def _init_input_gate_bias(self):
        """Modify only the input gate bias (first quarter of bias_ih)."""
        std = getattr(self.cfg, "input_gate_bias_std", 0.0)
        mean = getattr(self.cfg, "input_gate_bias_mean", 0.0)
        freeze = getattr(self.cfg, "freeze_all_biases", False)

        with torch.no_grad():
            for name, p in self.lstm.named_parameters():
                print(name)
                print(p.shape)
                if "bias_ih" in name or 'bias_hh' in name:
                    H = self.hidden_size
                    # LSTM gates: input, forget, cell, output
                    i_gate = slice(0, H)
                    h_gate = slice(3 * H, 4 * H)
                    p[H:2 * H].fill_(1.0)  # forget gate mean
                    if std and std > 0:
                        p[i_gate].normal_(mean, std)
                        p[h_gate].normal_(mean, std)

        if freeze:
            for name, p in self.lstm.named_parameters():
                if "bias" in name:
                    p.requires_grad_(False)

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x: LongTensor of shape (batch, seq_len)
           Tokens: 0..K_symbols-1 = real symbols
                    K_symbols     = blank token
                    K_symbols+1   = go cue
        """
        # Embed the sequence
        emb = self.embedding(x)  # (B, T, emb_dim)

        # add a last dimension of 1 to x
        # x = x.unsqueeze(-1)


        # x_float = x.float()

        # Forward through LSTM
        h, _ = self.lstm(emb)

        # Predict output logits
        y = self.readout(h)  # (B, T, K_symbols)

        return y, h


import torch
import torch.nn as nn


class RNNWithGateBias(nn.Module):
    """
    Standard RNN model that uses an embedding layer for symbol inputs.

    - Input tokens are integer indices: 0..K_symbols-1 (symbols), K_symbols (blank), K_symbols+1 (go)
    - Uses nn.Embedding instead of one-hot encodings.
    - Only the input-to-hidden bias (bias_ih) is initialized differently (mean/std from cfg).
    - All other weights and biases remain at PyTorch defaults.
    """

    def __init__(self, input_dim, emb_dim, hidden_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim  # +2 for blank and go tokens
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size

        # Embedding for all tokens (symbols + blank + go)
        # self.embedding = nn.Embedding(input_dim, emb_dim)

        # Standard RNN
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=getattr(cfg, "num_layers", 1),
            batch_first=True,
            # nonlinearity='tanh' is the default
        )

        # Output layer (predict over K_symbols, not including blank/go)
        self.readout = nn.Linear(hidden_size, cfg.K_symbols)

        # Custom initialization for input-to-hidden bias
        self._init_bias()

    # ----------------------------------------------------------------------
    def _init_bias(self):
        """Modify only the input-to-hidden bias (bias_ih)."""
        std = getattr(self.cfg, "input_gate_bias_std", 0.0)
        mean = getattr(self.cfg, "input_gate_bias_mean", 0.0)
        freeze = getattr(self.cfg, "freeze_all_biases", False)

        with torch.no_grad():
            for name, p in self.rnn.named_parameters():
                if "bias_ih" in name or "bias_hh" in name:
                    if std and std > 0:
                        nn.init.normal_(p, mean, std)

        if freeze:
            for name, p in self.rnn.named_parameters():
                if "bias" in name:
                    p.requires_grad_(False)

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x: LongTensor of shape (batch, seq_len)
           Tokens: 0..K_symbols-1 = real symbols
                    K_symbols     = blank token
                    K_symbols+1   = go cue
        """
        # Embed the sequence
        # emb = self.embedding(x)  # (B, T, emb_dim)
        # 1. One-hot encode the input tensor 'x'
        # self.input_dim (e.g., K_symbols + 2) is the number of classes
        x_one_hot = torch.nn.functional.one_hot(x, num_classes=self.cfg.K_symbols+2)

        # 2. Cast the one-hot tensor to float.
        # 'autocast' will automatically convert this to torch.float16 (Half)
        x_float = x_one_hot.float()

        # Forward through RNN
        # h contains all hidden states for the sequence
        h, _ = self.rnn(x_float)

        # Predict output logits
        y = self.readout(h)  # (B, T, K_symbols)

        return y, h
