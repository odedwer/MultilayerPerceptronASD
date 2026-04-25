import torch
import torch.nn as nn
import torch.nn.functional as F

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
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# (Redundant imports removed for brevity)

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

        # Custom initialization for all gates
        self._init_biases()

    # ----------------------------------------------------------------------
    def _init_biases(self):
        """Modify all gates based on cfg."""
        freeze = getattr(self.cfg, "freeze_all_biases", False)
        dr_gates = getattr(self.cfg, "gates_dr", ("input", "forget", "cell", "output"))

        # Get global defaults
        g_mean = getattr(self.cfg, "global_bias_mean", 0.0)
        g_std = getattr(self.cfg, "global_bias_std", 0.0)

        # Get gate-specific defaults (default to empty dict if None or missing)
        g_means = getattr(self.cfg, "bias_means", None) or {}
        g_stds = getattr(self.cfg, "bias_std", None) or {}

        with torch.no_grad():
            for name, p in self.lstm.named_parameters():
                if "bias_ih" in name or 'bias_hh' in name:
                    H = self.hidden_size

                    # 1. Start with global mean/std
                    p.fill_(g_mean)
                    if g_std > 0:
                        p.normal_(g_mean, g_std)

                    # 2. Apply gate-specific overrides if provided in dr_gates
                    # LSTM gates in bias_ih: [i, f, g, o]
                    if dr_gates:
                        # Mapping gate names to slices
                        gate_map = {
                            "input": slice(0, H),
                            "forget": slice(H, 2 * H),
                            "cell": slice(2 * H, 3 * H),
                            "output": slice(3 * H, 4 * H)
                        }

                        for gate_name in dr_gates:
                            if gate_name in gate_map:
                                s = gate_map[gate_name]
                                # Use gate-specific mean/std if available, else global
                                m = g_means.get(gate_name, g_mean)
                                std = g_stds.get(gate_name, g_std)

                                p[s].fill_(m)
                                if std > 0:
                                    p[s].normal_(m, std)

        if freeze:
            for name, p in self.lstm.named_parameters():
                if "bias" in name:
                    p.requires_grad_(False)

    # ----------------------------------------------------------------------
    def step(self, x_t, h_t, c_t):
        """
        Single step of the LSTM.
        :param x_t: Input indices (B, 1)
        :param h_t: Hidden state (B, H)
        :param c_t: Cell state (B, H)
        :return: h_next (B, H), (h_next (B, H), c_next (B, H))
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        emb = self.embedding(x_t)  # (B, 1, emb_dim)
        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(0)
            c_t = c_t.unsqueeze(0)
        out, (h_next, c_next) = self.lstm(emb, (h_t, c_t))
        h_next = h_next.squeeze(0)
        c_next = c_next.squeeze(0)
        return h_next, (h_next, c_next)

    def forward(self, x):
        emb = self.embedding(x)              # (B, T, emb_dim)
        H, (h_n, c_n) = self.lstm(emb)      # H: (B, T, hidden_size)
        logits = self.readout(H)             # (B, T, K_symbols)
        return logits, H


class RNNWithGateBias(nn.Module):
    """
    Standard RNN model that uses an embedding layer for symbol inputs.

    - Input tokens are integer indices: 0..K_symbols-1 (symbols), K_symbols (blank), K_symbols+1 (go)
    - Uses nn.Embedding instead of one-hot encodings.
    - All biases are initialized according to cfg.
    """

    def __init__(self, input_dim, emb_dim, hidden_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim  # +2 for blank and go tokens
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size

        # Standard RNN
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=getattr(cfg, "num_layers", 1),
            batch_first=True,
        )

        # Output layer (predict over K_symbols, not including blank/go)
        self.readout = nn.Linear(hidden_size, cfg.K_symbols)

        # Custom initialization for biases
        self._init_biases()

    # ----------------------------------------------------------------------
    def _init_biases(self):
        """Modify biases based on cfg."""
        freeze = getattr(self.cfg, "freeze_all_biases", False)

        # Get global defaults
        g_mean = getattr(self.cfg, "global_bias_mean", 0.0)
        g_std = getattr(self.cfg, "global_bias_std", 0.0)

        # Get gate-specific defaults (default to empty dict if None or missing)
        g_means = getattr(self.cfg, "bias_means", None) or {}
        g_stds = getattr(self.cfg, "bias_std", None) or {}

        with torch.no_grad():
            for name, p in self.rnn.named_parameters():
                if "bias_ih" in name or "bias_hh" in name:
                    # Start with global
                    p.fill_(g_mean)
                    if g_std > 0:
                        p.normal_(g_mean, g_std)

                    # If specific biases are provided for the RNN (as a single set)
                    if g_means:
                        # We'll just use the first one or assume it' Denotes all
                        p.normal_(next(iter(g_means.values())), next(iter(g_stds.values()), g_std))
                    elif g_stds:
                         p.normal_(g_mean, next(iter(g_stds.values())))

        if freeze:
            for name, p in self.rnn.named_parameters():
                if "bias" in name:
                    p.requires_grad_(False)

    # ----------------------------------------------------------------------
    def step(self, x_t, h_t):
        """
        Single step of the RNN.
        :param x_t: Input indices (B, 1)
        :param h_t: Hidden state (B, H)
        :return: h_next (B, H), h_next (B, H)
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        x_one_hot = F.one_hot(x_t, num_classes=self.cfg.K_symbols + 2).float()  # (B, 1, K+2)
        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(0)
        out, h_next = self.rnn(x_one_hot, h_t)
        h_next = h_next.squeeze(0)
        return h_next, h_next

    def forward(self, x):
        x_one_hot = F.one_hot(x, num_classes=self.cfg.K_symbols + 2).float()  # (B, T, K+2)
        H, h_n = self.rnn(x_one_hot)         # H: (B, T, hidden_size)
        logits = self.readout(H)             # (B, T, K_symbols)
        return logits, H
