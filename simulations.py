import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from sympy.polys.polyoptions import Gaussian
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale):
        super(MLP, self).__init__()
        self._layers = nn.Sequential()
        self._layers.add_module('fc1', nn.Linear(input_size, hidden_size, bias=True))
        self.relu = nn.ReLU()
        self._layers.add_module('relu', self.relu)
        self.w_scale = w_scale
        self.b_scale = b_scale
        for i in range(n_hidden):
            self._layers.add_module(f'fc{i + 2}', nn.Linear(hidden_size, hidden_size, bias=True))
            self._layers.add_module(f'relu{i + 2}', self.relu)
        self._layers.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=True))
        self._layers.add_module('sigmoid', nn.Sigmoid())
        # self.reinitialize()

    def get_out_activation(self):
        return self._layers[-1]

    def forward(self, x):
        return self._layers(x)

    def reinitialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=self.w_scale)
                nn.init.uniform_(m.bias, -self.b_scale, self.b_scale)


class DataGenerator:

    def __init__(self, emb_dim, n_gaussians, locs, scales, labels=None):
        self.emb_dim = emb_dim
        self.n_gaussians = n_gaussians
        self.locs = np.array(locs)
        self.scales = scales
        self.labels = labels if labels is not None else list(range(n_gaussians))

    def create(self, n_samples):
        X = []
        y = []
        for i in range(self.n_gaussians):
            X.append(torch.randn(n_samples, self.emb_dim) * self.scales[i] + self.locs[i])
            y.append(torch.ones(n_samples) * self.labels[i])

        return torch.vstack(X), torch.hstack(y)[:, None]

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


def sigmoid(x, thresh, slope, shift):
    return (1 / (1 + np.exp(-slope * (x - thresh)))) + shift


# %%
# Generate some dummy data
input_size = 2
hidden_size = 100
n_hidden = 3
output_size = 1
num_samples = 100
b_scale = 1.0
w_scale = 1.0
scale = 1
loc = 2
# %%
np.random.seed(0)
dg = DataGenerator(input_size, 2, [-loc, loc], [scale, scale])

X, y = dg.create(num_samples)

grid = dg.get_centers_grid(150)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# %%
# Initialize the model, loss function, and optimizer
def train_model(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                X_test, y_test, dataloader):
    resps = []
    fit_results = []
    fit_pcov = []
    x = dg.project_data(grid)
    model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale)
    model.reinitialize(seed=42)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.eval()
    with torch.no_grad():
        resps.append(model(torch.tensor(grid, dtype=torch.float32)).detach().numpy())
    model.train()
    # Training loop
    num_epochs = 150
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            resp = model(torch.tensor(grid, dtype=torch.float32)).detach().numpy()
            resps.append(resp)
            # fit the sigmoid function to the resp

            try:
                params, pcov = curve_fit(sigmoid, np.squeeze(x), np.squeeze(resp))
                fit_pcov.append(pcov)
                fit_results.append(params)
            except RuntimeError:
                fit_pcov.append(np.full((3, 3), np.nan))
                fit_results.append(np.full(3, np.nan))

        model.train()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
    return resps, fit_results, fit_pcov


# test the model
resps_low_bias, params_low_bias, pcov_low_bias = train_model(input_size, hidden_size, n_hidden, output_size, w_scale,
                                                             b_scale, X_test, y_test,
                                                             dataloader)
resps_high_bias, params_high_bias, pcov_high_bias = train_model(input_size, hidden_size, n_hidden, output_size, w_scale,
                                                                20., X_test, y_test, dataloader)
params_low_bias = np.array(params_low_bias)
params_high_bias = np.array(params_high_bias)
pcov_low_bias = np.array(pcov_low_bias)
pcov_high_bias = np.array(pcov_high_bias)

# %% plot the change in slope
plt.figure(figsize=(5, 5))
plt.plot(range(150), params_low_bias[:, 1], label="Low Bias", color='blue', markersize=1)
plt.fill_between(range(150), params_low_bias[:, 1] - np.sqrt(pcov_low_bias[:, 1, 1]),
                 params_low_bias[:, 1] + np.sqrt(pcov_low_bias[:, 1, 1]), alpha=0.5, color='blue')
plt.plot(range(150), params_high_bias[:, 1], label="High Bias", color='red', markersize=1)
plt.fill_between(range(150), params_high_bias[:, 1] - np.sqrt(pcov_high_bias[:, 1, 1]),
                 params_high_bias[:, 1] + np.sqrt(pcov_high_bias[:, 1, 1]), alpha=0.5, color='red')
plt.title(f"Slope over training, num_samples={num_samples}")
plt.legend()
plt.show()


# %% Plot the resps, from before training until after training colored by epoch on a scale from 0 (red) to num_epochs (blue)

def plot_decision_throught_learning(grid, resps, X_train, y_train, generator: DataGenerator):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=0, vmax=len(resps))
    for i in range(len(resps)):
        ax.plot(grid, resps[i], color=cmap(norm(i)))
    ax.scatter(generator.project_data(X_train), ((y_train - 0.5) * 1.05) + 0.5, c=y_train, s=5, alpha=0.5)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    return fig


fig_low_bias = plot_decision_throught_learning(grid, resps_low_bias, X_train, y_train, dg)
fig_low_bias.suptitle("Low Bias")
plt.show()

fig_high_bias = plot_decision_throught_learning(grid, resps_high_bias, X_train, y_train, dg)
fig_high_bias.suptitle("High Bias")
plt.show()


# %%

def animate_decision_through_learning(name, grid, resps, X_train, y_train, generator: DataGenerator):
    # animate using func_anim
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=0, vmax=len(resps))
    x_train_proj = generator.project_data(X_train)
    proj_grid = generator.project_data(grid)

    ax.scatter(x_train_proj, ((y_train - 0.5) * 1.05) + 0.5, c=y_train, s=5, alpha=0.5)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    line, = ax.plot([], [])
    ax.set_xlim(x_train_proj.min() * 1.1, x_train_proj.max() * 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Epoch 0")
    ax.set_xlabel("Projection")
    ax.set_ylabel("Output")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(proj_grid, resps[i])
        # set the data color
        line.set_color(cmap(norm(i)))
        ax.set_title(f"Epoch {i}")
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=tqdm(range(len(resps))), interval=500, blit=True)
    anim.save(f"{name}.mp4", writer="ffmpeg")

    return anim


anim_low_bias = animate_decision_through_learning("low bias", grid, resps_low_bias, X_train, y_train, dg)
anim_high_bias = animate_decision_through_learning("high bias", grid, resps_high_bias, X_train, y_train, dg)

# %%
import scipy.stats

proj_grid = dg.project_data(grid)
pdf1 = scipy.stats.norm(loc=-2, scale=1).pdf(proj_grid)
plt.plot(proj_grid, pdf1)
pdf2 = scipy.stats.norm(loc=2, scale=1).pdf(proj_grid)
plt.plot(proj_grid, pdf2)
plt.plot(proj_grid, pdf2 / (pdf1 + pdf2))
plt.show()

optimal_sigmoid_params, _ = curve_fit(sigmoid, np.squeeze(proj_grid), np.squeeze(pdf2 / (pdf1 + pdf2)))
