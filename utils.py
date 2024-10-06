from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
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
    """
    A simple MLP model with ReLU activation and Sigmoid output, that can be reinitialized and set to specific bias scale
    """

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

    def set_forward_hook(self, hook_generator):
        for name, m in self._layers.named_modules():
            m.register_forward_hook(hook_generator(name))

    def get_out_activation(self):
        return self._layers[-1]

    def forward(self, x):
        return self._layers(x)

    def reinitialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, 0, self.b_scale)


def forward_hook_generator(name, activations):
    def hook(model, input, output):
        activations[name].append(output.detach().numpy())

    return hook


class DataGenerator:
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


def sigmoid(x, thresh, slope):
    """
    Sigmoid function
    :param x: Input
    :param thresh: Threshold
    :param slope: Slope
    """
    return 1 / (1 + np.exp(-slope * (x - thresh)))


# Initialize the model, loss function, and optimizer
def train_model(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                X_test, y_test, dataloader, dg, grid, optimizer_type=optim.Adam, num_epochs=300):
    """
    Create and train the model using the given arguments
    :param input_size: The dimension of the input
    :param hidden_size: The size of the hidden layers
    :param n_hidden: The number of hidden layers
    :param output_size: The size of the output layer
    :param w_scale: The scale of the weights
    :param b_scale: The scale of the biases
    :param X_test: The test data
    :param y_test: The test labels
    :param dataloader: The DataLoader for the training data
    :param dg: The DataGenerator object
    :param grid: The grid for the centers on which the sigmoid will be fitted
    :param optimizer_type: The optimizer type
    :param num_epochs: The number of epochs for training
    :return: The trained model, the responses, the fitted parameters, and the covariance matrices of the parameters
    """
    resps = []
    fit_results = []
    fit_pcov = []
    activations = defaultdict(list)
    x = dg.project_data(grid)
    model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale)
    model.reinitialize(seed=42)
    model.set_forward_hook(lambda x, a=activations: forward_hook_generator(x, a))
    criterion = nn.BCELoss()
    optimizer = optimizer_type(model.parameters(), lr=0.001)
    model.eval()
    with torch.no_grad():
        resps.append(model(torch.tensor(grid, dtype=torch.float32)).detach().numpy())
    model.train()
    # Training loop
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
                fit_pcov.append(np.full((2, 2), np.nan))
                fit_results.append(np.full(2, np.nan))

        model.train()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
    return model, resps, fit_results, fit_pcov, activations


def animate_decision_through_learning(name, grid, resps, X_train, y_train, generator: DataGenerator):
    """
    Animate the 1D decision boundary through learning
    :param name: The name of the file
    :param grid: The grid on which the sigmoid was fitted
    :param resps: The responses of the model
    :param X_train: The training data
    :param y_train: The training labels
    :param generator: The DataGenerator object, used to project the data to 1D
    """
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


def create_dataset(input_size, num_samples, loc, scale, n_gaussians=2, seed=0):
    """
    Create a dataset using the DataGenerator
    :param input_size: The dimension of the input
    :param num_samples: The number of samples
    :param loc: The means of the Gaussians
    :param scale: The scale of the Gaussians
    :param n_gaussians: The number of Gaussians
    :param seed: The seed for the random number generator
    :return: X_train, X_test, y_train, y_test, The DataGenerator object, grid, training data DataLoader
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    dg = DataGenerator(input_size, n_gaussians, [-loc, loc], [scale, scale])

    X, y = dg.create(num_samples)

    grid = dg.get_centers_grid(150)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X_train, X_test, y_train, y_test, dg, grid, dataloader


# %% Plotting
def plot_decision_throught_learning(grid, resps, X_train, y_train, generator: DataGenerator):
    """
    Plot the decision boundary through learning
    :param grid: The grid on which the response was evaluated
    :param resps: The responses of the model
    :param X_train: The training data
    :param y_train: The training labels
    :param generator: The DataGenerator object
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=0, vmax=len(resps))
    for i in range(len(resps)):
        ax.plot(grid, resps[i], color=cmap(norm(i)))
    ax.scatter(generator.project_data(X_train), ((y_train - 0.5) * 1.05) + 0.5, c=y_train, s=5, alpha=0.5)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    return fig


def plot_change_in_slope(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, num_epochs):
    """
    Plot the change in slope over training
    :param params_low_bias: The fitted parameters of the low bias model sigmoid
    :param params_high_bias: The fitted parameters of the high bias model sigmoid
    :param pcov_low_bias: The covariance matrix of the low bias model sigmoid fit
    :param pcov_high_bias: The covariance matrix of the high bias model sigmoid fit
    :param num_epochs: The number of epochs
    """
    fig = plt.figure(figsize=(5, 5))
    plt.plot(range(num_epochs), params_low_bias[:, 1], label="Low Bias", color='blue', markersize=1)
    plt.fill_between(range(num_epochs), params_low_bias[:, 1] - np.sqrt(pcov_low_bias[:, 1, 1]),
                     params_low_bias[:, 1] + np.sqrt(pcov_low_bias[:, 1, 1]), alpha=0.5, color='blue')
    plt.plot(range(num_epochs), params_high_bias[:, 1], label="High Bias", color='red', markersize=1)
    plt.fill_between(range(num_epochs), params_high_bias[:, 1] - np.sqrt(pcov_high_bias[:, 1, 1]),
                     params_high_bias[:, 1] + np.sqrt(pcov_high_bias[:, 1, 1]), alpha=0.5, color='red')
    plt.title(f"Slope over training, num_samples={num_epochs}")
    plt.legend()
    return fig


def plot_km(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, num_epochs):
    """
    Plot the x value for which the sigmoid crosses 0.5
    :param params_low_bias: The fitted parameters of the low bias model sigmoid
    :param params_high_bias: The fitted parameters of the high bias model sigmoid
    :param pcov_low_bias: The covariance matrix of the low bias model sigmoid fit
    :param pcov_high_bias: The covariance matrix of the high bias model sigmoid fit
    :param num_epochs: The number of epochs
    """
    fig = plt.figure(figsize=(5, 5))
    plt.plot(range(num_epochs // 50, num_epochs), params_low_bias[num_epochs // 50:, 0], label="Low Bias", color='blue',
             markersize=1)
    plt.fill_between(range(num_epochs // 50, num_epochs),
                     params_low_bias[num_epochs // 50:, 0] - np.sqrt(pcov_low_bias[num_epochs // 50:, 0, 0]),
                     params_low_bias[num_epochs // 50:, 0] + np.sqrt(pcov_low_bias[num_epochs // 50:, 0, 0]), alpha=0.5,
                     color='blue')
    plt.plot(range(num_epochs // 50, num_epochs), params_high_bias[num_epochs // 50:, 0], label="High Bias",
             color='red',
             markersize=1)
    plt.fill_between(range(num_epochs // 50, num_epochs),
                     params_high_bias[num_epochs // 50:, 0] - np.sqrt(pcov_high_bias[num_epochs // 50:, 0, 0]),
                     params_high_bias[num_epochs // 50:, 0] + np.sqrt(pcov_high_bias[num_epochs // 50:, 0, 0]),
                     alpha=0.5,
                     color='red')
    plt.title(f"Threshold over training")
    plt.legend()
    return fig


def plot_learning_speed(params_low_bias, params_high_bias, num_epochs):
    """
    Plot the change in slope and threshold over epochs
    :param params_low_bias: The fitted parameters of the low bias model sigmoid
    :param params_high_bias: The fitted parameters of the high bias model sigmoid
    :param num_epochs: The number of epochs
    """
    fig, (ax_slope, ax_epoch) = plt.subplots(1, 2, figsize=(10, 5))
    ax_slope.plot(range(num_epochs - 1), np.diff(params_low_bias[:, 1]), label="Low Bias", color='blue', markersize=1)
    ax_slope.plot(range(num_epochs - 1), np.diff(params_high_bias[:, 1]), label="High Bias", color='red', markersize=1)
    ax_slope.set_title("Slope change speed")
    ax_slope.legend()

    ax_epoch.plot(range(num_epochs // 50, num_epochs - 1), np.diff(params_low_bias[num_epochs // 50:, 0]),
                  label="Low Bias",
                  color='blue', markersize=1)
    ax_epoch.plot(range(num_epochs // 50, num_epochs - 1), np.diff(params_high_bias[num_epochs // 50:, 0]),
                  label="High Bias", color='red', markersize=1)
    ax_epoch.set_title("Threshold change speed")
    ax_epoch.legend()
    return fig


def plot_variance_sliding_window(params_low_bias, params_high_bias):
    """
    Plot the variance of the threshold over a sliding window
    :param params_low_bias: The fitted parameters of the low bias model sigmoid
    :param params_high_bias: The fitted parameters of the high bias model sigmoid
    """
    global fig
    fig, ax = plt.subplots()
    window_size = 10
    low_bias_var = np.lib.stride_tricks.sliding_window_view(params_low_bias[:, 0], window_shape=window_size).var(1)
    high_bias_var = np.lib.stride_tricks.sliding_window_view(params_high_bias[:, 0], window_shape=window_size).var(1)
    ax.plot(range(1, low_bias_var.size + 1), low_bias_var, label="Low Bias", color='blue')
    ax.plot(range(1, high_bias_var.size + 1), high_bias_var, label="High Bias", color='red')
    ax.set_title("Threshold variance over training")
    ax.set_xlabel(f"Window, #epochs per window={window_size}")
    ax.set_ylabel("Variance")
    ax.legend()


def plot_decision_boundary(X_train, y_train, model, ax, title):
    """
    Plot the decision boundary with training data
    :param X_train: The training data
    :param y_train: The training labels
    :param model: The model
    :param ax: The axis
    """
    train_x_c0 = X_train[y_train.flatten() == 0, :]
    train_x_c1 = X_train[y_train.flatten() == 1, :]

    # create a 2d points grid for the entire space
    n_point_in_grid = 201
    linspace = np.linspace(X_train.min() - 1, X_train.max() + 1, n_point_in_grid)
    grid_x, grid_y = np.meshgrid(linspace, linspace)

    grid_input = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    classifications = model(torch.tensor(grid_input, dtype=torch.float32)).detach().numpy().reshape(
        n_point_in_grid, n_point_in_grid)

    c = ax.pcolormesh(linspace, linspace, classifications, vmin=0, vmax=1, alpha=0.5, cmap='coolwarm')
    ax.scatter(*train_x_c1.T)
    ax.scatter(*train_x_c0.T)
    # plot the countour of the decision boundary by coloring the sep_grid points according to the model response
    ax.set_title(title)
    return c
