import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import torch

from utils import *

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 14
# %%
# Parameters
NUM_EPOCHS = 150
INPUT_SIZE = 2
HIDDEN_SIZE = 600
N_HIDDEN = 4
OUTPUT_SIZE = 1
NUM_SAMPLES = 500
B_SCALE = .1
B_SCALE_HIGH = 5.
W_SCALE = np.sqrt(5. * (2 / HIDDEN_SIZE))
SCALE = 1
LOC = 2
OPTIM_TYPE = "SGD"


def run_experiment(
    input_size, num_samples, loc, scale, n_hidden, hidden_size, output_size,
    w_scale, b_scale, b_scale_high, num_epochs, opt_type, seed, dataset_id, run_id, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs("figures", exist_ok=True)
    X_train, X_test, y_train, y_test, dg, grid, dataloader = create_gaussian_dataset(
        input_size, num_samples, loc, scale, 2, seed=seed, device=device
    )
    opt = optim.Adam if opt_type == "Adam" else optim.SGD
    (model_low_bias, resps_low_bias,
     params_low_bias, pcov_low_bias, inp_dist, pretrain_distances) = train_mlp_model(
        input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, X_train,
        y_train, X_test, y_test, dataloader, dg, grid, opt, num_epochs, device=device)
    (model_high_bias, resps_high_bias,
     params_high_bias, pcov_high_bias, _, pretrain_distances_wide) = train_mlp_model(
        input_size, hidden_size, n_hidden, output_size, w_scale, b_scale_high,
        X_train, y_train, X_test, y_test, dataloader, dg, grid, opt, num_epochs, device=device)

    model_low_bias.eval()
    model_high_bias.eval()

    params_low_bias = np.array(params_low_bias)
    params_high_bias = np.array(params_high_bias)
    pcov_low_bias = np.array(pcov_low_bias)
    pcov_high_bias = np.array(pcov_high_bias)

    activations = {}
    activations_wide = {}
    model_low_bias.set_activations_hook(activations)
    model_high_bias.set_activations_hook(activations_wide)

    model_low_bias(X_train)
    model_high_bias(X_train)
    layer_distances = {k: pairwise_distances(v)[np.triu_indices(X_train.shape[0])] for k, v in activations.items()}
    layer_distances_wide = {k: pairwise_distances(v)[np.triu_indices(X_train.shape[0])] for k, v in
                            activations_wide.items()}
    # pdf = PdfPages(f"figures/input output distances {OPTIM_TYPE}.pdf")
    # for layer, distances in layer_distances.items():
    #     if 'activation_func' in layer.lower():
    #         fig = scatterplot_layer_distances(inp_dist, layer, pretrain_distances, pretrain_distances_wide, pdf)
    #         plt.show()
    #         plt.close(fig)
    # pdf.close()
    # %%
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 4, width_ratios=[1.2, 1, 1.2, 0.075], height_ratios=[0.5, 0.5])  # , 0.85
    # gs = GridSpec(2, 9, width_ratios=[1, 1, 0.5,0.2, 1, 1, 1, 0.2, 0.1], height_ratios=[0.5, 0.5]) #, 0.85
    l1_ax = fig.add_subplot(gs[0, 0])
    l2_ax = fig.add_subplot(gs[1, 0], sharex=l1_ax)
    # subax_l2 = inset_axes(l2_ax, width="40%", height="40%", loc='center right')
    slope_ax = fig.add_subplot(gs[0, 1])
    thresh_var_ax = fig.add_subplot(gs[1, 1], sharex=slope_ax)
    learned_func_low_bias_ax = fig.add_subplot(gs[0, 2])
    learned_func_high_bias_ax = fig.add_subplot(gs[1, 2])
    cax_learned_func = fig.add_subplot(gs[0:2, 3])

    input_dist_bins = np.linspace(inp_dist.min(), inp_dist.max(), 501)
    input_dist_idx = np.digitize(inp_dist, input_dist_bins)

    avg_dist_l1 = np.array(
        [layer_distances["activation_func1"][input_dist_idx == i].mean()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    avg_dist_l4 = np.array(
        [layer_distances["activation_func4"][input_dist_idx == i].mean()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    avg_dist_l1_wide = np.array(
        [layer_distances_wide["activation_func1"][input_dist_idx == i].mean()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in
         range(1, len(input_dist_bins))]
    )
    avg_dist_l4_wide = np.array(
        [layer_distances_wide["activation_func4"][input_dist_idx == i].mean()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in
         range(1, len(input_dist_bins))]
    )

    std_dist_l1 = np.array(
        [layer_distances["activation_func1"][input_dist_idx == i].std()
         if np.sum(input_dist_idx == i) > 0 else np.nan
         for i in range(1, len(input_dist_bins))]
    )
    std_dist_l4 = np.array(
        [layer_distances["activation_func4"][input_dist_idx == i].std()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    std_dist_l1_wide = np.array(
        [layer_distances_wide["activation_func1"][input_dist_idx == i].std()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    std_dist_l4_wide = np.array(
        [layer_distances_wide["activation_func4"][input_dist_idx == i].std()
         if np.sum(input_dist_idx == i) > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )

    for distances, std, distances_wide, std_wide, axes, label in zip(
            [avg_dist_l1, avg_dist_l4],
            [std_dist_l1, std_dist_l4],
            [avg_dist_l1_wide, avg_dist_l4_wide],
            [std_dist_l1_wide, std_dist_l4_wide],
            [l1_ax, l2_ax], ["L1 tanh", "L4 tanh"]
    ):

        l1, = axes.plot(input_dist_bins[1:], distances, label=label, color=NT_COLOR)
        axes.fill_between(input_dist_bins[1:], distances - std, distances + std, color=NT_COLOR, alpha=0.3)
        l2, = axes.plot(input_dist_bins[1:], distances_wide, label=label + " high variance", color=ASD_COLOR)
        axes.fill_between(input_dist_bins[1:], distances_wide - std_wide, distances_wide + std_wide, color=ASD_COLOR,
                          alpha=0.3)


        leg = axes.legend([l1,l2], [label,label + " high variance"],markerscale=10, fontsize=12, loc="lower right")
        for lh in leg.legend_handles:
            lh.set_alpha(1)

        axes.set_xlabel('Input distance')
        axes.set_ylabel(f'{label} distance')
    # subax_l2.plot(input_dist_bins[1:], avg_dist_l4_wide, color=ASD_COLOR)
    # subax_l2.fill_between(input_dist_bins[1:], avg_dist_l4_wide - std_dist_l4_wide, avg_dist_l4_wide + std_dist_l4_wide, color=ASD_COLOR, alpha=0.3)
    plot_change_in_slope(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, NUM_EPOCHS, ax=slope_ax)
    slope_ax.set_xlabel("Epoch")
    # plot the threshold variance
    plot_learning_speed(params_low_bias, params_high_bias, NUM_EPOCHS, ax_slope=thresh_var_ax)
    thresh_var_ax.set_xlabel("Epoch")
    # plot the resps, from before training until after training colored by epoch on a scale from 0 (red) to num_epochs (blue)
    plot_decision_throught_learning(grid, resps_low_bias, X_train, y_train, dg, ax=learned_func_low_bias_ax,
                                    cax=cax_learned_func)
    plot_decision_throught_learning(grid, resps_high_bias, X_train, y_train, dg, ax=learned_func_high_bias_ax,
                                    cax=cax_learned_func)
    learned_func_high_bias_ax.set_xlabel("Projection unto the separating line")
    learned_func_high_bias_ax.set_ylabel("$P(C=1)$")
    learned_func_low_bias_ax.set_xlabel("Projection unto the separating line")
    learned_func_low_bias_ax.set_ylabel("$P(C=1)$")
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.925, bottom=0.075, left=0.05, right=0.95)
    fig.text(0.04, 0.955, 'A', fontsize=24, fontweight='bold')
    fig.text(0.04, 0.465, 'B', fontsize=24, fontweight='bold')
    fig.text(0.33, 0.955, 'C', fontsize=24, fontweight='bold')
    fig.text(0.33, 0.465, 'D', fontsize=24, fontweight='bold')
    fig.text(0.59, 0.955, 'E', fontsize=24, fontweight='bold')
    fig.text(0.59, 0.465, 'F', fontsize=24, fontweight='bold')
    plt.savefig(f"figures/{OPTIM_TYPE}_MLP_dataset{dataset_id}_run{run_id}.pdf")
    plt.show()


def scatterplot_layer_distances(inp_dist, layer, pretrain_distances, pretrain_distances_wide, pdf=None):
    fig, axes = plt.subplots(1, 1, figsize=(7, 5), rasterized=True)
    fig.suptitle(layer)
    axes.scatter(inp_dist, pretrain_distances[layer], label=layer, alpha=0.1, s=1)
    axes.scatter(inp_dist, pretrain_distances_wide[layer], label=layer + ' high var', alpha=0.3, s=1)
    leg = plt.legend(markerscale=10, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    axes.set_xlabel('Input distance')
    axes.set_ylabel('Layer distance')
    # axes[1].set_title("After training")
    # axes[1].scatter(inp_dist, distances, label=layer, alpha=0.5, s=1)
    # axes[1].scatter(inp_dist, layer_distances_wide[layer], label=layer + ' wide', alpha=0.5, s=1)
    # axes[1].legend()
    # axes[1].set_xlabel('Input distance')
    fig.savefig(f"figures/input output distances {layer}.pdf")
    if pdf is not None:
        pdf.savefig(fig)
    return fig


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_id in range(1,4):
        print("Dataset", dataset_id)
        for run_id in range(1, 4):
            print("Run", run_id)
            run_experiment(
                INPUT_SIZE, NUM_SAMPLES, LOC, SCALE, N_HIDDEN, HIDDEN_SIZE, OUTPUT_SIZE,
                W_SCALE, B_SCALE, B_SCALE_HIGH, NUM_EPOCHS, OPTIM_TYPE, seed=dataset_id*10+run_id,
                dataset_id=dataset_id, run_id=run_id, device=device
            )
# %%