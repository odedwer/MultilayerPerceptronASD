import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
W_SCALE = np.sqrt(1. * (2 / HIDDEN_SIZE))
SCALE = 1
LOC = 2
OPTIM_TYPE = "SGD"
# %%
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, dg, grid, dataloader = create_dataset(INPUT_SIZE, NUM_SAMPLES, LOC, SCALE, 2)

    opt = optim.Adam if OPTIM_TYPE == "Adam" else optim.SGD
    # train the models
    (model_low_bias, resps_low_bias,
     params_low_bias, pcov_low_bias, inp_dist, pretrain_distances) = train_model(INPUT_SIZE, HIDDEN_SIZE, N_HIDDEN,
                                                                                 OUTPUT_SIZE, W_SCALE, B_SCALE, X_train,
                                                                                 y_train, X_test,
                                                                                 y_test, dataloader, dg, grid, opt,
                                                                                 NUM_EPOCHS)
    (model_high_bias, resps_high_bias,
     params_high_bias, pcov_high_bias, _, pretrain_distances_wide) = train_model(INPUT_SIZE, HIDDEN_SIZE, N_HIDDEN,
                                                                                 OUTPUT_SIZE, W_SCALE, B_SCALE_HIGH,
                                                                                 X_train, y_train,
                                                                                 X_test, y_test, dataloader, dg, grid,
                                                                                 opt,
                                                                                 NUM_EPOCHS)
    # for layer, distances in pretrain_distances.items():
    #     if 'sigmoid' in layer or "activation_func" in layer.lower():
    #         plt.figure()
    #         if 'activation_func' in layer.lower():
    #             plt.title("Layer " + layer[-1])
    #         else:
    #             plt.title(layer)
    #         plt.scatter(inp_dist, distances, label=rf"$\sigma_b={B_SCALE}$", alpha=0.5)
    #         plt.scatter(inp_dist, pretrain_distances_wide[layer], label=rf"$\sigma_b={B_SCALE_HIGH}$", alpha=0.5)
    #         plt.legend(fontsize=15)
    #         plt.xlabel('Input distance')
    #         plt.ylabel('Layer distance')
    #     plt.show()
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
    pdf = PdfPages(f"input output distances {OPTIM_TYPE}.pdf")
    for layer, distances in layer_distances.items():
        if 'activation_func' in layer.lower():
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
            fig.savefig(f"input output distances {layer}.pdf")
            pdf.savefig(fig)
            plt.show()
            plt.close(fig)
    pdf.close()
    # %%
    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 4, width_ratios=[1.2, 1, 1.2, 0.075], height_ratios=[0.5, 0.5])  # , 0.85
    # gs = GridSpec(2, 9, width_ratios=[1, 1, 0.5,0.2, 1, 1, 1, 0.2, 0.1], height_ratios=[0.5, 0.5]) #, 0.85
    l1_ax = fig.add_subplot(gs[0, 0])
    l2_ax = fig.add_subplot(gs[1, 0], sharex=l1_ax)
    slope_ax = fig.add_subplot(gs[0, 1])
    thresh_var_ax = fig.add_subplot(gs[1, 1], sharex=slope_ax)
    learned_func_low_bias_ax = fig.add_subplot(gs[0, 2])
    learned_func_high_bias_ax = fig.add_subplot(gs[1, 2])
    cax_learned_func = fig.add_subplot(gs[0:2, 3])
    # slope_ax = fig.add_subplot(gs[0, :3])
    # thresh_var_ax = fig.add_subplot(gs[1, :3], sharex=slope_ax)
    # learned_func_low_bias_ax = fig.add_subplot(gs[0, 4:-1])
    # cax_learned_func = fig.add_subplot(gs[0:2, -1])
    # learned_func_high_bias_ax = fig.add_subplot(gs[1, 4:-1], sharex=learned_func_low_bias_ax)
    # decision_bound_low_bias_ax = fig.add_subplot(gs[2, :4])
    # decision_bound_high_bias_ax = fig.add_subplot(gs[2, 4:-1], sharey=decision_bound_low_bias_ax)
    # cax_decision_bound = fig.add_subplot(gs[2, -1])
    # plot the change in slope
    input_dist_bins = np.linspace(inp_dist.min(), inp_dist.max(), 501)
    input_dist_idx = np.digitize(inp_dist, input_dist_bins)

    avg_dist_l1 = np.array(
        [pretrain_distances["activation_func1"][input_dist_idx == i].mean()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    avg_dist_l4 = np.array(
        [pretrain_distances["activation_func4"][input_dist_idx == i].mean()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    avg_dist_l1_wide = np.array(
        [pretrain_distances_wide["activation_func1"][input_dist_idx == i].mean()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in
         range(1, len(input_dist_bins))]
    )
    avg_dist_l4_wide = np.array(
        [pretrain_distances_wide["activation_func4"][input_dist_idx == i].mean()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in
         range(1, len(input_dist_bins))]
    )

    std_dist_l1 = np.array(
        [pretrain_distances["activation_func1"][input_dist_idx == i].std()
         if (input_dist_idx == i).sum() > 0 else np.nan
         for i in range(1, len(input_dist_bins))]
    )
    std_dist_l4 = np.array(
        [pretrain_distances["activation_func4"][input_dist_idx == i].std()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    std_dist_l1_wide = np.array(
        [pretrain_distances_wide["activation_func1"][input_dist_idx == i].std()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in range(1, len(input_dist_bins))]
    )
    std_dist_l4_wide = np.array(
        [pretrain_distances_wide["activation_func4"][input_dist_idx == i].std()
         if (input_dist_idx == i).sum() > 0 else np.nan for i in range(1, len(input_dist_bins))]
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


        leg = axes.legend([l1,l2], [label,label + " high variance"],markerscale=10, fontsize=12)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

        axes.set_xlabel('Input distance')
        axes.set_ylabel(f'{label} distance')

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
    # # plot decision boundary with training data
    # c_low = plot_decision_boundary(X_train, y_train, model_low_bias, decision_bound_low_bias_ax)
    # c_high = plot_decision_boundary(X_train, y_train, model_high_bias, decision_bound_high_bias_ax)
    # decision_bound_low_bias_ax.set_xlabel("$x_1$")
    # decision_bound_low_bias_ax.set_ylabel("$x_2$")
    # decision_bound_high_bias_ax.set_xlabel("$x_1$")
    # fig.colorbar(c_low, cax=cax_decision_bound, orientation='vertical')
    # cax_decision_bound.set_ylabel("$P(C=1)$")
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.925, bottom=0.075, left=0.05, right=0.95)
    # fig.text(0.025, 0.95, 'A', fontsize=20, fontweight='bold')
    # fig.text(0.375, 0.95, 'C', fontsize=20, fontweight='bold')
    # fig.text(0.025, 0.68, 'B', fontsize=20, fontweight='bold')
    # fig.text(0.375, 0.68, 'D', fontsize=20, fontweight='bold')
    fig.text(0.04, 0.955, 'A', fontsize=24, fontweight='bold')
    fig.text(0.04, 0.465, 'B', fontsize=24, fontweight='bold')
    fig.text(0.33, 0.955, 'C', fontsize=24, fontweight='bold')
    fig.text(0.33, 0.465, 'D', fontsize=24, fontweight='bold')
    fig.text(0.59, 0.955, 'E', fontsize=24, fontweight='bold')
    fig.text(0.59, 0.465, 'F', fontsize=24, fontweight='bold')
    # fig.text(0.025, 0.4, 'E', fontsize=20, fontweight='bold')
    # fig.text(0.475, 0.4, 'F', fontsize=20, fontweight='bold')
    plt.savefig(f"{OPTIM_TYPE} MLP.pdf")
    plt.show()

#     pdf_name = f"{OPTIM_TYPE}_simulations_low_b_{B_SCALE}_high_b_{B_SCALE_HIGH}_w_{W_SCALE}_loc_{LOC}_scale_{SCALE}.pdf"
#     with PdfPages(pdf_name) as pdf:
#         # plot the change in slope
#         fig = plot_change_in_slope(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, NUM_EPOCHS)
#         pdf.savefig(fig)
#         plt.show()
#         # plot the x value for which the sigmoid crosses 0.5
#         fig = plot_km(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, NUM_EPOCHS)
#         pdf.savefig(fig)
#         plt.show()
#
#         # plot the diff over epochs for each model for slopes and thresholds
#         fig = plot_learning_speed(params_low_bias, params_high_bias, NUM_EPOCHS)
#         pdf.savefig(fig)
#         plt.show()
#
#         # Plot a sliding window of the variance
#         fig = plot_variance_sliding_window(params_low_bias, params_high_bias)
#         pdf.savefig(fig)
#         plt.show()
#
#         # Plot the resps, from before training until after training colored by epoch on a scale from 0 (red) to num_epochs (blue)
#         fig_low_bias = plot_decision_throught_learning(grid, resps_low_bias, X_train, y_train, dg)
#         fig_low_bias.suptitle("Low variance in bias")
#         pdf.savefig(fig_low_bias)
#         plt.show()
#
#         fig_high_bias = plot_decision_throught_learning(grid, resps_high_bias, X_train, y_train, dg)
#         fig_high_bias.suptitle("High variance in bias")
#         pdf.savefig(fig_high_bias)
#         plt.show()
#
#         # plot decision boundary with training data
#         fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(12, 5))
#         c_low = plot_decision_boundary(X_train, y_train, model_low_bias, ax_low, "Low variance in bias")
#         c_high = plot_decision_boundary(X_train, y_train, model_high_bias, ax_high, "High variance in bias")
#         fig.colorbar(c_low, ax=[ax_low, ax_high], orientation='vertical')
#         plt.suptitle("Decision boundary")
#         pdf.savefig(fig)
#         plt.show()
#
# # anim_low_bias = animate_decision_through_learning("low bias", grid, resps_low_bias, X_train, y_train, dg)
# # anim_high_bias = animate_decision_through_learning("high bias", grid, resps_high_bias, X_train, y_train, dg)
