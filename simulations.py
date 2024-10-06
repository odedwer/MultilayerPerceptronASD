from matplotlib.backends.backend_pdf import PdfPages
from utils import *

# %%
# Parameters
NUM_EPOCHS = 150
INPUT_SIZE = 2
HIDDEN_SIZE = 200
N_HIDDEN = 3
OUTPUT_SIZE = 1
NUM_SAMPLES = 1000
B_SCALE = 1.0
B_SCALE_HIGH = 5.
W_SCALE = 1.0
SCALE = 1
LOC = 1
OPTIM_TYPE = "SGD"
# %%
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, dg, grid, dataloader = create_dataset(INPUT_SIZE, NUM_SAMPLES, LOC, SCALE, 2)

    opt = optim.Adam if OPTIM_TYPE == "Adam" else optim.SGD
    # train the models
    (model_low_bias, resps_low_bias,
     params_low_bias, pcov_low_bias, low_bias_activations) = train_model(INPUT_SIZE, HIDDEN_SIZE, N_HIDDEN,
                                                                         OUTPUT_SIZE, W_SCALE, B_SCALE, X_test,
                                                                         y_test, dataloader, dg, grid, opt, NUM_EPOCHS)
    (model_high_bias, resps_high_bias,
     params_high_bias, pcov_high_bias, high_bias_activations) = train_model(INPUT_SIZE, HIDDEN_SIZE, N_HIDDEN,
                                                                            OUTPUT_SIZE, W_SCALE, B_SCALE_HIGH,
                                                                            X_test, y_test, dataloader, dg, grid, opt,
                                                                            NUM_EPOCHS)

    model_low_bias.eval()
    model_high_bias.eval()

    params_low_bias = np.array(params_low_bias)
    params_high_bias = np.array(params_high_bias)
    pcov_low_bias = np.array(pcov_low_bias)
    pcov_high_bias = np.array(pcov_high_bias)

    pdf_name = f"{OPTIM_TYPE}_simulations_low_b_{B_SCALE}_high_b_{B_SCALE_HIGH}_w_{W_SCALE}_loc_{LOC}_scale_{SCALE}.pdf"
    with PdfPages(pdf_name) as pdf:
        # plot the change in slope
        fig = plot_change_in_slope(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, NUM_EPOCHS)
        pdf.savefig(fig)
        plt.show()
        # plot the x value for which the sigmoid crosses 0.5
        fig = plot_km(params_low_bias, params_high_bias, pcov_low_bias, pcov_high_bias, NUM_EPOCHS)
        pdf.savefig(fig)
        plt.show()

        # plot the diff over epochs for each model for slopes and thresholds
        fig = plot_learning_speed(params_low_bias, params_high_bias, NUM_EPOCHS)
        pdf.savefig(fig)
        plt.show()

        # Plot a sliding window of the variance
        fig = plot_variance_sliding_window(params_low_bias, params_high_bias)
        pdf.savefig(fig)
        plt.show()

        # Plot the resps, from before training until after training colored by epoch on a scale from 0 (red) to num_epochs (blue)
        fig_low_bias = plot_decision_throught_learning(grid, resps_low_bias, X_train, y_train, dg)
        fig_low_bias.suptitle("Low Bias")
        pdf.savefig(fig_low_bias)
        plt.show()

        fig_high_bias = plot_decision_throught_learning(grid, resps_high_bias, X_train, y_train, dg)
        fig_high_bias.suptitle("High Bias")
        pdf.savefig(fig_high_bias)
        plt.show()

        # plot decision boundary with training data
        fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(12, 5))
        c_low = plot_decision_boundary(X_train, y_train, model_low_bias, ax_low, "Low Bias")
        c_high = plot_decision_boundary(X_train, y_train, model_high_bias, ax_high, "High Bias")
        fig.colorbar(c_low, ax=[ax_low, ax_high], orientation='vertical')
        plt.suptitle("Decision boundary")
        pdf.savefig(fig)
        plt.show()

# anim_low_bias = animate_decision_through_learning("low bias", grid, resps_low_bias, X_train, y_train, dg)
# anim_high_bias = animate_decision_through_learning("high bias", grid, resps_high_bias, X_train, y_train, dg)
