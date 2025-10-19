import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import *

plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# Parameters
NUM_EPOCHS = 150
INPUT_SIZE = 2
HIDDEN_SIZE = 600
N_HIDDEN = 3
OUTPUT_SIZE = 1
NUM_SAMPLES = 1000
B_SCALE = 1.0
B_SCALE_HIGH = 10.
W_SCALE = np.sqrt(1. * (2 / HIDDEN_SIZE))
SCALE = 1
LOC = 2
OPTIM_TYPE = "Adam"

# %%
X_train, X_test, y_train, y_test, dg, grid, dataloader = create_gaussian_dataset(INPUT_SIZE, NUM_SAMPLES, LOC, SCALE, 2)
model = MLP(INPUT_SIZE, HIDDEN_SIZE, N_HIDDEN, OUTPUT_SIZE, W_SCALE, B_SCALE)
model.reinitialize(92)
model_wide = MLP(INPUT_SIZE, HIDDEN_SIZE, N_HIDDEN, OUTPUT_SIZE, W_SCALE, B_SCALE_HIGH)
model_wide.reinitialize(92)
activations = {}
activations_wide = {}
model.set_activations_hook(activations)
model_wide.set_activations_hook(activations_wide)

model(X_train)
model_wide(X_train)
input_distances = pair_distances(X_train.detach().numpy())
layer_distances = {k: pair_distances(v) for k, v in activations.items()}
layer_distances_wide = {k: pair_distances(v) for k, v in activations_wide.items()}

for layer, distances in layer_distances.items():
    if 'relu' in layer:
        plt.figure()
        plt.title(layer)
        plt.scatter(input_distances, distances, label=layer,alpha=0.5)
        plt.scatter(input_distances, layer_distances_wide[layer], label=layer + ' wide',alpha=0.5)
        plt.legend()
        plt.xlabel('Input distance')
        plt.ylabel('Layer distance')
    plt.show()
