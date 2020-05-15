# %% Chaotic LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set problem dimensions
batch_size = 1
input_size = 1
hidden_size = 2


# Generator a grid of points 1d
def generate_points(pt1, pt2, s_list):
    for s in s_list:
        if s == 1.0:
            yield pt2
        elif s == 0.0:
            yield pt1
        else:
            yield {name: s * pt2[name] + (1 - s) * pt1[name] for name, _ in pt1.items()}


# Generate a grid of points 2d
def generate_points_2d(pt1, pt2, pt3, s1_list, s2_list):
    """
    Yields grid points from a 2-dimensional grid defined by
        - three points pt1, pt2, and pt3, and
        - two scaling factors s1 and s2.
    Scaling factor s1 interpolates between pt1 and pt2, while
    scaling factor s2 interpolates between pt1 and pt3:
        - s1 = 0 and s2 = 0 corresponds to pt1
        - s1 = 1 and s2 = 0 corresponds to pt2
        - s1 = 0 and s2 = 1 corresponds to pt3
    """
    for s1 in s1_list:
        for s2 in s2_list:
            if s1 == 0.0 and s2 == 0.0:
                yield pt1
            elif s1 == 1.0 and s2 == 0.0:
                yield pt2
            elif s1 == 0.0 and s2 == 1.0:
                yield pt3
            else:
                yield {name: s1 * pt2[name] + (1 - s1) * pt1[name] +
                             s2 * pt3[name] + (1 - s2) * pt1[name] for name in pt1.keys()}


# Set 2 points
zero_params = {"weight_hh_l0": torch.zeros((8, 2)),
               "weight_ih_l0": torch.zeros((8, 1))}
chaotic_params = {"weight_hh_l0": torch.tensor([[-1.0, -4.0],
                                                [-3.0, -2.0],  # W_hi
                                                [-2.0, 6.0],
                                                [0, -6.0],  # W_hf
                                                [-1.0, -6.0],
                                                [6.0, -9.0],  # W_hg
                                                [4.0, 1.0],
                                                [-9.0, -7.0]]),  # W_ho
                  "weight_ih_l0": torch.zeros((8, 1))}


# Define model
lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bias=False)


# Define initial condition (arbitrarily)
h_0 = 0.5*torch.ones((1, batch_size, hidden_size))
c_0 = 0.5*torch.ones((1, batch_size, hidden_size))
state_0 = (h_0, c_0)


# %% Generate strange attractor for lstm
# Set constants
burnout = 100
seq_len = 5000
# Zero input
input = torch.zeros((seq_len, batch_size, input_size))
# Load parameters
lstm.load_state_dict(chaotic_params)
# Simulate for zero input
with torch.no_grad():
    output_data, state_n = lstm(input, state_0)
    output_data = output_data[burnout:, :]  # remove initial samples
# Convert to numpy
output_data_np = np.squeeze(output_data.detach().numpy())

plt.scatter(output_data_np[:, 0], output_data_np[:, 1], s=4)
plt.show()

# %% Compute loss function and output along the system
# Set constants
seq_len = 200
burnout = 100
seq_len = 200  # seq_len > burnout
n_points = 5000
min_scale = 0.5
max_scale = 2
# Get output data
output_data = output_data[:seq_len-burnout, :]  # remove extra samples
# Zero input
input = torch.zeros((seq_len, batch_size, input_size))
# Load parameters
scale_list = np.linspace(min_scale, max_scale, n_points-1)
scale_list = np.append(scale_list, 1.0)  # make sure 1 belongs here
scale_list = np.sort(scale_list)  # order array
initial_points = generate_points(zero_params, chaotic_params, scale_list)
final_points = np.zeros((seq_len-burnout, n_points))
loss_list = []
for i, p in tqdm(enumerate(initial_points)):
    lstm.load_state_dict(p)
    # Simulate for zero input
    with torch.no_grad():
        output, state_n = lstm(input, state_0)
        output = output[burnout:, :]  # remove initial samples
        loss = F.mse_loss(output, output_data)
    # Convert to numpy
    final_points[:, i] = np.squeeze(output.detach().numpy())[:, 0]
    # Get loss
    loss_list.append(loss)

# %% Plot bifurcation diagram and cost function
_, ax = plt.subplots(2, 1)
ax[0].scatter(np.repeat(scale_list, seq_len-burnout),  final_points.T, s=0.1)
ax[0].set_ylabel("First output")
ax[1].plot(scale_list, loss_list)
ax[1].scatter(1, 0, marker='o')
ax[1].set_ylabel("MSE Loss")
ax[1].set_xlabel("$s$")
plt.show()

# %% Compute loss function on 2-dimensional grid
min_s1 = 0
max_s1 = 2
min_s2 = -0.5
max_s2 = 0.5
n_points_s1 = 150
n_points_s2 = 150
s1_list = np.linspace(min_s1, max_s1, n_points_s1)
s2_list = np.linspace(min_s2, max_s2, n_points_s2)
s1_list = np.append(s1_list, 0.0)
s1_list = np.append(s1_list, 1.0)
s2_list = np.append(s2_list, 0.0)
n_points_s1 += 2
n_points_s2 += 1
s1_list = np.sort(s1_list)
s2_list = np.sort(s2_list)
random_params = {"weight_hh_l0": torch.rand((8, 2)), "weight_ih_l0": torch.rand((8, 1))}
initial_points_2d = generate_points_2d(zero_params, chaotic_params, random_params, s1_list, s2_list)
final_points_2d = np.zeros((seq_len-burnout, n_points_s1 * n_points_s2))
loss_array_2d = np.zeros(n_points_s1 * n_points_s2)
for i, p in tqdm(enumerate(initial_points_2d)):
    lstm.load_state_dict(p)
    # Simulate for zero input
    with torch.no_grad():
        output, state_n = lstm(input, state_0)
        output = output[burnout:, :]  # remove initial samples
        loss = F.mse_loss(output, output_data)
    # Convert to numpy
    final_points_2d[:, i - 1] = np.squeeze(output.detach().numpy())[:, 0]
    # Get loss
    loss_array_2d[i - 1] = float(loss.numpy())

#%% Contourf plot
fig, ax = plt.subplots()
CS = ax.contourf(S1, S2, LOSS, 1000)
# ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Loss evaluated on 2-dimensional grid \n '
             'pt1 (s1=0,s2=0) = zero_params, \n '
             'pt2 (s1=1,s2=0) = chaotic_params, \n '
             'pt3 (s1=0,s2=1) = random_params')
plt.xlabel('s1 (interpolation between pt1 and pt2)')
plt.ylabel('s2 (interpolation between pt1 and pt3)')
fig.colorbar(CS)
plt.show()
