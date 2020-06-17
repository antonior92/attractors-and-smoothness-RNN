# %% Generate plots
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def plot_2d(trajectory, ax, n, i, j, initial_epoch, final_epoch, burnout):
    # Get dimensions
    seq_len, n_sequences, n_inputs, n_outputs, n_epochs = trajectory.shape

    rr = min(n_epochs, final_epoch)

    # Plot bifurcation diagram and cost function
    cm = plt.get_cmap('hsv')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=rr)
    for xx in range(initial_epoch, rr):
        traj = trajectory[burnout:-1, n, i, j, xx]
        traj_del = trajectory[burnout + 1:, n, i, j, xx] - trajectory[burnout:-1, n, i, j, xx]
        ax.scatter(xx, traj, traj_del, s=20, marker='.', color=cm(norm(xx)), alpha=0.15)

    return ax


def plot_1d(trajectory, ax, n, i, j, initial_epoch, epochs, burnout):
    # Get dimensions
    seq_len, n_sequences, n_inputs, n_outputs, n_epochs = trajectory.shape
    rr = min(n_epochs, epochs)
    cm = plt.get_cmap('hsv')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=rr)
    for xx in range(initial_epoch, rr):
        traj = trajectory[burnout:, n, i, j, xx]
        ax.scatter((seq_len-burnout)*[xx], traj, s=20, marker='.', color=cm(norm(xx)), alpha=0.15)
    return ax


def plot_transient_response(trajectory, ax, n, i, j, epoch, burnout):
    traj = trajectory[burnout:, n, i, j, epoch]
    ax.plot(traj)
    return ax


def plot(trajectory, ax, n=0, i=0, j=0, tp='orbit1d', initial_epoch=0, final_epoch=-1,
         burnout=0):
    final_epoch = trajectory.shape[-1] - final_epoch + 1 if final_epoch < 0 else final_epoch
    # Some plotting constant
    time_label = 't'
    state_label = 'x'
    diff_label = 'dx'
    epoch_label = 'epoch'
    # Plot according to the type
    if tp in ['orbit1d']:
        ax = plot_1d(trajectory, ax, n, i, j, initial_epoch, final_epoch, burnout)
        ax.set_xlabel(epoch_label)
        ax.set_ylabel(state_label)
    elif tp in ['orbit2d']:
        ax = plot_2d(trajectory, ax, n, i, j, initial_epoch, final_epoch, burnout)
        ax.set_xlabel(epoch_label)
        ax.set_ylabel(state_label)
        ax.set_zlabel(diff_label)
    elif tp in ['transient']:
        epoch = final_epoch
        ax = plot_transient_response(trajectory, ax, n, i, j, epoch, burnout)
        ax.set_xlabel(time_label)
        ax.set_ylabel(state_label)
    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate bifurcation diagrams (a.k.a. orbit diagram) '
                                                 'from points saved during training.')
    parser.add_argument('path', type=str, metavar='PATH',
                        help='path to pytorch file containing points for generating such diagram.')
    parser.add_argument('-t', '--type', choices=['orbit1d', 'orbit2d', 'transient'], metavar='TYPE',
                        default='orbit1d', help='type of plot.')
    parser.add_argument('-n', '--sequence_num', metavar='N', type=int, default=0,
                        help='which sequence to use.')
    parser.add_argument('-i', '--input_num', metavar='I', type=int, default=0,
                        help='which input to apply.')
    parser.add_argument('-j', '--t_output_num', metavar='J', type=int, default=0,
                        help='which output to sample from.')
    parser.add_argument('-ie', '--initial_epoch', metavar='N', type=int, default=0,
                        help='Initial epoch to plot.')
    parser.add_argument('-fe', '--final_epoch', metavar='N', type=int, default=-1,
                        help='Final epoch to plot.')
    parser.add_argument('-b', '--burnout', metavar='B', type=int, default=0,
                        help='Burnout period, time for the system to stabilize.')
    args = parser.parse_args()


    # Get datapoints
    ckpt = torch.load(args.path, map_location='cpu')
    trajectory = ckpt["trajectory"].detach().numpy()

    # Get figure and axis
    fig = plt.figure(figsize=(6, 6))
    if args.type == 'orbit2d':
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    plot(trajectory, ax, tp=args.type, i=args.input_num, j=args.t_output_num, n=args.sequence_num,
         initial_epoch=args.initial_epoch, final_epoch=args.final_epoch, burnout=args.burnout)
    plt.show()