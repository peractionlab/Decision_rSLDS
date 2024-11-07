import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import util

# Plotting parameters
color_names = ["windows blue", "red", "amber", "faded green", "purple"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

def plot_data(x_train, title, xlabel, ylabel):
    """
    Plot the latent states over time for multiple trials.
    
    Parameters:
    - x_train: Latent states for multiple trials.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(5, 5))
    trial_num = x_train.shape[0]
    trial_len = x_train.shape[1]
    plt.title(title)
    
    # Plot each trial's latent states
    for i in range(trial_num):
        plt.plot(x_train[i, :trial_len, 0], x_train[i, :trial_len, 1])
    
    plt.grid(True, lw=1, ls='--', c='c')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_synthetic_dynamics(As, bs, xlim=(-5, 10), ylim=(-5, 10), nxpts=20, nypts=20,
                            alpha=0.8, ax=None, figsize=(6, 6)):
    """
    Plot the synthetic dynamics based on the given transition matrices and biases.
    
    Parameters:
    - As: List of transition matrices.
    - bs: List of bias vectors.
    - xlim: Range for the x-axis.
    - ylim: Range for the y-axis.
    - nxpts: Number of points along the x-axis.
    - nypts: Number of points along the y-axis.
    - alpha: Transparency level for the plot.
    - ax: Matplotlib axis object (optional).
    - figsize: Figure size.
    
    Returns:
    - ax: Matplotlib axis object.
    """
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    z = np.zeros(xy.shape[0])

    # Determine the mode for each point
    for i, p in enumerate(xy):
        if np.abs(p[1] - p[0]) <= 1:
            z[i] = 0
        elif p[1] - p[0] > 1: 
            z[i] = 1
        else:
            z[i] = 2
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    # Plot the dynamics for each mode
    for k, (A, b) in enumerate(zip(As, bs)):
        dxydt_m = xy.dot(A.T) + b - xy
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)
    
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_xticks(range(-4, 12, 2))
    ax.set_yticks(range(-4, 12, 2))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    return ax

def plot_most_likely_dynamics(model, xlim=(-50, 50), ylim=(-50, 50), nxpts=20, nypts=20,
                              alpha=0.8, ax=None, figsize=(6, 6)):
    """
    Plot the most likely dynamics based on the model's transition probabilities.
    
    Parameters:
    - model: Model object containing dynamics and transitions.
    - xlim: Range for the x-axis.
    - ylim: Range for the y-axis.
    - nxpts: Number of points along the x-axis.
    - nypts: Number of points along the y-axis.
    - alpha: Transparency level for the plot.
    - ax: Matplotlib axis object (optional).
    - figsize: Figure size.
    
    Returns:
    - ax: Matplotlib axis object.
    """
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the most likely mode for each point
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xticks(range(-4, 12, 2))
        ax.set_yticks(range(-4, 12, 2))
    
    # Plot the dynamics for each mode
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)
    
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Find and plot attractors
    atts = util.find_attractor(model)
    for k in range(model.K):
        ax.scatter(atts[k][0], atts[k][1], color=colors[k % len(colors)], marker='*')
    
    plt.tight_layout()
    return ax