import numpy as np
import matplotlib.pyplot as plt


def plot_data_scatter(features, targets):
    """ Plots the features and the targets in a 3D scatter plot

       Args:
           features: 2D numpy-array features
           targets: ltargets
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = features[:, 0]
    x2 = features[:, 1]

    ax.scatter(x1, x2, targets, c='b', marker='o')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel('y')

    plt.show()


def plot_progress(costs):
    """ Plots the costs over the iterations

    Args:
        costs: history of costs
    """
    plt.figure()
    plt.plot(range(1, len(costs) + 1), costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def plot_evaluation(x, y, final_theta):
    """ Plots the data x, y together with the final model

    Args:
        x: numpy arrays, x values from the data set
        y: vector, y values from the data set
        final_theta: numpy array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = x[:, 0]
    x2 = x[:, 1]
    ax.scatter(x1, x2, y, c='b', marker='o')

    x1_grid = np.linspace(x1.min(), x1.max(), 20)
    x2_grid = np.linspace(x2.min(), x2.max(), 20)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

    Z = final_theta[0] + final_theta[1] * X1_mesh + final_theta[2] * X2_mesh

    ax.plot_surface(X1_mesh, X2_mesh, Z, color='r', alpha=0.5)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel('y')

    plt.show()
