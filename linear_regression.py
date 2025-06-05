import numpy as np


def create_feature_matrix(sample_size, n_features, x_min, x_max):
    """ Creates random feature vectors in a given intervall

    Args:
        sample_size: number feature vectors
        n_features: number of features for each vector
        x_min: lower bound value ranges
        x_max: upper bound value ranges

    Returns:
        x: 2D array containing feature vecotrs
        with shape (sample_size, n_features)
    """
    return np.random.uniform(
        low=x_min,
        high=x_max,
        size=(sample_size, n_features),
    )


def linear_hypothesis(thetas):
    """ Combines given list argument in a linear
    equation and returns it as a function

    Args:
        thetas: list of coefficients

    Returns:
        lambda that models a linear function based on thetas and x
    """
    return lambda x: np.hstack((np.ones((x.shape[0], 1)), x)) @ thetas


def generate_targets(X, theta, sigma):
    """ Combines given arguments in a linear equation with X,
    adds some Gaussian noise and returns the result

    Args:
        X: 2D numpy feature matrix
        theta: list of coefficients
        sigma: standard deviation of the gaussian noise

    Returns:
        target values for X
    """
    h = linear_hypothesis(theta)
    y = h(X)
    noise = np.random.normal(loc=0.0, scale=sigma, size=y.shape)

    return y + noise


def mse_cost_function(x, y):
    """ Implements MSE cost function as a
    function J(theta) on given traning data

    Args:
        x: vector of x values
        y: vector of ground truth values y

    Returns:
        lambda J(theta) that models the cost function
    """
    m = x.shape[0]

    return lambda theta: (1 / (2 * m)) * np.sum(
        (linear_hypothesis(theta)(x) - y) ** 2
    )


def update_theta(x, y, theta, learning_rate):
    """ Updates learnable parameters theta

    The update is done by calculating the partial derivities of
    the cost function including the linear hypothesis. The
    gradients scaled by a scalar are subtracted from the given
    theta values.

    Args:
        x: 2D numpy array of x values
        y: array of y values corresponding to x
        theta: current theta values
        learning_rate: value to scale the negative gradient

    Returns:
        theta: Updated theta vector
    """
    m = x.shape[0]
    prediction = linear_hypothesis(theta)(x)
    x = np.hstack((np.ones((m, 1)), x))

    gradient = (1/m) * (x.T @ (prediction - y))

    return theta - learning_rate * gradient


def gradient_descent(
    learning_rate,
    theta, iterations,
    x,
    y,
    cost_function,
    verbose: bool = False
):
    """ Minimize theta values of a linear model based on MSE cost function

    Args:
        learning_rate: scalar, scales the negative gradient
        theta: initial theta values
        x: vector, x values from the data set
        y: vector, y values from the data set
        iterations: scalar, number of theta updates
        cost_function: python function for computing the cost

    Returns:
        history_cost: cost after each iteration
        history_theta: Updated theta values after each iteration
    """
    costs = []
    thetas = []

    J = cost_function(x, y)

    for i in range(iterations):
        theta = update_theta(x, y, theta, learning_rate)
        cost = J(theta)

        costs.append(cost)
        thetas.append(theta.copy())

        if verbose and i % (iterations // 10) == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, Theta = {theta}")

    return costs, thetas
