import numpy as np
from linear_regression import (
    create_feature_matrix,
    linear_hypothesis,
    generate_targets,
    mse_cost_function,
    gradient_descent,
)
from plotting_functions import (
    plot_data_scatter,
    plot_progress,
    plot_evaluation,
)


def main():
    sample_size = 100
    n_features = 2
    x_min = [1.5, -0.5]
    x_max = [11., 5.0]

    X = create_feature_matrix(sample_size, n_features, x_min, x_max)
    assert len(X[:, 0]) == sample_size
    assert len(X[0, :]) == n_features
    for i in range(n_features):
        assert np.max(X[:, i]) <= x_max[i]
        assert np.min(X[:, i]) >= x_min[i]

    assert len(linear_hypothesis([.1, .2, .3])(X)) == sample_size

    theta = (2., 3., -4.)
    sigma = 3.
    y = generate_targets(X, theta, sigma)
    assert len(y) == sample_size

    plot_data_scatter(X, y)

    J = mse_cost_function(X, y)
    print(J(theta))

    alpha = .001
    nb_iterations = 50000
    start_values_theta = [42., 42., 42.]
    history_cost, history_theta = gradient_descent(
        alpha,
        start_values_theta,
        nb_iterations,
        X,
        y,
        mse_cost_function,
        True,
    )

    plot_progress(history_cost)
    plot_evaluation(X, y, history_theta[-1])


if __name__ == "__main__":
    main()
