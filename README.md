# Multivariate Linear Regression

A simple implementation of multivariate linear regression using gradient descent, built with NumPy and Matplotlib.

## Contents

1. **Feature Matrix Creation**: Generates random feature vectors with multiple variables ($x_1$, $x_2$)
2. **Target Generation**: Creates synthetic data with linear relationship $y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + noise$
3. **Cost Function**: Calculates Mean Squared Error between predictions and true values
4. **Gradient Descent**: Iteratively finds optimal parameters $\theta_0$, $\theta_1$, and $\theta_2$
5. **3D Visualization**: Shows data points and fitted plane in 3D space

## Generated Plots

### 1. 3D Data Scatter Plot

![figure_1](https://github.com/gxstxxv/Multivariate-Lineare-Regression/blob/main/plots/Figure_1.png)<br>
Shows the synthetic data points distributed in 3D space with two features ($x_1$, $x_2$) and target values ($y$).

### 2. Cost Function Progress

![figure_2](https://github.com/gxstxxv/Multivariate-Lineare-Regression/blob/main/plots/Figure_2_.png)<br>
Displays the cost reduction over iterations during gradient descent optimization.

### 3. Final Model Evaluation

![figure_3](https://github.com/gxstxxv/Multivariate-Lineare-Regression/blob/main/plots/Figure_3.png)<br>
Shows the original data points with the fitted regression plane in 3D.
