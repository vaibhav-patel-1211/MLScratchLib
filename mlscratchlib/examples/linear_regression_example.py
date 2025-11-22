import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlscratchlib.supervised.linear_regression import LinearRegression
from mlscratchlib.utils.visualization import Plotter

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with different regularization options
models = {
    'No Regularization': LinearRegression(learning_rate=0.01, n_iterations=1000),
    'L1 (Lasso)': LinearRegression(learning_rate=0.01, n_iterations=1000, regularization='l1', alpha=0.1),
    'L2 (Ridge)': LinearRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', alpha=0.1)
}

# Train models and collect histories
histories = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Calculate scores
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    # Store cost history
    histories[f'{name} (Train R²={train_score:.3f}, Test R²={test_score:.3f})'] = model.cost_history_

# Plot training histories using Plotter
Plotter.plot_training_history(
    histories,
    title='Training History for Different Regularization Methods',
    xlabel='Iteration',
    ylabel='Cost'
)

# Print and collect feature importance for each model
print("\nFeature Importance (Model Coefficients):")
feature_names = diabetes.feature_names

for name, model in models.items():
    print(f"\n{name}:")
    for feature, coef in zip(feature_names, model.weights_):
        print(f"{feature}: {coef:.4f}")
    print(f"Bias: {model.bias_:.4f}")

# Create scatter plot of predictions vs actual values for the best model
best_model = models['L2 (Ridge)']
y_pred = best_model.predict(X_test_scaled)

# Use Plotter to create scatter plot
Plotter.plot_scatter(
    X=np.column_stack((y_test, y_pred)),
    title='Predictions vs Actual Values (L2 Regularization)',
    xlabel='Actual Values',
    ylabel='Predicted Values'
)