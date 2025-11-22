# 📦 MLScratchLib

**MLScratchLib** is a lightweight Python library that implements popular machine learning algorithms **from scratch** using only NumPy and basic Python. This library is designed for learning, experimentation, and educational purposes, providing a deep understanding of how these algorithms work under the hood without relying on high-level libraries like scikit-learn, TensorFlow, or PyTorch.

---

## 🚀 Features

### Supervised Learning

#### Regression

- **Linear Regression** - Standard linear regression with gradient descent optimization
  - Supports batch and stochastic gradient descent
  - L1 (Lasso) and L2 (Ridge) regularization options
  - Cost history tracking
- **Ridge Regression** - Linear regression with L2 regularization
- **Lasso Regression** - Linear regression with L1 regularization
- **Elastic Net Regression** - Combines L1 and L2 regularization
- **Support Vector Regression (SVR)** - Support vector machine for regression tasks

#### Classification

- **Logistic Regression** - Binary and multiclass classification
  - Supports batch and stochastic gradient descent
  - L1 and L2 regularization options
  - One-vs-rest strategy for multiclass classification
- **K-Nearest Neighbors (KNN)** - Classification and regression
  - Euclidean and Manhattan distance metrics
  - Uniform and distance-based weighting
- **Decision Tree** - Classification and regression trees
  - Configurable depth and splitting criteria
  - Gini impurity and information gain
- **Support Vector Machine (SVM)** - Classification with kernel support
  - Linear, polynomial, and RBF kernels
  - Sequential Minimal Optimization (SMO) algorithm
  - Stochastic Gradient Descent (SGD) variant for large datasets
- **Gradient Boosting** - Ensemble learning method

### Unsupervised Learning

- **K-Means Clustering** - Partition-based clustering
  - K-Means++ initialization
  - Within-cluster sum of squares (WCSS) calculation
- **Hierarchical Clustering** - Agglomerative and divisive clustering
  - Single, complete, and average linkage methods
  - Dendrogram visualization support
- **DBSCAN** - Density-based clustering
  - Automatically determines number of clusters
  - Identifies noise points
- **Principal Component Analysis (PCA)** - Dimensionality reduction
  - Variance explained calculation
  - Component selection
- **Singular Value Decomposition (SVD)** - Matrix factorization
  - Truncated SVD for dimensionality reduction
  - Variance explained analysis

---

## 📦 Installation

The library is available on PyPI and can be installed using pip:

```bash
pip install mlscratchlib
```

### Requirements

- Python 3.6+
- NumPy
- Matplotlib (for visualization utilities)

All dependencies are automatically installed when you install the package via pip.

---

## 🎯 Quick Start

### Linear Regression Example

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlscratchlib.supervised.linear_regression import LinearRegression

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LinearRegression(
    learning_rate=0.01,
    n_iterations=1000,
    regularization='l2',  # Optional: 'l1', 'l2', or 'none'
    alpha=0.1             # Regularization strength
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
score = model.score(X_test_scaled, y_test)
print(f"R² Score: {score:.4f}")
```

### Logistic Regression Example

```python
from mlscratchlib.supervised.logistic_regression import LogisticRegression

# Create model
model = LogisticRegression(
    learning_rate=0.01,
    n_iterations=1000,
    regularization='l2',
    alpha=0.1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get probabilities
y_proba = model.predict_proba(X_test)
```

### K-Means Clustering Example

```python
from mlscratchlib.unsupervised.kmeans import KMeans

# Create model
kmeans = KMeans(
    n_clusters=3,
    max_iter=300,
    init='kmeans++',  # or 'random'
    random_state=42
)

# Fit to data
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get centroids
centroids = kmeans.centroids
```

### Decision Tree Example

```python
from mlscratchlib.supervised.decision_tree import DecisionTreeClassifier

# Create model
tree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

# Train
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)
```

### KNN Classifier Example

```python
from mlscratchlib.supervised.knn import KNNClassifier

# Create model
knn = KNNClassifier(
    n_neighbors=5,
    metric='euclidean',  # or 'manhattan'
    weights='distance'   # or 'uniform'
)

# Train (KNN just stores the training data)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
```

### SVM Example

```python
from mlscratchlib.supervised.svm import SVM

# Create model with RBF kernel
svm = SVM(
    C=1.0,
    kernel='rbf',  # 'linear', 'poly', or 'rbf'
    gamma='scale',
    max_iter=1000
)

# Train
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)
```

### PCA Example

```python
from mlscratchlib.unsupervised.pca import PCA

# Create model
pca = PCA(n_components=2)

# Fit and transform
X_reduced = pca.fit_transform(X)

# Transform new data
X_new_reduced = pca.transform(X_new)

# Access components
components = pca.components
```

---

## 📚 Detailed Usage

### Linear Regression

```python
from mlscratchlib.supervised.linear_regression import LinearRegression

model = LinearRegression(
    learning_rate=0.01,      # Step size for gradient descent
    n_iterations=1000,       # Maximum iterations
    batch_size=None,         # None for batch GD, int for SGD
    regularization='l2',     # 'none', 'l1', or 'l2'
    alpha=0.1                # Regularization strength
)

model.fit(X_train, y_train)

# Access training history
cost_history = model.cost_history_
weights = model.weights_
bias = model.bias_

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
```

### Ridge and Lasso Regression

```python
from mlscratchlib.supervised.ridge_regression import RidgeRegression
from mlscratchlib.supervised.lasso_regression import LassoRegression

# Ridge Regression
ridge = RidgeRegression(learning_rate=0.01, n_iterations=1000, alpha=0.1)
ridge.fit(X_train, y_train)

# Lasso Regression
lasso = LassoRegression(learning_rate=0.01, n_iterations=1000, alpha=0.1)
lasso.fit(X_train, y_train)
```

### Logistic Regression

```python
from mlscratchlib.supervised.logistic_regression import LogisticRegression

model = LogisticRegression(
    learning_rate=0.01,
    n_iterations=1000,
    batch_size=None,        # None for batch, int for SGD
    regularization='l2',
    alpha=0.1
)

model.fit(X_train, y_train)

# For binary classification
predictions = model.predict(X_test)

# For multiclass classification (one-vs-rest)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Access model attributes
classes = model.classes_
weights = model.weights_
bias = model.bias_
cost_history = model.cost_history_
```

### Decision Tree

```python
from mlscratchlib.supervised.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
clf = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    min_impurity_decrease=0.0
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Regression
reg = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### K-Means Clustering

```python
from mlscratchlib.unsupervised.kmeans import KMeans

kmeans = KMeans(
    n_clusters=3,
    max_iter=300,
    random_state=42,
    init='kmeans++'  # or 'random'
)

kmeans.fit(X)

# Access results
labels = kmeans.labels_
centroids = kmeans.centroids

# Calculate WCSS (Within-Cluster Sum of Squares)
wcss = kmeans._compute_wcss(X)
```

### DBSCAN

```python
from mlscratchlib.unsupervised.dbscan import DBSCAN

dbscan = DBSCAN(
    eps=0.5,          # Maximum distance between samples
    min_samples=5,    # Minimum samples in a neighborhood
    metric='euclidean'
)

dbscan.fit(X)

# Access results
labels = dbscan.labels_  # -1 indicates noise points
core_samples = dbscan.core_sample_indices_
```

### Hierarchical Clustering

```python
from mlscratchlib.unsupervised.hierarchical import HierarchicalClustering

hierarchical = HierarchicalClustering(
    n_clusters=3,
    method='agglomerative',  # or 'divisive'
    linkage='complete'       # 'single', 'complete', or 'average'
)

hierarchical.fit(X)

labels = hierarchical.labels_
```

### Principal Component Analysis

```python
from mlscratchlib.unsupervised.pca import PCA

pca = PCA(n_components=2)

# Fit and transform
X_reduced = pca.fit_transform(X)

# Transform new data
X_new_reduced = pca.transform(X_new)

# Access attributes
components = pca.components
mean = pca.mean
std = pca.std
```

### Singular Value Decomposition

```python
from mlscratchlib.unsupervised.svd import SVD

svd = SVD(n_components=10)  # or None for all components

svd.fit(X)
X_reduced = svd.transform(X)

# Access attributes
singular_values = svd.singular_values_
explained_variance = svd.explained_variance_
explained_variance_ratio = svd.explained_variance_ratio_
components = svd.components_
```

---

## 📖 API Reference

### Supervised Learning

#### Regression

- `LinearRegression` - Base linear regression with regularization options
- `RidgeRegression` - L2 regularized linear regression
- `LassoRegression` - L1 regularized linear regression
- `ElasticNetRegression` - Combined L1 and L2 regularization
- `SVR` - Support Vector Regression

#### Classification

- `LogisticRegression` - Binary and multiclass logistic regression
- `KNNClassifier` / `KNNRegressor` - K-nearest neighbors
- `DecisionTreeClassifier` / `DecisionTreeRegressor` - Decision trees
- `SVM` / `SVMSGD` - Support Vector Machines
- `GradientBoosting` - Gradient boosting classifier

### Unsupervised Learning

- `KMeans` - K-means clustering
- `HierarchicalClustering` - Hierarchical clustering
- `DBSCAN` - Density-based clustering
- `PCA` - Principal Component Analysis
- `SVD` - Singular Value Decomposition

All classes follow the standard scikit-learn API pattern:

- `fit(X, y=None)` - Train the model
- `predict(X)` - Make predictions
- `transform(X)` - Transform data (for dimensionality reduction)
- `score(X, y)` - Evaluate model performance (where applicable)

---

## 🔬 Examples

The library includes example scripts in the `examples/` directory:

- `linear_regression_example.py` - Demonstrates linear regression with different regularization methods
- `logistic_regression_example.py` - Logistic regression example
- `decision_tree_example.py` - Decision tree usage
- `clustering_example.py` - K-means clustering demonstration

You can run these examples to see the library in action!

---

## 🎓 Educational Purpose

This library is specifically designed for:

- **Learning**: Understand how machine learning algorithms work internally
- **Teaching**: Use in courses and tutorials to explain ML concepts
- **Experimentation**: Modify and extend algorithms to see how changes affect performance
- **Research**: Implement custom variations of standard algorithms

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add new algorithms or improve existing ones, please feel free to submit pull requests.

---

## 📝 License

See the LICENSE file for details.

---

## 👤 Author

**PATEL VAIBHAV**

- Email: vaibhav1211patel@gmail.com

---

## 📌 Notes

- This library is **not intended for production use** where performance is critical
- For production applications, use optimized libraries like scikit-learn, TensorFlow, or PyTorch
- The focus is on **readability and educational value** over performance optimization
- All implementations are built from scratch using only NumPy and standard Python libraries

---

## 🔗 Links

- **PyPI Package**: [mlscratchlib](https://pypi.org/project/mlscratchlib/)
- **Installation**: `pip install mlscratchlib`

---

**Happy Learning! 🚀**
