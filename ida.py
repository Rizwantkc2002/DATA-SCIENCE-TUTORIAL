import numpy as np  # importing numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # importing sklearn

# New data: 3 data points, 2 features
X = np.array([[1, 2], [2, 3], [3, 6]])  # Input features
y = np.array([0, 0, 1])  # Class labels (2 samples of class 0, 1 sample of class 1)

# --- Using scikit-learn ---
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda_sklearn = lda.fit_transform(X, y)

print("LDA Transformed Data (sklearn):\n", X_lda_sklearn)

# --- Manual Calculation of LDA ---

# Class means
mean_0 = X[y == 0].mean(axis=0)  # Mean of class 0
mean_1 = X[y == 1].mean(axis=0)  # Mean of class 1

# Compute within-class scatter matrix
S_W = np.zeros((2, 2))
for label in np.unique(y):
    class_samples = X[y == label]
    mean_vec = class_samples.mean(axis=0).reshape(1, -1)
    S_W += (class_samples - mean_vec).T @ (class_samples - mean_vec)

# Compute between-class scatter matrix
overall_mean_diff = (mean_1 - mean_0).reshape(-1, 1)
S_B = np.dot(overall_mean_diff, overall_mean_diff.T)

# Eigenvalue problem
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
lda_direction = eigvecs[:, np.argmax(eigvals)].real  # Ensure it's real

print("LDA Direction Vector (manual computation):\n", lda_direction)
