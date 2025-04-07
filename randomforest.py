import numpy as np  # for numerical computations
import pandas as pd  # for data manipulation
from sklearn.model_selection import train_test_split  # to split dataset
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from sklearn.metrics import mean_squared_error  # to evaluate performance

# Simulated housing dataset
data = {
    'size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'age': [10, 15, 20, 5, 25, 7, 12, 5, 8, 18],
    'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

df = pd.DataFrame(data)

# Feature matrix and target vector
X = df[['size', 'bedrooms', 'age']].values
y = df['price'].values

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 5  # Number of RF models
predictions = []

# Train multiple Random Forest models with different seeds
for i in range(n_estimators):
    model = RandomForestRegressor(n_estimators=10, random_state=i)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions.append(preds)

# Average predictions
final_predictions = np.mean(predictions, axis=0)

# Evaluate
mse = mean_squared_error(y_test, final_predictions)
print(f"Mean Squared Error: {mse:.2f}")
