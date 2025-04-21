import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate a random DataFrame
np.random.seed(42)  # For reproducibility
n_samples = 100
X = np.random.rand(n_samples, 1) * 10  # Random feature values between 0 and 10
y = 3 * X.squeeze() + 7 + np.random.randn(n_samples) * 2  # Linear relation with noise

# Create a DataFrame
data = pd.DataFrame({'Feature': X.squeeze(), 'Target': y})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Feature']], data['Target'], test_size=0.2, random_state=42)

# Perform linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Output results
print("Linear Regression Results:")
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()
plt.show()

