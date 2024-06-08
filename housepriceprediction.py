import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('C:/Users/hp/Devil/house_data.csv')

# Inspect the data
print(data.head())
print(data.describe())
print(data.info())

# Select only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=[np.number])  # Change 1

# Plot heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', square=True)  # Change 2
plt.show()

# Plot distribution of the target variable 'SalePrice'
# sns.histplot(data['SalePrice'], kde=True)
# plt.show()

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data)

# Separate features and target
# features = data.drop('SalePrice', axis=1)
# target = data['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = model.score(X_test, y_test)
print(f'R^2 Score: {r2}')

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Load new data
new_data = pd.read_csv('C:/Users/hp/Devil/house_data.csv')
new_data = pd.get_dummies(new_data)

# Ensure the new data has the same columns as the training data
missing_cols = set(features.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[features.columns]

# Predict new data
new_predictions = model.predict(new_data)
print(new_predictions)
