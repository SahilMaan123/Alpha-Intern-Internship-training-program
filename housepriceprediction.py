import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('C:/Users/hp/Devil/house_data.csv')

print(data.head())
print(data.describe())
print(data.info())

data.fillna(data.mean(), inplace=True)

data = pd.get_dummies(data)

numeric_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', square=True)
plt.show()

target = data['Price (in rupees)']
features = data.drop('Price (in rupees)', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = model.score(X_test, y_test)
print(f'R^2 Score: {r2}')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

new_data = pd.read_csv('C:/Users/hp/Devil/house_data.csv')
new_data = pd.get_dummies(new_data)


missing_cols = set(features.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[features.columns]

new_predictions = model.predict(new_data)
print(new_predictions)
