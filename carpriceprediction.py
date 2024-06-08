import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('C:/Users/hp/Devil/car.csv')

print(df.head())

print(df.isnull().sum())

df.dropna(subset=['selling_price'], inplace=True)

df['year'] = df['year'].astype(int)
df['km_driven'] = df['km_driven'].astype(int)

X = df.drop(columns=['selling_price'])
y = df['selling_price']

numerical_cols = ['year', 'km_driven']
categorical_cols = [col for col in X.columns if col not in numerical_cols]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared5 Error (RMSE): {rmse}")

# Example new data
new_data = pd.DataFrame({
    'name': ['Maruti Swift Dzire VDI'],
    'year': [2014],
    'km_driven': [70000],
    'fuel': ['Diesel'],
    'seller_type': ['Individual'],
    'transmission': ['Manual'],
    'owner': ['First Owner']
})

new_data = new_data[X.columns]

predicted_price = pipeline.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")
