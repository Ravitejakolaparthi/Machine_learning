import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Data Set 
# Given area per Houses and No's Bedroom it contians
X = np.array([
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2000, 4],
    [2200, 4],
    [2500, 5],
    [2800, 5],
    [3000, 6],
    [3200, 6]
])
# actual Prices for each house correspondingly
y = np.array([
    50,
    60,
    75,
    90,
    100,
    110,
    125,
    140,
    150,
    160
])


# Split data Training and Testing
x_train,x_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 187
)
# make a model to learn pattern i.e Linear Regression

Model = LinearRegression()

# Train this Model With trianing Data set
Model.fit(x_train,y_train)

# Test with Testing Data
# Calculate Errors on test data
# Test performance
y_pre = Model.predict(x_test)
test_mae = mean_absolute_error(y_test, y_pre)
test_mse = mean_squared_error(y_test, y_pre)
test_r2 = r2_score(y_test, y_pre)

# Test with Training data

# Calculate Erroes on Training data set
# Training performance
y_try = Model.predict(x_train)
train_mae = mean_absolute_error(y_train, y_try)
train_mse = mean_squared_error(y_train, y_try)
train_r2 = r2_score(y_train, y_try)

print("Test:")
print("MAE:", test_mae)
print("MSE:", test_mse)
print("R²:", test_r2)

print("\nTraining:")
print("MAE:", train_mae)
print("MSE:", train_mse)
print("R²:", train_r2)

print(Model.predict([[1000,3]]))
