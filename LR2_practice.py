import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Model choosed Linear Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

x = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
    [7, 90],
    [8, 95]
])

y = np.array([
    30,
    35,
    42,
    48,
    55,
    62,
    70,
    78
])

# splitting Data  training nd testing
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.25,
    random_state = 187
)

# Creating a Model
Model = LinearRegression()

#Training model
Model.fit(x_train,y_train)

# To make predictions
print(Model.predict([[4,75]]))

#To Evaluate Model
y_predict = Model.predict(x_test) 
# print(y_predict)
#find metrics between y_predict and y_test

mae = mean_absolute_error(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

print(mse)
print(mae)
print(r2)


