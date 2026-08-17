import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    12,
    19,
    31,
    38,
    52,
    57,
    71,
    79,
    91,
    105
])


x_train,x_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state=187
)

Model = LinearRegression()

# Model Trained
Model.fit(x_train,y_train)

# Testing with test data
y_pre = Model.predict(x_test)

#Errors
mae = mean_absolute_error(y_test,y_pre)
mse = mean_squared_error(y_test,y_pre)
r2 = r2_score(y_test,y_pre)

train_pred  = Model.predict(x_train)

Mae = mean_absolute_error(y_train,train_pred)
Mse = mean_squared_error(y_train,train_pred)
rr2 = r2_score(y_train,train_pred)

print([mae,mse,r2])
print([Mae,Mse,rr2])


