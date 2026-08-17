import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression #type:ignore
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
x = np.array([  [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]]) 
y = np.array([2, 3, 4, 5, 6, 7, 8, 9])
n = x.size

X_train,X_test,Y_train,Y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)
# Create a model
model = LinearRegression()
model.fit(X_train,Y_train) # Scratch_math_calculations of w and b are handled here autoMatically

# testing on test data
y_predict = model.predict(X_test)

#Evalution
mae=mean_absolute_error(Y_test,y_predict)
mse=mean_squared_error(Y_test,y_predict)
r2=r2_score(Y_test,y_predict)

# W is called Coefficient
# b is called intercept

# leanred W
print(model.coef_)
# learned b
print(model.intercept_)
print("MAE : ",mae) # Low is Good
print("MSE : ",mse) # Low is Good
print("r2  : ",r2)  # High is Good

# we tell overfiiting based on metrics mearuse based on Trianing data and testing data
# if metrics on Training and Testing are both are opposite direction then Overfitting

x_new = np.array([[10]])
prediction = model.predict(x_new)
print(prediction)

a = np.array([[20]])
pre = model.predict(a)
print(pre)
