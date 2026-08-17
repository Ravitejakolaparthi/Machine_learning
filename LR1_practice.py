import numpy as np
from sklearn.model_selection import train_test_split # Model selection is used to split data and import train_test_split class form it
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

x = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
              ])
y = np.array([35, 45, 55, 65, 75, 85])

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size= 0.33,
    random_state = 12
)

Model = LinearRegression()
Model.fit(x_train,y_train)

y_predict = Model.predict(x_test)

# mae = mean_absolute_error(y_test,y_predict)
# mse = mean_squared_error(y_test,y_predict)
# r2 = r2_score(y_test,y_predict)

# print(Model.coef_) # Coffecient i.e W
# print(Model.intercept_) # intercept i.e b

# print("Mae :",mae)
# print("Mse :",mse)
# print("r2  :",r2)

print(Model.predict([[7]]))

