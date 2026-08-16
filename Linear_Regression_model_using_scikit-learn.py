import numpy as np
x = np.array([[1]
              ,[2]
              ,[3]
              ,[4]]) 
y = np.array([2,3,4,5])
n = x.size
from sklearn.model_selection import train_test_split # type: ignore
X_train,X_test,Y_train,Y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)
# Create a model
from sklearn.linear_model import LinearRegression #type:ignore
model = LinearRegression()
model.fit(X_train,Y_train) # Scratch_math_calculations of w and b are handled here autoMatically

# leanred W
print(model.coef_)
# learned b
print(model.intercept_)

x_new = np.array([[10]])
prediction = model.predict(x_new)
print(prediction)



