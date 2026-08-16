import numpy as np
x = np.array([1,2,3,4])
y = np.array([2,3,4,5])
n = x.size

w = 0
b = 0
learning_rate = 1

for i in range(20):
    # prdections
    y1 = x*w + b

    # error
    e = y - y1

    # mse
    mse = np.mean(np.square(e))

    #dw gradient of w
    dw = (-2/n)*np.sum(x*e)

    #db gradient of b
    db = (-2/n)*np.sum(e)

    #Update w and b
    new_w = w - learning_rate*dw
    new_b = b - learning_rate*db

    w = new_w
    b = new_b

    print("Mse:",mse)
    print("w:",w)
    print("b:",b)
