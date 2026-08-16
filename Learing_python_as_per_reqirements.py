import numpy as np
x = np.array([
    [10,20,30],
    [40,50,60]
])
# print(x.shape)
# print(x.ndim)
# print(x.size)
# print(type(x))

# #############  INDEXING   #########

# x[0] -> 10 20 30 -> gives row
# x[1] -> 40 50 60 
# [0,1] -> 20 [1,2]->60 -> particular element

# x[;,0] -> 10 20 30 -> give row  X wrong
# x[;,1] -> 40 50 60 

# x[rows,cols] , :  -> all
# print(y[:,5])  # in ml way [samples,features]
# print(y[1,:])
######################################
y = np.array([
    [1,2,3,4,5,6],
    [7,8,9,10,11,16],
    [12,13,14,15,17,18]
])


############# SLICING ##########


    # [rows,cols]
    # [strat:stop,start:stop] intermeditate 
    # start included stop excluded
    # example [0:2,:] from above data Y 
    # 0:2 ->rows i.e from 0 row to until 2 row
    # : -> all elements
    #
    # example
    # y[1:3,:] -> all column elements from row 1 to row 2
    # y[:,1:4] -> all row elements from col 1 to col 3#
print(y[1:3,:])
print(y[:,1:4])
###############################

#################Aggregations############
a = np.array([5,10,15,20,25])
b = np.array([1,2,3,4,5])

print(np.sum(a))
print(np.mean(a))
print(np.min(a))
print(np.max(a))

# operations are same as normal a+b or a-b something
# but aggregations does have a Syntax 
#
# Numpy as np -> aggregation functions sum -> np.sum(array_name) , 
# min -> np.min(array_name)
# max -> np.max(a) 
# mean -> np.mean(a)#
########################################

############BroadCasting##################
print(a+5) # Here 5 is Broad Cast for array

#
# In broad casting we do not have any new thing if we add two different shapes (sizes
# it give their sum but in if we add scalar to a vector it differ
# in that case it gives the scalar sum to it i.e addes scalar to
# each element.
# #################Row - Broad Casting##########
# Add row wise but can only have adding row as 1 not more than that

c = np.array([1,2,3])
d = np.array([
    [10],
    [20]
])

######################Reshape################
#
# 
# Converts elements into list into some 2D Arrays like that
# 
#  example : a = np.array([1,2,3,4,5,6,7,8,9])
#  i can convert it into some 2D array until shape is not 
#  effected by elements #

e = np.array([1,2,3,4,5,6,7,8,9])
print(e.reshape(3,3))


############### Zeros , Ones , Arranges #################
# np.zeros((3,2)) -> gives 3 rows and 2 cols of 0's
# np.ones((2,4)) -> gives 2 rows and 4 cols of 1's
# np.arrange((1,10,3)) -> gives numbers from 1 to 10 and gap  b/w each number is 3
# and 10 is excluded


#################### Matrix Multiplication ###############
# 5 12
# # 21 32

# # 12 24
# # 36 44