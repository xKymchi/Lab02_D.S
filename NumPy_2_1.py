
# Read the links: https://numpy.org/doc/stable/user/quickstart.html  and https://numpy.org/doc/stable/user/basics.broadcasting.html  before solving the exercises. 

import numpy as np

# ### Print out the dimension (number of axes), shape, size and the datatype of the matrix A.

A = np.arange(1, 16).reshape(3,5)
print(A.ndim, A.shape, A.size, A.dtype)

# ### Do the following computations on the matrices B and C: 
# * Elementwise subtraction. 
# * Elementwise multiplication. 
# * Matrix multiplication (by default you should use the @ operator).
B = np.arange(1, 10).reshape(3, 3)
C = np.ones((3, 3))*2

print(B)
print()
print(C)

## subtraction
sub = B - C
print(sub)
## or simply
print(B - C)

## multiply
mult = B * C
print(mult)
## or simply
print(B * C)

##matrix
matrix = B @ C 
print(matrix)
## or simply
print(B @ C)

# ### Do the following calculations on the matrix:
# * Exponentiate each number elementwise (use the np.exp function).
# 
# * Calculate the minimum value in the whole matrix. 
# * Calculcate the minimum value in each row. 
# * Calculcate the minimum value in each column. 
# 
# 
# * Find the index value for the minimum value in the whole matrix (hint: use np.argmin).
# * Find the index value for the minimum value in each row (hint: use np.argmin).
# 
# 
# * Calculate the sum for all elements.
# * Calculate the mean for each column. 
# * Calculate the median for each column. 

B = np.arange(1, 10).reshape(3, 3)
print(B)

## Exponent
exponent = np.exp(B)
print(exponent)
## or simply
print(np.exp(B))

## Minimum value in whole matrix
print(np.min(B))

## Minimum value in each row 
print(np.min(B, axis = 1))

## Minimum value in each column 
print(np.min(B, axis = 0))

## Find index value for the minimum value in whole matrix 
np.argmin(B)

## Find index value for the minimum value in each row
np.argmin(B, axis = 1)

## calculate sum for ALL elements
np.sum(B)

## Calculate the MEAN for each column 
np.mean(B, axis = 0)

## Calculate the MEDIAN for each column
np.median(B, axis = 0)

# ### What does it mean when you provide fewer indices than axes when slicing? See example below.
print(A)
A[1]

# **Answer:**
## in the given example above we have provided with index 1 which is the second row in the array, 
## this means that we are now only accesing the elements within the second row

# ### Iterating over multidimensional arrays is done with respect to the first axis, so in the example below we iterate trough the rows. If you would like to iterate through the array *elementwise*, how would you do that?
A

for i in A:
    print(i)

for element in np.nditer(A):
    print(element)
## or
for row in A:
    for element in row:
        print(element)

# ### Explain what the code below does. More specifically, b has three axes - what does this mean? 
a = np.arange(30)
b = a.reshape((2, 3, -1))
print(a)
print()

print(b)

## We start with creating a Numpy array named 'a' with 30 elements,
## due to the startpoint of indexing being 0, we have integers 0 to 29 which in total is 30 elements.
## Then we declare a variable named 'b' which uses the 'reshape' method in that belong to the ndarray object to rechange the shape of the array
## this method allows us to add/remove dimensions or change number of elements in each dimension.
## We print 'a' which is the original array
## then we print 'b' which is the array with the changes added to it
## in this case we started with a 1D array then reshaped the array into a 3D array. 

##-------------------------------------------------------------------------------------------------------------
# # For the exercises below, read the document *"matematik_yh_antonio_vektorer_matriser_utdrag"*
# # Solutions to the exercises and recorded videos can be found here: https://github.com/AntonioPrgomet/matematik_foer_yh
# 
# # If you find the exercises below very hard, do not worry. Try your best, that will be enough. 

# ### Broadcasting
# **Read the following link about broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html#basics-broadcasting**

# # Remark on Broadcasting when doing Linear Algebra calculations in Python. 

# ### From the mathematical rules of matrix addition, the operation below (m1 + m2) does not make sense. The reason is that matrix addition requires two matrices of the same size. In Python however, it works due to broadcasting rules in NumPy. So you must be careful when doing Linear Algebra calculations in Python since they do not follow the "mathematical rules". This can however easily be handled by doing some simple programming, for example validating that two matrices have the same shape is easy if you for instance want to add two matrices. 
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([1, 1])
print(m1 + m2)

# ### The example below would also not be allowed if following the "mathematical rules" in Linear Algebra. But it works due to broadcasting in NumPy. 
v1 = np.array([1, 2, 3])
print(v1 + 1)

A = np.arange(1, 5).reshape(2,2)
print(A)

b = np.array([2, 2])
print(b)

# # Vector- and matrix algebra Exercises
# **Now you are going to create a function that can be reused every time you add or multiply matrices. The function is created so that we do the addition and multiplication according to the rules of vector- and matrix algebra.**
# 
# **Create a function "add_mult_matrices" that takes two matrices as input arguments (validate that the input are of the type numpy.ndarray by using the isinstance function), a third argument that is either 'add' or 'multiply' that specifies if you want to add or multiply the matrices (validate that the third argument is either 'add' or 'multiply'). When doing matrix addition, validate that the matrices have the same size. When doing matrix multiplication, validate that the sizes conform (i.e. number of columns in the first matrix is equal to the number of rows in the second matrix).**

def add_mult_matrices(arg1,arg2,operation):
    if not isinstance(arg1, np.ndarray) or not isinstance (arg2, np.ndarray):
        raise ValueError ('the given arguments are not matrices')
    
    if operation == 'add':
        if arg1.size == arg2.size:
            return arg1 + arg2
        else:
            raise ValueError ('The matrices do not have the same sizze..')
    elif operation == 'multiply':
        if arg1.size == arg2.size:
            return arg1 * arg2
        else: 
            raise ValueError ('The matrices do not have the same shape..')

A = np.array ([[1, 2, 3]])
B = np.array ([[9, 8, 7]])
print(add_mult_matrices(A, B, 'add'))
print(add_mult_matrices(A, B, 'multiply'))

# ### Solve all the exercises in chapter 10.1 in the book "Matematik för yrkeshögskolan" by using Python. 
## 10.1.1
## Definera vektorn x enligt nedan , x = (4,3)
x = np.array([4, 3])

## (a) vilken dimension har vektron x? 
print(x.shape)

## (b) beräkna 5x?
print(5 * x)

## (c) beräkna 3x?
print(3 * x)

## (d) beräkna 5x + 3x?
five_x = 5 * x
three_x = 3 * x
print(five_x + three_x)

## (e) beräkna 8x?
print(8 * x)

## (f) beräknar 4x - x? 
four_x = 4 * x
print(four_x - x)

## (g) beräkna x^T, vilken blir den nya dimensionen efter att transponeringen utförts? 
print(x.reshape(-1, 1))

## (h) är x + x^T definerat?
print(x + x.reshape(-1, 1))
## we are supposed to get an error but due to broadcasting we get an answer which we shouldn't. 
## so, the answer is no, it's not defined since the dimensions do not match!

## (i) beräkna ||x||.
print(np.linalg.norm(x))

## -----------
## 10.1.2 
v = np.array([3, 7, 0, 11])

## (a) vilken dimension har vektorn v?
print(v.shape)

## (b) beräkna 2v?
print(2 * v)

## (c) beräkna 5v + 2v?
five_v = 5 * v
two_v = 2 * v
print(five_v + two_v)

## (d) beräkna 4v - 2v?
four_v = 4 * v
two_v = 2 * v
print(four_v - two_v)

## (e) beräkna v^T, vilken blir den nya dimensionen efter att transponeringen utförts? 
v.reshape(-1, 1)

## (f) beräkna ||v||.
print(np.linalg.norm(v))

## -----------
## 10.1.3 
## definera vektorn 
# v1 = (4, 3, 1, 5) och v2 = (2, 3, 1, 1)
v1 = ([4, 3, 1, 5])
v2 = ([2, 3, 1, 1])

## (a) beräkna ||v1||.
print(np.linalg.norm(v1))

## (b) beräkna ||v1|| - ||v2||.
v1_norm = np.linalg.norm(v1)
v2_norm = np.linalg.norm(v2)
print(v1_norm - v2_norm)

## -----------
# ### Solve all the exercises, except 10.2.4, in chapter 10.2 in the book "Matematik för yrkeshögskolan" by using Python. 

## 10.2.1 
A = np.array([[2, 1, -1,], [1, -1, 1]])
B = np.array([[4, -2, 1], [2, -4, -2]])
C = np.array([[1, 2,], [2, 1]])
D = np.array([[3, 4], [4, 3]])
E = np.array([1,2])
I = np.array([[1, 0], [0, 1]])

## (a) 2A
print(2 * A)

## (b) B - 2A
two_a = 2 * A
print(B - two_a)

## (c) 3C - 2E 
three_c = 3 * C
two_e = 2 * E
print(three_c - two_e)
## its not supposed to be defined but is due to broadcasting

## (d) 2D - 3C 
two_d = 2 * D
three_c = 3 * C
print(two_d - three_c)

## (e) D^T + 2D
transp_d = D.T
two_d = 2 * D
print(transp_d + two_d)

## (f) 2C^T - 2D^T
transp_2c = 2 * C.T 
transp_2d = 2 * D.T
print(transp_2c - transp_2d)

## (g) A^T - B 
transp_a = A.T
try:
    result_sub = transp_a - B
except:
    print("not defined!..")

## (h) AC
try: 
   result = A @ B 
except:
    print("not defined!..")

## (i) CD 
print(C @ D)

## (j) CB 
print(C @ B)

## (k) CI
print(C @ I)

## (l) AB^T 
transp_b = B.T
result = A @ transp_b
print(result)

## -----------
## 10.2.2 
A = np.array([[2, 3, 4], [5, 4, 1]])

## beräkna AA^T
transp_AA = A.T
print(A @ transp_AA)

## -----------
## 10.2.3 
A1 = np.array([[1, 2], [2, 4]])
B1 = np.array([[2, 1], [1, 3]])
C1 = np.array([[4, 3], [0, 2]])

## AB
result_ab1 = A1 @ B1
print('AB : ')
print(result_ab1)
print()

## AC 
result_ac1 = A1 @ C1
print('AC : ')
print(result_ac1)
print()

## B
print('B : ')
print(B1)
print()

## C
print('C : ')
print(C1)
print()

print('is AB EQUAL to AC?')
bool_is_equal = np.array_equal(result_ab1, result_ac1)
print(bool_is_equal)
print()

print('is B NOT EQUAL to C?')
bool_is_not_equal = not np.array_equal(B1, C1)
print(bool_is_not_equal)

## -----------
# ### Copies and Views
# Read the following link: https://numpy.org/doc/stable/user/basics.copies.html

# **Basic indexing creates a view, How can you check if v1 and v2 is a view or copy? If you change the last element in v2 to 123, will the last element in v1 be changed? Why?**
v1 = np.arange(4)
v2 = v1[-2:]
print(v1)
print(v2)

## changes made to the copy do not change the original array. 

# The base attribute of a view returns the original array while it returns None for a copy.
print(v1.base)
print(v2.base)

# The last element in v1 will be changed aswell since v2 is a view, meaning they share the same data buffer.
v2[-1] = 123
print(v1)
print(v2)

## Answer : 
## Every Numpy array have the attribute 'base' which allows us to simply know what is a view and what is a copy. 
## when using the attribute 'base', if 'None' is returned it means that the array is a copy and that the array owns the data,
## meaning that if any changes happen here, it will not reflect on the original array.
## BUT
## if an actual array is returned it means that it is a view of the orginal array and does not own the data.
## if there would be any changes happening the original array would also be affected.
## thereby v1 in this case is a copy since 'None' is returned and v2 is a view of v1 because a array is returned.
##
## Since v2 is a view of v1 any changes in v2 will be reflected on v1 because they share the same data buffer. 



