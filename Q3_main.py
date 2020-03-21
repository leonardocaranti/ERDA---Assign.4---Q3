import numpy as np
import matplotlib as plt



A = np.array([[1,2,3], [4,5,6], [7,8,9], [2,3,6]])

f = np.array([func(A[0,0], A[0,1], A[0,2]), func(A[1,0], A[1,1], A[1,2]), func(A[2,0], A[2,1], A[2,2]), func(A[3,0], A[3,1], A[3,2])])
A_t = A.transpose()

A_t_a_inv = np.linalg.inv(A_t_A)
a = np.dot(A_t_a_inv, np.dot(A_t, f))

print(a)

def squares(a_s):
    sum = 0
    for i in range(len(a_s)):
        sum += (np.dot(a_s[i], A[i]) - f[i])
    return sum

print()
x = sp.optimize.least_squares(squares, [3.6, 4.11, -7.9]).x()
print(x)
