import numpy as np
import matplotlib as plt

A = np.array([[1,2,3], [4,5,6], [7,8,9], [2,3,6]])

f = np.array(1,2,3,4,5,6,7,8,9,10)

A_inv = np.linalg.inv(A)
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
