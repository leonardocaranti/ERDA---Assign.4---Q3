import numpy as np
import matplotlib as plt

A = np.array([[60.30,58.37,17.40,880,8800,268,233000,7,2,8], 
              [60.30,63.69,16.80,880,8200,292,233000,8,2,5], 
              [35.80,33.62,12.60,850,3500,142,65317,3,2,1], 
              [35.80,42.12,12.60,850,4300,188,76900,3,2,5],
              [64.44,70.67,19.40,920,11500,408,390100,9,4,5],
              [60.90,63.80,18.50,900,11800,320,297500,5,2,15], 
              [60.10,62.80,16.30,920,11500,294,252650,6,2,13], 
              [60.10,68.30,17.02,903,12000,344,254100,8,2,8], 
              [26.00,31.68,9.86,850,3300,88,36500,2,2,17],
              [28.72,36.24,10.55,850,3300,100,45000,2,2,32]])

f = np.array([[119600],[121870],[38150],[42493],[183500],[142430],[115300],[119950],[21810],[28080]])

A_inv = np.linalg.inv(A)
a = np.dot(A_inv, f)


#737-800 for A330-200
A[0] = np.array([35.80,39.47,12.55,842,5436,189,79016,3,2,5])

f_new = np.dot(A, a)

print("estimated weight of the new aircraft is:")
print(f_new[0])
