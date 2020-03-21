import numpy as np
import matplotlib as plt
import random

A = np.array([[random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)],
              [random.randint(26,61),random.randint(33,69),random.randint(9,20),random.randint(850,920),random.randint(3300,11800),random.randint(100,410),random.randint(36500,390100),random.randint(2,9),random.randint(2,4),random.randint(1,32)]])

              
a = np.array([[-5.07857732e-01],
              [ 5.17846678e-01],
              [ 5.39271090e-01],
              [ 5.83602390e-03],
              [ 4.92007574e-04],
              [-9.22134148e-02],
              [ 2.63328880e-05],
              [ 1.18307014e+00],
              [-9.58213468e-01],
              [ 3.53751037e-02]])


f = np.dot(A,a)

print(f)

