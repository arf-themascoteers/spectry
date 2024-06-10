import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

x1 = np.pad(x, ((0, 0), (1, 1)), mode='edge')

print(x1)