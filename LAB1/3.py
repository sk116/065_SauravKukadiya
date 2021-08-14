import numpy as np

arr1 = np.array([[1, 2], [4, 3], [5, 7]])
arr2 = np.array([[5, 1, 2], [3, 2, 1]])
arr3 = np.matmul(arr1,arr2)
print(arr3)
print()

arr4 = np.array([[2, 3, 0], [5, 6, 3], [0, 4, 1]])
arr5 = np.multiply(arr3, arr4)
print(arr5)
print()

print('Mean of matrix 1 :- ')
m = np.mean(arr1)
print(m)
print()

