import numpy as np

a = np.asarray([1, 2])
b = np.asarray([3, 4])
c = np.asarray([5, 6])

d = np.stack([a, b, c], axis=1).ravel()
print(d)