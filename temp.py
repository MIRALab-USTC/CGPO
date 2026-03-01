import numpy as np

a = np.array([5,7,7,9])
a = a / 30
print(f"{a.mean() * 100:.2f}, {a.std() * 100:.2f}")