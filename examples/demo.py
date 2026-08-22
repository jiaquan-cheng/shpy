from typing import Annotated

import numpy as np

a = np.array([[1, 2], [3, 4]])
b: Annotated[np.ndarray, (2, 3)] = np.zeros((2, 3))
c: Annotated[np.ndarray, (3, 3)] = np.full((3, 3), 7)
d: Annotated[np.ndarray, (2, 2)] = a
e: Annotated[np.ndarray, (2, 3)] = a @ b - b
f: int = 5
g: Annotated[np.ndarray, (2, 3)] = f * e
dim: int = 3
h: Annotated[np.ndarray, (3, 2)] = np.zeros((dim, 2))
ht: Annotated[np.ndarray, (3, 3)] = h.T


wrong_shape: Annotated[np.ndarray, (2, 3)] = a
wrong_shape2: Annotated[np.ndarray, (2, 3)] = a @ c
wrong_shape3: Annotated[np.ndarray, (2, 3)] = a - b
