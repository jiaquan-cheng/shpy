from typing import Annotated

import numpy as np

a: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2], [3, 4]])
b: Annotated[np.ndarray, (4, 4)] = np.zeros((4, 4))
c: Annotated[np.ndarray, (2, 4)] = a @ b
