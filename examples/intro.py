from typing import Annotated

import numpy as np

a: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2], [3, 4]])
