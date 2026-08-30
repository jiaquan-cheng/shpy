import pytest

SHAPE_CASES = [
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.full((2, 3), 5)
        at: Annotated[np.ndarray, (3, 2)] = a.T
        b: Annotated[np.ndarray, (3, 2)] = np.full((2, 3), 5).T
        """,
        {"a": (2, 3), "at": (3, 2), "b": (3, 2)},
        [],
        id="transpose",
    ),
]
