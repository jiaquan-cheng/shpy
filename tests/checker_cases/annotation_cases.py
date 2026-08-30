import pytest

from shpy.checker import ShapeError

ANNOTATION_CASES = [
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (1, 1, 1, 1)] = np.zeros((1, 1, 1, 1))
        c: Annotated[np.ndarray, (4, 2, 4)] = np.ones((4, 2, 4))
        d: Annotated[np.ndarray, (6, 3)] = np.empty((6, 3), 5)
        e: Annotated[np.ndarray, (2, 3)] = np.full((2, 3), 5)
        """,
        {"a": (2, 2), "b": (1, 1, 1, 1), "c": (4, 2, 4), "d": (6, 3), "e": (2, 3)},
        [],
        id="explicit_annotations",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.zeros((1, 1, 1, 1))
        c = np.ones((4, 2, 4))
        d = np.empty((6, 3), 5)
        e = np.full((2, 3), 5)
        """,
        {"a": (2, 2), "b": (1, 1, 1, 1), "c": (4, 2, 4), "d": (6, 3), "e": (2, 3)},
        [],
        id="implicit_array_inference",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 2)] = b
        c = b
        """,
        {"a": (2, 2), "b": (2, 2), "c": (2, 2)},
        [],
        id="variable_propagation",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4], [5, 6]])
        b: Annotated[np.ndarray, (1, 1, 1, 1)] = np.zeros((1, 1))
        c: Annotated[np.ndarray, (4, 2, 4)] = np.ones((4, 2, 1, 1))
        d: Annotated[np.ndarray, (6, 3)] = np.empty((2, 3), 5)
        e: Annotated[np.ndarray, (2, 3)] = np.full((2, 1), 5)
        """,
        {"a": (2, 2), "b": (1, 1, 1, 1), "c": (4, 2, 4), "d": (6, 3), "e": (2, 3)},
        [
            {"line": 1, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 5, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="annotation_mismatch_detected",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (3, 3)] = a
        """,
        {"a": (2, 2), "b": (3, 3)},
        [{"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value}],
        id="variable_propagation_mismatch",
    ),
    pytest.param(
        """
        a: float = 2
        b: int = 3
        c: Annotated[np.ndarray, (2, 3)] = np.full((a, b), 5)
        d: Annotated[np.ndarray, (2, 3)] = np.zeros((a, b))
        """,
        {"a": (1,), "b": (1,), "c": (2, 3), "d": (2, 3)},
        [],
        id="variable_shape_creation",
    ),
    pytest.param(
        """
            a: float = 2
            b: int = 3
            c: Annotated[np.ndarray, (2, 1)] = np.full((a, b), 5)
            d: Annotated[np.ndarray, (1, )] = np.zeros((a, b))
            """,
        {"a": (1,), "b": (1,), "c": (2, 1), "d": (1,)},
        [
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="variable_shape_creation_mismatch",
    ),
]
