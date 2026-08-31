import pytest

from shpy.checker import ShapeError

MATH_CASES = [
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 2)] = np.array([[5, 6], [7, 8]])
        c: Annotated[np.ndarray, (2, 2)] = np.array([[9, 10], [11, 12]])
        d: Annotated[np.ndarray, (2, 2)] = a + b - c
        e: Annotated[np.ndarray, (2, 2)] = a / b * c
        f: float = 2
        g: Annotated[np.ndarray, (2, 2)] = f * a
        """,
        {
            "a": (2, 2),
            "b": (2, 2),
            "c": (2, 2),
            "d": (2, 2),
            "e": (2, 2),
            "f": (1,),
            "g": (2, 2),
        },
        [],
        id="binary_operation",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 3)] = np.array([[5, 6, 7], [8, 9, 10]])
        c: Annotated[np.ndarray, (2, 3)] = a + b
        d: Annotated[np.ndarray, (2, 3)] = a - b
        e: Annotated[np.ndarray, (2, 3)] = a * b
        f: Annotated[np.ndarray, (2, 3)] = a / b
        """,
        {"a": (2, 2), "b": (2, 3), "c": (2, 3), "d": (2, 3), "e": (2, 3), "f": (2, 3)},
        [
            {"line": 3, "code": ShapeError.ELEMENTWISE_MISMATCH.value},
            {"line": 4, "code": ShapeError.ELEMENTWISE_MISMATCH.value},
            {"line": 5, "code": ShapeError.ELEMENTWISE_MISMATCH.value},
            {"line": 6, "code": ShapeError.ELEMENTWISE_MISMATCH.value},
        ],
        id="binary_operation_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 3)] = np.array([[5, 6, 7], [8, 9, 10]])
        c: Annotated[np.ndarray, (2, 3)] = a @ b
        """,
        {"a": (2, 2), "b": (2, 3), "c": (2, 3)},
        [],
        id="matrix_multiplication",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (3, 3)] = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        c: Annotated[np.ndarray, (2, 3)] = a @ b
        """,
        {"a": (2, 2), "b": (3, 3), "c": (2, 3)},
        [{"line": 3, "code": ShapeError.MATMUL_MISMATCH.value}],
        id="matrix_multiplication_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 3)] = np.array([[5, 6, 7], [8, 9, 10]])
        c: Annotated[np.ndarray, (3, 3)] = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
        d: Annotated[np.ndarray, (2, 3)] = a @ b @ c - b
        """,
        {"a": (2, 2), "b": (2, 3), "c": (3, 3), "d": (2, 3)},
        [],
        id="matrix_operation_chain",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2, 2)] = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b: Annotated[np.ndarray, (2, 2, 3)] = np.array([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]])
        c: Annotated[np.ndarray, (2, 2, 3)] = a @ b
        """,
        {"a": (2, 2, 2), "b": (2, 2, 3), "c": (2, 2, 3)},
        [],
        id="matrix_multiplication_multi_dimensions",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2, 2)] = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b: Annotated[np.ndarray, (3, 2, 3)] = np.array([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]], [[21, 22, 23], [24, 25, 26]]])
        c: Annotated[np.ndarray, (2, 3)] = a @ b
        d: Annotated[np.ndarray, (7, 3)] = 1
        e: Annotated[np.ndarray, (2, 3)] = a @ d
        """,
        {"a": (2, 2, 2), "b": (3, 2, 3), "c": (2, 3), "d": (7, 3), "e": (2, 3)},
        [
            {"line": 3, "code": ShapeError.MATMUL_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 5, "code": ShapeError.MATMUL_MISMATCH.value},
        ],
        id="matrix_multiplication_multi_dimensions_mismatch",
    ),
    pytest.param(
        """
        a: float = 2
        b: int = 3
        c: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2, 3], [4, 5, 6]])
        d: Annotated[np.ndarray, (2, 3)] = a * c
        e: Annotated[np.ndarray, (2, 3)] = b * c
        """,
        {"a": (1,), "b": (1,), "c": (2, 3), "d": (2, 3), "e": (2, 3)},
        [],
        id="scalar_multiplication",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        b: Annotated[np.ndarray, (3, 4)] = np.sum(a, axis=0)
        c: Annotated[np.ndarray, (2, 4)] = np.mean(a, axis=1)
        d: Annotated[np.ndarray, (2, 3)] = np.prod(a, axis=-1)
        e: Annotated[np.ndarray, (3, 4)] = a.sum(axis=0)
        f: Annotated[np.ndarray, (2, 4)] = a.mean(axis=1)
        g: Annotated[np.ndarray, (1,)] = np.sum(a)
        """,
        {
            "a": (2, 3, 4),
            "b": (3, 4),
            "c": (2, 4),
            "d": (2, 3),
            "e": (3, 4),
            "f": (2, 4),
            "g": (1,),
        },
        [],
        id="reduction_operations_success",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        b: Annotated[np.ndarray, (2, 4)] = np.sum(a, axis=0)
        c: Annotated[np.ndarray, (3, 4)] = np.mean(a, axis=1)
        d: Annotated[np.ndarray, (2, 4)] = np.prod(a, axis=-1)
        e: Annotated[np.ndarray, (2, 4)] = a.sum(axis=0)
        f: Annotated[np.ndarray, (3, 4)] = a.mean(axis=1)
        g: Annotated[np.ndarray, (2,)] = np.sum(a)
        """,
        {
            "a": (2, 3, 4),
            "b": (2, 4),
            "c": (3, 4),
            "d": (2, 4),
            "e": (2, 4),
            "f": (3, 4),
            "g": (2,),
        },
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 5, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 6, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 7, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="reduction_operations_mismatch",
    ),
]
