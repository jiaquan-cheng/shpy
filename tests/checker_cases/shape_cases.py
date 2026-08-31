import pytest

from src.shpy.checker import ShapeError

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
    pytest.param(
        """
            a: Annotated[np.ndarray, (2, 3)] = np.full((2, 3), 5)
            at: Annotated[np.ndarray, (2, 3)] = a.T
            b: Annotated[np.ndarray, (2, 3)] = np.full((2, 3), 5).T
            """,
        {"a": (2, 3), "at": (2, 3), "b": (2, 3)},
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="transpose_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (6,)] = a.reshape(6)
        c: Annotated[np.ndarray, (3, 2)] = a.reshape((3, 2))
        d: Annotated[np.ndarray, (2, 8)] = np.full((4, 2, 2), 5).reshape((2, 8))
        e: Annotated[np.ndarray, (16, 1)] = reshape(d, (16, 1))
        f: Annotated[np.ndarray, (8, 2)] = np.reshape(d, (8, -1))
        g: Annotated[np.ndarray, (2, 2, 2, 2)] = np.reshape(d, (2, 2, -1, 2))
        """,
        {
            "a": (2, 3),
            "b": (6,),
            "c": (3, 2),
            "d": (2, 8),
            "e": (16, 1),
            "f": (8, 2),
            "g": (2, 2, 2, 2),
        },
        [],
        id="reshape_operations",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (2, 3)] = a.reshape(7)
        c: Annotated[np.ndarray, (2, 3)] = a.reshape((4, 2))
        d = np.reshape(a, (-1, -1))
        """,
        {"a": (2, 3), "b": (2, 3), "c": (2, 3), "d": None},
        [
            {"line": 2, "code": ShapeError.RESHAPE_MISMATCH.value},
            {"line": 3, "code": ShapeError.RESHAPE_MISMATCH.value},
            {"line": 4, "code": ShapeError.RESHAPE_MISMATCH.value},
        ],
        id="reshape_mismatch",
    ),
    pytest.param(
        """
            a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
            b: Annotated[np.ndarray, (6,)] = a.flatten()
            c: Annotated[np.ndarray, (6,)] = np.flatten(a)
            d: Annotated[np.ndarray, (6,)] = flatten(a)
            """,
        {"a": (2, 3), "b": (6,), "c": (6,)},
        [],
        id="flatten",
    ),
    pytest.param(
        """
                a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
                b: Annotated[np.ndarray, (7,)] = a.flatten()
                c: Annotated[np.ndarray, (7,)] = np.flatten(a)
                d: Annotated[np.ndarray, (7,)] = flatten(a)
                """,
        {"a": (2, 3), "b": (7,), "c": (7,), "d": (7,)},
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="flatten_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (6,)] = a.ravel()
        c: Annotated[np.ndarray, (6,)] = np.ravel(a)
        d: Annotated[np.ndarray, (6,)] = ravel(a)
        """,
        {"a": (2, 3), "b": (6,), "c": (6,), "d": (6,)},
        [],
        id="ravel",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (7,)] = a.ravel()
        c: Annotated[np.ndarray, (7,)] = np.ravel(a)
        d: Annotated[np.ndarray, (7,)] = ravel(a)
        """,
        {"a": (2, 3), "b": (7,), "c": (7,), "d": (7,)},
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="ravel_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (1, 2, 1, 3)] = np.ones((1, 2, 1, 3))
        b: Annotated[np.ndarray, (2, 3)] = a.squeeze()
        c: Annotated[np.ndarray, (2, 3)] = np.squeeze(a)
        d: Annotated[np.ndarray, (2, 1, 3)] = a.squeeze(axis=0)
        e: Annotated[np.ndarray, (1, 2, 3)] = np.squeeze(a, axis=2)
        """,
        {
            "a": (1, 2, 1, 3),
            "b": (2, 3),
            "c": (2, 3),
            "d": (2, 1, 3),
            "e": (1, 2, 3),
        },
        [],
        id="squeeze",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (1, 2, 1, 3)] = np.ones((1, 2, 1, 3))
        b: Annotated[np.ndarray, (1, 2, 3)] = a.squeeze()
        c: Annotated[np.ndarray, (1, 2, 3)] = np.squeeze(a)
        d: Annotated[np.ndarray, (1, 2, 3)] = a.squeeze(axis=0)
        e: Annotated[np.ndarray, (2, 3)] = np.squeeze(a, axis=2)
        """,
        {
            "a": (1, 2, 1, 3),
            "b": (1, 2, 3),
            "c": (1, 2, 3),
            "d": (1, 2, 3),
            "e": (2, 3),
        },
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 5, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="squeeze_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (2, 1, 3)] = a.expand_dims(1)
        c: Annotated[np.ndarray, (2, 3, 1)] = np.expand_dims(a, 2)
        d: Annotated[np.ndarray, (1, 2, 3)] = expand_dims(a, 0)
        e: Annotated[np.ndarray, (2, 1, 3)] = a.expand_dims(axis=1)
        f: Annotated[np.ndarray, (2, 3, 1)] = np.expand_dims(a, axis=2)
        """,
        {
            "a": (2, 3),
            "b": (2, 1, 3),
            "c": (2, 3, 1),
            "d": (1, 2, 3),
            "e": (2, 1, 3),
            "f": (2, 3, 1),
        },
        [],
        id="expand_dims",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (2, 3)] = a.expand_dims(1)
        c: Annotated[np.ndarray, (2, 3)] = np.expand_dims(a, 2)
        d: Annotated[np.ndarray, (2, 3)] = expand_dims(a, 0)
        e: Annotated[np.ndarray, (2, 3)] = a.expand_dims(axis=1)
        f: Annotated[np.ndarray, (2, 3)] = np.expand_dims(a, axis=2)
        """,
        {"a": (2, 3), "b": (2, 3), "c": (2, 3), "d": (2, 3), "e": (2, 3), "f": (2, 3)},
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 5, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 6, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="expand_dims_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3, 4)] = np.ones((2, 3, 4))
        b: Annotated[np.ndarray, (4, 3, 2)] = a.swapaxes(0, 2)
        c: Annotated[np.ndarray, (2, 4, 3)] = np.swapaxes(a, 1, 2)
        d: Annotated[np.ndarray, (3, 2, 4)] = swapaxes(a, 0, 1)
        """,
        {
            "a": (2, 3, 4),
            "b": (4, 3, 2),
            "c": (2, 4, 3),
            "d": (3, 2, 4),
        },
        [],
        id="swapaxes",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3, 4)] = np.ones((2, 3, 4))
        b: Annotated[np.ndarray, (2, 3, 4)] = a.swapaxes(0, 2)
        c: Annotated[np.ndarray, (2, 3, 4)] = np.swapaxes(a, 1, 2)
        d: Annotated[np.ndarray, (2, 3, 4)] = swapaxes(a, 0, 1)
        """,
        {"a": (2, 3, 4), "b": (2, 3, 4), "c": (2, 3, 4), "d": (2, 3, 4)},
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="swapaxes_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (4, 1)] = a.resize((4, 1))
        c: Annotated[np.ndarray, (3, 2)] = np.resize(a, (3, 2))
        d: Annotated[np.ndarray, (6,)] = resize(a, 6)
        """,
        {"a": (2, 3), "b": (4, 1), "c": (3, 2), "d": (6,)},
        [],
        id="resize",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 3)] = np.ones((2, 3))
        b: Annotated[np.ndarray, (2, 3)] = a.resize((4, 1))
        c: Annotated[np.ndarray, (2, 3)] = np.resize(a, (3, 2))
        d: Annotated[np.ndarray, (2, 3)] = resize(a, 6)
        """,
        {"a": (2, 3), "b": (2, 3), "c": (2, 3), "d": (2, 3)},
        [
            {"line": 2, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value},
            {"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value},
        ],
        id="resize_mismatch",
    ),
]
