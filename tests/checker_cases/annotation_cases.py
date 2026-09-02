import pytest

from shpy.checker import ErrorCode

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
            {"line": 1, "code": ErrorCode.ANNOTATION.value},
            {"line": 2, "code": ErrorCode.ANNOTATION.value},
            {"line": 3, "code": ErrorCode.ANNOTATION.value},
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
            {"line": 5, "code": ErrorCode.ANNOTATION.value},
        ],
        id="annotation_mismatch_detected",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (3, 3)] = a
        """,
        {"a": (2, 2), "b": (3, 3)},
        [{"line": 2, "code": ErrorCode.ANNOTATION.value}],
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
            {"line": 3, "code": ErrorCode.ANNOTATION.value},
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
        ],
        id="variable_shape_creation_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (3, 2)] = np.random.rand(3, 2)
        b: Annotated[np.ndarray, (5,)] = np.random.randn(5)
        c: Annotated[np.ndarray, (2, 2, 2)] = np.random.random((2, 2, 2))
        d: Annotated[np.ndarray, (2, 2, 2)] = np.random.random_sample((2, 2, 2))
        e: Annotated[np.ndarray, (2, 2, 2)] = np.random.ranf((2, 2, 2))
        f: Annotated[np.ndarray, (2, 2, 2)] = np.random.sample((2, 2, 2))
        g: Annotated[np.ndarray, (10, 4)] = np.random.randint(0, 10, size=(10, 4))
        h: Annotated[np.ndarray, (3, 3)] = np.random.uniform(0, 1, size=(3, 3))
        i: Annotated[np.ndarray, (4, 4)] = np.random.normal(0, 1, size=(4, 4))
        """,
        {
            "a": (3, 2),
            "b": (5,),
            "c": (2, 2, 2),
            "d": (2, 2, 2),
            "e": (2, 2, 2),
            "f": (2, 2, 2),
            "g": (10, 4),
            "h": (3, 3),
            "i": (4, 4),
        },
        [],
        id="random_generation_success",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (3, 3)] = np.random.rand(3, 2)
        b: Annotated[np.ndarray, (2,)] = np.random.randn(5)
        c: Annotated[np.ndarray, (4, 4)] = np.random.random((2, 2))
        d: Annotated[np.ndarray, (4, 4)] = np.random.random_sample((2, 2))
        e: Annotated[np.ndarray, (4, 4)] = np.random.ranf((2, 2))
        f: Annotated[np.ndarray, (4, 4)] = np.random.sample((2, 2))
        g: Annotated[np.ndarray, (5, 5)] = np.random.randint(0, 10, size=(2, 2))
        h: Annotated[np.ndarray, (5, 5)] = np.random.uniform(0, 1, size=(2, 2))
        i: Annotated[np.ndarray, (5, 5)] = np.random.normal(0, 1, size=(2, 2))
        """,
        {
            "a": (3, 3),
            "b": (2,),
            "c": (4, 4),
            "d": (4, 4),
            "e": (4, 4),
            "f": (4, 4),
            "g": (5, 5),
            "h": (5, 5),
            "i": (5, 5),
        },
        [
            {"line": 1, "code": ErrorCode.ANNOTATION.value},
            {"line": 2, "code": ErrorCode.ANNOTATION.value},
            {"line": 3, "code": ErrorCode.ANNOTATION.value},
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
            {"line": 5, "code": ErrorCode.ANNOTATION.value},
            {"line": 6, "code": ErrorCode.ANNOTATION.value},
            {"line": 7, "code": ErrorCode.ANNOTATION.value},
            {"line": 8, "code": ErrorCode.ANNOTATION.value},
            {"line": 9, "code": ErrorCode.ANNOTATION.value},
        ],
        id="random_generation_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (5, 5, 5)] = np.ones((5, 5, 5))
        step_slice: Annotated[np.ndarray, (3, 5, 3)] = a[0:5:2, :, ::2]
        scalar_multi: Annotated[np.ndarray, (5,)] = a[1, 2]
        ellipsis_slice: Annotated[np.ndarray, (5, 5)] = a[..., 0]
        """,
        {
            "a": (5, 5, 5),
            "step_slice": (3, 5, 3),
            "scalar_multi": (5,),
            "ellipsis_slice": (5, 5),
        },
        [],
        id="slicing_edge_cases_success",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (5, 5, 5)] = np.ones((5, 5, 5))
        step_slice: Annotated[np.ndarray, (2, 5, 2)] = a[0:5:2, :, ::2]
        scalar_multi: Annotated[np.ndarray, (5, 5)] = a[1, 2]
        ellipsis_slice: Annotated[np.ndarray, (5,)] = a[..., 0]
        """,
        {
            "a": (5, 5, 5),
            "step_slice": (2, 5, 2),
            "scalar_multi": (5, 5),
            "ellipsis_slice": (5,),
        },
        [
            {"line": 2, "code": ErrorCode.ANNOTATION.value},
            {"line": 3, "code": ErrorCode.ANNOTATION.value},
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
        ],
        id="slicing_edge_cases_mismatch",
    ),
    pytest.param(
        """
        seq: Annotated[np.ndarray, (10,)] = np.arange(0, 10, 1)
        space: Annotated[np.ndarray, (5,)] = np.linspace(0, 1, 5)
        log_space: Annotated[np.ndarray, (50,)] = np.logspace(0, 2, 50)
        geom_space: Annotated[np.ndarray, (10,)] = np.geomspace(1, 1000, 10)
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        grid: Annotated[np.ndarray, (2, 3)] = np.meshgrid(x, y) # Default indexing='xy' -> (len(y), len(x))
        """,
        {
            "seq": (10,),
            "space": (5,),
            "log_space": (50,),
            "geom_space": (10,),
            "x": (3,),
            "y": (2,),
            "grid": (2, 3),
        },
        [],
        id="sequence_and_grid_functions_success",
    ),
    pytest.param(
        """
        seq: Annotated[np.ndarray, (5,)] = np.arange(0, 10, 1)
        space: Annotated[np.ndarray, (10,)] = np.linspace(0, 1, 5)
        log_space: Annotated[np.ndarray, (10,)] = np.logspace(0, 2, 50)
        geom_space: Annotated[np.ndarray, (50,)] = np.geomspace(1, 1000, 10)
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        grid: Annotated[np.ndarray, (3, 2)] = np.meshgrid(x, y)
        """,
        {
            "seq": (5,),
            "space": (10,),
            "log_space": (10,),
            "geom_space": (50,),
            "x": (3,),
            "y": (2,),
            "grid": (3, 2),
        },
        [
            {"line": 1, "code": ErrorCode.ANNOTATION.value},
            {"line": 2, "code": ErrorCode.ANNOTATION.value},
            {"line": 3, "code": ErrorCode.ANNOTATION.value},
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
            {"line": 7, "code": ErrorCode.ANNOTATION.value},
        ],
        id="sequence_and_grid_functions_mismatch",
    ),
]
