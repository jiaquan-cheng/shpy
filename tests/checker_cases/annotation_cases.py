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
        f: Annotated[np.ndarray, ()] = np.ones(())
        g: Annotated[np.ndarray, (0, 5)] = np.zeros((0, 5))
        """,
        {
            "a": (2, 2),
            "b": (1, 1, 1, 1),
            "c": (4, 2, 4),
            "d": (6, 3),
            "e": (2, 3),
            "f": (),
            "g": (0, 5),
        },
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
        {"a": (), "b": (), "c": (2, 3), "d": (2, 3)},
        [],
        id="variable_shape_creation",
    ),
    pytest.param(
        """
            a: float = 2
            b: int = 3
            c: Annotated[np.ndarray, (2, 1)] = np.full((a, b), 5)
            d: Annotated[np.ndarray, ()] = np.zeros((a, b))
            """,
        {"a": (), "b": (), "c": (2, 1), "d": ()},
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
        negative_slice: Annotated[np.ndarray, (2, 5)] = a[-2:, :, 0]
        negative_index: Annotated[np.ndarray, (5, 5)] = a[-1]
        reverse_slice: Annotated[np.ndarray, (5, 5, 5)] = a[::-1, ::-1, ::-1]
        """,
        {
            "a": (5, 5, 5),
            "step_slice": (3, 5, 3),
            "scalar_multi": (5,),
            "ellipsis_slice": (5, 5),
            "negative_slice": (2, 5),
            "negative_index": (5, 5),
            "reverse_slice": (5, 5, 5),
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
        negative_slice: Annotated[np.ndarray, (3, 5)] = a[-2:, :, 0]
        negative_index: Annotated[np.ndarray, (5,)] = a[-1]
        reverse_slice: Annotated[np.ndarray, (3, 3, 3)] = a[::-1, ::-1, ::-1]
        """,
        {
            "a": (5, 5, 5),
            "step_slice": (2, 5, 2),
            "scalar_multi": (5, 5),
            "ellipsis_slice": (5,),
            "negative_slice": (3, 5),
            "negative_index": (5,),
            "reverse_slice": (3, 3, 3),
        },
        [
            {"line": 2, "code": ErrorCode.ANNOTATION.value},
            {"line": 3, "code": ErrorCode.ANNOTATION.value},
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
            {"line": 5, "code": ErrorCode.ANNOTATION.value},
            {"line": 6, "code": ErrorCode.ANNOTATION.value},
            {"line": 7, "code": ErrorCode.ANNOTATION.value},
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
        grid: Annotated[np.ndarray, (2, 3)] = np.meshgrid(x, y) 
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
    pytest.param(
        """
        n = 5
        m = 3.0
        a: Annotated[np.ndarray, (n, m)] = np.zeros((5, 3))
        b: Annotated[np.ndarray, (n,)] = np.ones(n)
        """,
        {"n": (), "m": (), "a": (5, 3), "b": (5,)},
        [],
        id="scalar_variable_in_annotation_success",
    ),
    pytest.param(
        """
        n = 5
        m = 3
        f = 3.5
        a: Annotated[np.ndarray, (n, m)] = np.zeros((4, 3))
        b: Annotated[np.ndarray, (n,)] = np.ones(m)
        c: Annotated[np.ndarray, (f,)] = np.ones(3)
        d: Annotated[np.ndarray, (3,)] = np.zeros((f, 2))
        """,
        {"n": (), "m": (), "f": (), "a": (5, 3), "b": (5,), "c": (3,), "d": (3,)},
        [
            {"line": 4, "code": ErrorCode.ANNOTATION.value},
            {"line": 5, "code": ErrorCode.ANNOTATION.value},
            {"line": 6, "code": ErrorCode.VALUE.value},
            {"line": 7, "code": ErrorCode.VALUE.value},
        ],
        id="scalar_variable_in_annotation_mismatch",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (9, 9)] = np.zeros((9, 9))
        def transform(x: Annotated[np.ndarray, (10, 20)]) -> Annotated[np.ndarray, (20, 10)]:
            a = np.zeros((20, 10))
            return x.T

        b: Annotated[np.ndarray, (10, 20)] = np.zeros((10, 20))
        c: Annotated[np.ndarray, (20, 10)] = transform(b)
        """,
        {"a": (9, 9), "b": (10, 20), "c": (20, 10)},
        [],
        id="function_annotation",
    ),
    pytest.param(
        """
        def transform(x: Annotated[np.ndarray, (10, 20)]) -> Annotated[np.ndarray, (20, 10)]:
            return x.T

        a: Annotated[np.ndarray, (5, 5)] = np.zeros((5, 5))
        b: Annotated[np.ndarray, (20, 10)] = transform(a)
        c: Annotated[np.ndarray, (10, 20)] = transform(b.T)
        """,
        {"a": (5, 5), "b": (20, 10), "c": (10, 20)},
        [
            {"line": 5, "code": ErrorCode.ANNOTATION.value},
            {"line": 6, "code": ErrorCode.ANNOTATION.value},
        ],
        id="function_annotation_mismatch",
    ),
    pytest.param(
        """
        global_var = np.zeros((2, 2))
        b = np.ones((3, 3))

        def unannotated_func():
            b = np.ones((2, 2))
            return global_var + b

        res = unannotated_func()
        """,
        {
            "global_var": (2, 2),
            "b": (3, 3),
            "res": (2, 2),
        },
        [],
        id="unannotated_function_scope_isolation_success",
    ),
    pytest.param(
        """
        global_var = np.zeros((2, 2))

        def unannotated_func():
            a = np.ones((5, 5))
            b = global_var + a
            return a

        res: Annotated[np.ndarray, (4, 4)] = unannotated_func()
        """,
        {
            "global_var": (2, 2),
            "res": (4, 4),
        },
        [
            {"line": 5, "code": ErrorCode.ELEMENTWISE.value},
            {"line": 8, "code": ErrorCode.ANNOTATION.value},
        ],
        id="unannotated_function_scope_and_mismatch_negative",
    ),
    pytest.param(
        """
        def func(x=np.zeros((2, 3))):
            return x

        res = func()
        """,
        {
            "res": (2, 3),
        },
        [],
        id="unannotated_function_default_arg_inferred",
    ),
    pytest.param(
        """
        def func(x=np.zeros((2, 3))):
            return x

        res = func(np.zeros((4, 4)))
        """,
        {
            "res": (4, 4),
        },
        [],
        id="unannotated_function_default_arg_overridden",
    ),
]
