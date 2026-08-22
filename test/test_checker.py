import ast
from textwrap import dedent

import pytest

from shpy.checker import ShapeChecker, ShapeError

TEST_CASES = [
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 3)] = np.zeros((2, 3))
        c: Annotated[np.ndarray, (4, 2, 4)] = np.ones((4, 2, 4))
        d: Annotated[np.ndarray, (2, 3)] = np.full((2, 3), 5)
        """,
        {"a": (2, 2), "b": (2, 3), "c": (4, 2, 4), "d": (2, 3)},
        [],
        id="explicit_annotations",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        """,
        {"a": (2, 2), "b": (2, 3)},
        [],
        id="implicit_array_inference",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
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
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (3, 3)] = np.array([[1, 2], [3, 4]])
        """,
        {"a": (3, 3)},
        [{"line": 3, "code": ShapeError.ANNOTATION_MISMATCH.value}],
        id="annotation_mismatch_detected",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (3, 3)] = a
        """,
        {"a": (2, 2), "b": (3, 3)},
        [{"line": 4, "code": ShapeError.ANNOTATION_MISMATCH.value}],
        id="variable_propagation_mismatch",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 2)] = np.array([[5, 6], [7, 8]])
        c: Annotated[np.ndarray, (2, 2)] = np.array([[9, 10], [11, 12]])
        d: Annotated[np.ndarray, (2, 2)] = a + b - c
        e: Annotated[np.ndarray, (2, 2)] = a / b * c
        """,
        {"a": (2, 2), "b": (2, 2), "c": (2, 2), "d": (2, 2), "e": (2, 2)},
        [],
        id="binary_operation",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 3)] = np.array([[5, 6, 7], [8, 9, 10]])
        c: Annotated[np.ndarray, (2, 3)] = a + b
        """,
        {"a": (2, 2), "b": (2, 3), "c": (2, 3)},
        [{"line": 5, "code": ShapeError.ELEMENTWISE_MISMATCH.value}],
        id="binary_operation_mismatch",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
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
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (3, 3)] = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        c: Annotated[np.ndarray, (2, 3)] = a @ b
        """,
        {"a": (2, 2), "b": (3, 3), "c": (2, 3)},
        [{"line": 5, "code": ShapeError.MATMUL_MISMATCH.value}],
        id="matrix_multiplication_mismatch",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2)] = np.array([[1, 2], [3, 4]])
        b: Annotated[np.ndarray, (2, 3)] = np.array([[5, 6, 7], [8, 9, 10]])
        c: Annotated[np.ndarray, (3, 3)] = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
        d: Annotated[np.ndarray, (2, 3)] = a @ b @ c - b
        """,
        {"a": (2, 2), "b": (2, 3), "c": (3, 3), "d": (2, 3)},
        [],
        id="matrix_multiplication_2",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
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
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2, 2)] = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b: Annotated[np.ndarray, (3, 2, 3)] = np.array([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]], [[21, 22, 23], [24, 25, 26]]])
        c: Annotated[np.ndarray, (2, 3)] = a @ b
        """,
        {"a": (2, 2, 2), "b": (3, 2, 3), "c": (2, 3)},
        [{"line": 5, "code": ShapeError.MATMUL_MISMATCH.value}],
        id="matrix_multiplication_multi_dimensions_mismatch",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: Annotated[np.ndarray, (2, 2, 2)] = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b: Annotated[np.ndarray, (3, 3)] = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])
        c: Annotated[np.ndarray, (2, 3)] = a @ b
        """,
        {"a": (2, 2, 2), "b": (3, 3), "c": (2, 3)},
        [{"line": 5, "code": ShapeError.MATMUL_MISMATCH.value}],
        id="matrix_multiplication_multi_dimensions_mismatch_2",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: float = 2
        b: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2, 3], [4, 5, 6]])
        c: Annotated[np.ndarray, (2, 3)] = a * b
        """,
        {"a": (1,), "b": (2, 3), "c": (2, 3)},
        [],
        id="scalar_multiplication",
    ),
    pytest.param(
        """
        import numpy as np
        from typing import Annotated
        a: float = 2
        b: Annotated[np.ndarray, (2, 3)] = np.full((a, 3), 5)
        c: Annotated[np.ndarray, (2, 3)] = np.zeros((a, 3))
        """,
        {"a": (1,), "b": (2, 3), "c": (2, 3)},
        [],
        id="variable_shape_creation",
    ),
]


@pytest.mark.parametrize("code, expected_symbols, expected_errors", TEST_CASES)
def test_shape_checker(code, expected_symbols, expected_errors):
    cleaned_code = dedent(code).strip()
    tree = ast.parse(cleaned_code)
    checker = ShapeChecker()
    checker.visit(tree)

    for var, shape in expected_symbols.items():
        assert checker.symbol_table.get(var) == shape, (
            f"Variable '{var}' has incorrect shape."
        )

    assert len(checker.errors) == len(expected_errors)

    for actual, expected in zip(checker.errors, expected_errors):
        assert actual["line"] == expected["line"]
        assert actual["code"] == expected["code"]
