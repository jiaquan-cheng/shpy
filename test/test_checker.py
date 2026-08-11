import ast
from textwrap import dedent

import pytest

from shape_checker.checker import ShapeChecker

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
        id="explicit_annotations_valid",
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
        [{"line": 3, "code": "ANNOTATION_MISMATCH"}],
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
        [{"line": 4, "code": "ANNOTATION_MISMATCH"}],
        id="variable_propagation_with_mismatch",
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
        assert expected["code"] in actual["message"]
