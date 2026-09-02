import ast
from textwrap import dedent

import pytest

from shpy.checker import Checker
from tests.checker_cases.annotation_cases import ANNOTATION_CASES
from tests.checker_cases.math_cases import MATH_CASES
from tests.checker_cases.shape_cases import SHAPE_CASES

TEST_CASES = ANNOTATION_CASES + MATH_CASES + SHAPE_CASES


@pytest.mark.parametrize("code, expected_symbols, expected_errors", TEST_CASES)
def test_checker(code, expected_symbols, expected_errors):
    cleaned_code = dedent(code).strip()
    tree = ast.parse(cleaned_code)
    checker = Checker()
    checker.visit(tree)

    for var, shape in expected_symbols.items():
        assert checker.shapes.get(var) == shape, (
            f"\nVariable mismatch for '{var}':\n"
            f"  Expected shape: {shape}\n"
            f"  Actual shape:   {checker.shapes.get(var)}\n"
            f"  Full table:     {checker.shapes}"
        )

    assert len(checker.errors) == len(expected_errors), (
        f"\nError count mismatch:\n"
        f"  Expected errors: {expected_errors}\n"
        f"  Actual errors:   {checker.errors}"
        f"  Full table:      {checker.shapes}"
    )

    for actual, expected in zip(checker.errors, expected_errors):
        assert actual["line"] == expected["line"], (
            f"\nError line mismatch:\n"
            f"  Expected line: {expected['line']}\n"
            f"  Actual line:   {actual['line']}\n"
            f"  Full table:    {checker.shapes}"
        )
        assert actual["code"] == expected["code"], (
            f"\nError code mismatch:\n"
            f"  Expected code: {expected['code']}\n"
            f"  Actual code:   {actual['code']}\n"
            f"  Full table:    {checker.shapes}"
        )
