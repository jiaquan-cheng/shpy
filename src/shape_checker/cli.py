import argparse
import ast
from pathlib import Path

from shape_checker.checker import ShapeChecker


def main():
    parser = argparse.ArgumentParser(description="A simple local Python type checker.")
    parser.add_argument("file", type=Path, help="Path to the Python file to check")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File '{args.file}' not found.")
        return
    code = args.file.read_text(encoding="utf-8")
    tree = ast.parse(code)
    checker = ShapeChecker()
    checker.visit(tree)

    for error in checker.errors:
        print(error["message"])
