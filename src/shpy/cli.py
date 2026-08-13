import argparse
import ast
import sys
from pathlib import Path

from shpy.checker import ShapeChecker


def discover_files(paths: list[Path]) -> list[Path]:
    """Finds all .py files."""
    files_to_check = []
    for path in paths:
        if not path.exists():
            print(f"Error: Path '{path}' not found.", file=sys.stderr)
            continue

        if path.is_file() and path.suffix == ".py":
            files_to_check.append(path)
        elif path.is_dir():
            for subpath in path.rglob("*.py"):
                # Skip hidden directories like .git, .venv, __pycache__
                if not any(part.startswith(".") for part in subpath.parts):
                    files_to_check.append(subpath)

    return files_to_check


def main():
    parser = argparse.ArgumentParser(description="A local Python shape checker.")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Files or directories to check"
    )
    args = parser.parse_args()
    files = discover_files(args.paths)
    all_errors = []

    for filepath in files:
        try:
            code = filepath.read_text(encoding="utf-8")
            tree = ast.parse(code, filename=str(filepath))
        except SyntaxError as e:
            all_errors.append(f"{filepath}:{e.lineno}: error: [SyntaxError] {e.msg} ")
            continue

        checker = ShapeChecker()
        checker.visit(tree)

        for error in checker.errors:
            line = error["line"]
            col = error["col"]
            code = error["code"]
            msg = error["message"]
            all_errors.append(f"{filepath}:{line}:{col}: error: [{code}] {msg}")

    all_errors.sort()
    for err in all_errors:
        print(err)

    if all_errors:
        print(f"\nFound {len(all_errors)} error(s) across {len(files)} file(s).")
        sys.exit(1)
    else:
        print(f"Success: Checked {len(files)} file(s), no shape errors found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
