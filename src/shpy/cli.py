import argparse
import ast
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

from shpy.checker import ShapeChecker

try:
    __version__ = importlib.metadata.version("shpy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="A local Python shape checker.")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Files or directories to check"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--show-shapes",
        action="store_true",
        help="Display inferred NumPy shapes for all variables",
    )
    args = parser.parse_args()
    files = discover_files(args.paths)
    all_errors = []
    all_symbols: dict[Path, dict[str, tuple[Any, ...] | None]] = {}

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

        if args.show_shapes:
            all_symbols[filepath] = checker.symbol_table

    if args.show_shapes and all_symbols:
        print("\n-------- Shapes Found --------")

        for filepath in sorted(all_symbols.keys(), key=str):
            print(f"\n{filepath}:")
            symbols = all_symbols[filepath]
            symbols_with_no_shape: list = []

            if not symbols:
                print("- (no shapes inferred)")
                continue

            for var_name, shape in sorted(symbols.items()):
                if shape is not None:
                    print(f"- {var_name}: {shape}")
                else:
                    symbols_with_no_shape.append(var_name)
            print(
                f"\nno shape inferred for:\n{', '.join(symbols_with_no_shape)}"
            ) if symbols_with_no_shape else ""

        print("\n" + "-" * 30 + "\n")

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
