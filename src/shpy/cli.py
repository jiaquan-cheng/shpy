import argparse
import ast
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

from shpy.checker import Checker

try:
    __version__ = importlib.metadata.version("shpy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


def main() -> None:
    """CLI entry point for the local Python shape checker."""
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
    all_errors: list[str] = []
    all_shapes: dict[Path, dict[str, tuple[Any, ...] | None]] = {}
    all_scalars: dict[Path, dict[str, int | float]] = {}

    for filepath in files:
        file_errors, shapes, scalars = _process_file(filepath)
        all_errors.extend(file_errors)
        if shapes:
            all_shapes[filepath] = shapes
        if scalars:
            all_scalars[filepath] = scalars

    if args.show_shapes and all_shapes:
        _print_shape_report(all_shapes, all_scalars)

    all_errors.sort()
    for err in all_errors:
        print(err)

    if all_errors:
        print(f"\nFound {len(all_errors)} error(s) across {len(files)} file(s).")
        sys.exit(1)
    else:
        print(f"Success: Checked {len(files)} file(s), no shape errors found.")
        sys.exit(0)


def discover_files(paths: list[Path]) -> list[Path]:
    """Discovers and collects all Python (.py) source files from the provided paths.

    Args:
        paths: A list of file or directory Path objects to scan.

    Returns:
        A list of resolved Path objects pointing to valid Python source files.
    """
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


def _process_file(
    filepath: Path,
) -> tuple[list[str], dict[str, tuple[Any, ...] | None], dict[str, int | float]]:
    """Processes a single source file, running syntax checks and shape analysis.

    Args:
        filepath: The path to the file to check.

    Returns:
        A tuple containing a list of formatted error messages,
        the extracted shapes dictionary, and the extracted scalars dictionary.
    """
    errors: list[str] = []
    try:
        code = filepath.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(filepath))
    except SyntaxError as e:
        errors.append(f"{filepath}:{e.lineno}: error: [SyntaxError] {e.msg} ")
        return errors, {}, {}

    checker = Checker()
    checker.visit(tree)

    for error in checker.errors:
        line = error["line"]
        col = error["col"]
        code = error["code"]
        msg = error["message"]
        errors.append(f"{filepath}:{line}:{col}: error: [{code}] {msg}")

    return errors, checker.env.shapes, checker.env.scalar_values


def _print_shape_report(
    all_shapes: dict[Path, dict[str, tuple[Any, ...] | None]],
    all_scalars: dict[Path, dict[str, int | float]],
) -> None:
    """Prints a formatted report of all inferred shapes and scalar values across files.

    Args:
        all_shapes: Mapping of file paths to their symbol shape definitions.
        all_scalars: Mapping of file paths to their tracked scalar variables.
    """
    print("\n-------- Symbol State (Shapes & Scalars) --------")

    for filepath in sorted(all_shapes.keys(), key=str):
        print(f"\n{filepath}:")
        symbols = all_shapes[filepath]
        scalars = all_scalars.get(filepath, {})

        if scalars:
            print("  Scalars:")
            for name, val in sorted(scalars.items()):
                print(f"    - {name} = {val}")

        if symbols:
            print("  Shapes:")
            symbols_with_no_shape = []
            for var_name, shape in sorted(symbols.items()):
                if shape is not None:
                    print(f"    - {var_name}: {shape}")
                else:
                    symbols_with_no_shape.append(var_name)

            if symbols_with_no_shape:
                print(
                    f"    - no shape inferred for: {', '.join(symbols_with_no_shape)}"
                )

        if not symbols and not scalars:
            print("  - (no symbols tracked)")

    print("\n" + "-" * 30 + "\n")


if __name__ == "__main__":
    main()
