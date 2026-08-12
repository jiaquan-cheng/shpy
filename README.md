# Shape Checker

[![CI](https://github.com/jiaquan-cheng/shape-checker/actions/workflows/ci.yaml/badge.svg)](https://github.com/jiaquan-cheng/shape-checker/actions/workflows/ci.yaml)

A lightweight static analyzer for validating NumPy array shapes at compile-time.

## Example

Without shape checker, we have to rely on comments, which are hard to maintain and can be incorrect:
```python
a = np.array([[1, 2], [3, 4]])      # (2, 3)
b = np.zeros((4, 4))                # (4, 4)
c = a @ b                           # (2, 4)
```

With the shape checker, we can annotate the shape like this:
```python
from typing import Annotated

a: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2], [3, 4]])
b = np.zeros((4, 4)) # or no annotation
c = a @ b
```
Running the checker will catch the mismatch before runtime:
```bash
uv run check examples/intro.py
````
```bash
[line 5] [ANNOTATION_MISMATCH] a annotated as (2, 3), but expression has the shape (2, 2). 
[line 7] [MATMUL_MISMATCH] Cannot multiply a (2, 3) and b (4, 4): inner dimensions must match (3 != 4). 
```

## Installation

Prerequisites: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

To install the dependencies, run:

```bash
uv sync
```

## Usage

```bash
uv run check path/to/your/file.py
```

## Development

To install the dependencies for development, run:
```bash
uv sync --dev
```

- `make` : Runs the test suite and quality checks.
- `make lint` : Runs [Ruff](https://docs.astral.sh/ruff/) and [Mypy](https://mypy-lang.org/) for code quality and type safety.
- `make format` : Auto-format code.
- `make test` : Runs [Pytest](https://pytest.org/).
