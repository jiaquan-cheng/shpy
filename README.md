# shpy

[![CI](https://github.com/jiaquan-cheng/shpy/actions/workflows/ci.yaml/badge.svg)](https://github.com/jiaquan-cheng/shpy/actions/workflows/ci.yaml)

A lightweight static analyzer for validating NumPy array shapes at compile-time.

## Example

Without `shpy`, we have to rely on comments, which are hard to maintain and can be incorrect:
```python
a = np.array([[1, 2], [3, 4]])  # (2, 3)
b = np.zeros((4, 4))  # (4, 4)
c = a @ b  # (2, 4)
```

With `shpy`, we can annotate shapes like this:
```python
from typing import Annotated

a: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2], [3, 4]])
b: Annotated[np.ndarray, (4, 4)] = np.zeros((4, 4))
c: Annotated[np.ndarray, (2, 4)] = a @ b
```
Running the checker will catch the mismatch before runtime:
```bash
shpy examples/intro.py
````
```bash
examples/intro.py:5:0: error: [AnnotationMismatch] a annotated as (2, 3), but expression has the shape (2, 2). 
examples/intro.py:7:4: error: [MatMulMismatch] cannot multiply a (2, 3) and b (4, 4): inner dimensions must match (3 != 4). 

Found 2 error(s) across 1 file(s).
```

## Installation

Prerequisites: Python 3.13+

To install `shpy` directly:

```bash
pip install git+https://github.com/jiaquan-cheng/shpy.git
```
## Usage


```bash
shpy path/to/your/file_or_directory
```

## Development

Prerequisites: Python 3.13+, [uv](https://docs.astral.sh/uv/)

To get started locally:
```bash
git clone https://github.com/jiaquan-cheng/shpy.git
cd shpy
make setup
```

- `make` : Runs the test suite and quality checks.
- `make lint` : Runs [Ruff](https://docs.astral.sh/ruff/) and [Mypy](https://mypy-lang.org/) for code quality and type safety.
- `make format` : Auto-format code.
- `make test` : Runs [Pytest](https://pytest.org/).
