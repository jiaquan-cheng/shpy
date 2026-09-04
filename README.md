# shpy

[![CI](https://github.com/jiaquan-cheng/shpy/actions/workflows/ci.yaml/badge.svg)](https://github.com/jiaquan-cheng/shpy/actions/workflows/ci.yaml)

A lightweight static analyzer for validating NumPy array shapes at compile-time.

## Example

`shpy` can catch shape mismatches before runtime, improving code safety and maintainability. If you choose to annotate your NumPy arrays with their expected shapes, `shpy` will validate them.
```python
from typing import Annotated

a = np.ones((3, 2))
b = np.zeros((2, 3))
c: Annotated[np.ndarray, (3, 2)] = a.T
err_elem = a + b
err_matmul = a @ b.T
```
```bash
shpy examples/intro.py
````
```bash
examples/intro.py:7:0: error: [Annotation] c annotated as (3, 2), but expression has the shape (2, 3). 
examples/intro.py:8:11: error: [Elementwise] cannot combine a (3, 2) and b (2, 3) with element-wise operator. 
examples/intro.py:9:13: error: [MatMul] cannot multiply a (3, 2) and b.T (3, 2): inner dimensions must match (2 != 3). 

Found 3 error(s) across 1 file(s).
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

You can use the `--show-shapes` flag to display the inferred shapes of all expressions in the code:
```bash
shpy path/to/your/file_or_directory --show-shapes
```

## Features
- Infers shapes from NumPy array (`np.array([1, 2, 3])`) and NumPy functions (`np.zeros((3, 2))`, `a.T`).
- Validates shape annotations for NumPy arrays (`c: Annotated[np.ndarray, (3, 2)]`).
- Validates NumPy operations for shape compatibility (`a + b`, `a @ b`).
- Infers shapes for simple functions calls and function bodies (`c = custom_func(a, b)`).
- Tracks scalar variables used in shape definitions (`np.zeros((dim, 2))`).

Checkout `examples/demo.py` for a more comprehensive demonstration of `shpy`'s capabilities.

## Limitations
- Only supports a subset of NumPy arrays and functions.
- No support for dynamic shape inference (e.g., shapes that depend on runtime values).
- No control flow support (if, for, while).
- No support for nested/recursive function definitions.

If you noticed any bugs or have any feature requests, please report them on [GitHub Issues](https://github.com/jiaquan-cheng/shpy/issues).

## Development

Prerequisites: Python 3.13+, [uv](https://docs.astral.sh/uv/)

To get started locally:
```bash
git clone https://github.com/jiaquan-cheng/shpy.git
cd shpy
make setup
```
We would recommend to use the `--show-shapes` flag when developing to see the inferred shapes of all expressions in the code.

- `make` : Runs the test suite and quality checks.
- `make lint` : Runs [Ruff](https://docs.astral.sh/ruff/) and [Mypy](https://mypy-lang.org/) for code quality and type safety.
- `make format` : Auto-format code.
- `make test` : Runs [Pytest](https://pytest.org/).
