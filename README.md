# Shape Checker

A lightweight static analyzer for validating NumPy array shapes at compile-time.

## Example

Without shape checker, we can have to rely on comments, which are hard to maintain and can be incorrect:
```python
a = np.array([[1, 2], [3, 4]])  # (2, 3)
```

With the shape checker, we would annotate the shape like this:
```python
from typing import Annotated

a: Annotated[np.ndarray, (2, 3)] = np.array([[1, 2], [3, 4]])
```
Running the checker will catch the mismatch before runtime:
```bash
$ uv run check examples/intro.py
[line 5] [ShapeError.ANNOTATION_MISMATCH] a annotated as (2, 3), but expression has the shape (2, 2)
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


