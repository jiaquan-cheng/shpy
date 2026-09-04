from typing import Annotated

import numpy as np

# Basic arrays and implicit shape inference
a = np.array([[1, 2], [3, 4]])
b = np.zeros((2, 3))
c = np.full((3, 3), 7)
d = a

# Operators, transposes, and scalar dimensions
e = a @ b - b
f: int = 5
g = f * e

dim: int = 3
h = np.zeros((dim, 2))
ht = h.T

# Slicing and random arrays
tensor = np.ones((5, 5, 5))
step_slice = tensor[0:5:2, :, ::2]
rand_arr = np.random.rand(3, 2)

# Shape manipulations
flat_arr = a.reshape(4)
infer_dim_arr = np.reshape(np.full((4, 2, 2), 5), (8, -1))

squeeze_input = np.ones((1, 2, 1, 3))
squeezed = squeeze_input.squeeze()
expanded = np.ones((2, 3)).expand_dims(1)
swapped = np.ones((2, 3, 4)).swapaxes(0, 2)

# Reductions
tensor_3d = np.ones((2, 3, 4))
summed_axis = np.sum(tensor_3d, axis=0)
mean_axis = tensor_3d.mean(axis=1)

# Functions and scope handling
global_bias = np.ones((2, 3))


def compute_transform():
    local_data = np.zeros((2, 3))
    return local_data + global_bias


def inner_helper(x):
    return x + np.ones((2, 3))


def outer_helper():
    base = np.zeros((2, 3))
    return inner_helper(base)


result = compute_transform()
chained_result = outer_helper()


# Nested functions are ignored by the analyzer to maintain scope isolation
def outer_with_nested():
    def nested_inner(x):
        return x

    return nested_inner(np.zeros((2, 3)))


nested_out_of_scope_res = outer_with_nested()


# Guard to handle recursion safely without triggering infinite loops
def recursive_function(x):
    return recursive_function(x)


recursive_guard_res = recursive_function(a)

# Error cases requiring explicit shape constraints
wrong_shape: Annotated[np.ndarray, (2, 3)] = a
wrong_matmul: Annotated[np.ndarray, (2, 3)] = a @ c
wrong_elementwise: Annotated[np.ndarray, (2, 3)] = a - b
wrong_reshape: Annotated[np.ndarray, (2, 3)] = a.reshape((-1, -1))
