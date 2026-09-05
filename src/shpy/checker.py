import ast
import math
from enum import Enum
from typing import Any

from shpy.environment import Environment


class ErrorCode(Enum):
    ANNOTATION = "Annotation"
    ELEMENTWISE = "Elementwise"
    MATMUL = "MatMul"
    RESHAPE = "Reshape"
    VALUE = "Value"


class Checker(ast.NodeVisitor):
    """AST visitor that walks through Python code to infer and check NumPy shapes"""

    def __init__(self) -> None:
        self.env = Environment()
        self.functions: dict[str, ast.FunctionDef] = {}
        self.errors: list[dict[str, Any]] = []
        self.active_calls: set[str] = set()

        self.call_handlers = {
            "array": lambda node: (
                self._extract_literal_shape(node.args[0]) if node.args else None
            ),
            "zeros": lambda node: (
                self._extract_expr_shape(node.args[0]) if node.args else None
            ),
            "ones": lambda node: (
                self._extract_expr_shape(node.args[0]) if node.args else None
            ),
            "empty": lambda node: (
                self._extract_expr_shape(node.args[0]) if node.args else None
            ),
            "full": lambda node: (
                self._extract_expr_shape(node.args[0]) if node.args else None
            ),
            "reshape": lambda node: self._infer_reshape(node),
            "flatten": lambda node: self._infer_flatten(node),
            "ravel": lambda node: self._infer_flatten(node),
            "squeeze": lambda node: self._infer_squeeze(node),
            "expand_dims": lambda node: self._infer_expand_dims(node),
            "swapaxes": lambda node: self._infer_swapaxes(node),
            "resize": lambda node: self._infer_resize(node),
            "rand": lambda node: self._infer_random_shape(node),
            "randn": lambda node: self._infer_random_shape(node),
            "random": lambda node: self._infer_random_shape(node),
            "random_sample": lambda node: self._infer_random_shape(node),
            "ranf": lambda node: self._infer_random_shape(node),
            "sample": lambda node: self._infer_random_shape(node),
            "randint": lambda node: self._infer_randint_shape(node),
            "uniform": lambda node: self._infer_random_distribution_shape(node),
            "normal": lambda node: self._infer_random_distribution_shape(node),
            "sum": lambda node: self._infer_reduction(node),
            "mean": lambda node: self._infer_reduction(node),
            "prod": lambda node: self._infer_reduction(node),
            "min": lambda node: self._infer_reduction(node),
            "max": lambda node: self._infer_reduction(node),
            "arange": lambda node: self._infer_arange(node),
            "linspace": lambda node: self._infer_linspace(node),
            "logspace": lambda node: self._infer_linspace(node),
            "geomspace": lambda node: self._infer_linspace(node),
            "meshgrid": lambda node: self._infer_meshgrid(node),
        }

        self.attr_handlers = {"T": lambda node: self._infer_transpose(node)}

        self.binop_handlers = {
            ast.MatMult: self._infer_matmult_shape,
            ast.Add: self._infer_elementwise_shape,
            ast.Sub: self._infer_elementwise_shape,
            ast.Mult: self._infer_elementwise_shape,
            ast.Div: self._infer_elementwise_shape,
        }

    @property
    def shapes(self) -> dict[str, tuple[Any, ...] | None]:
        """Convenience property for tests and CLI to access global shapes."""
        return self.env.shapes

    @property
    def scalar_values(self) -> dict[str, int | float]:
        """Convenience property for tests and CLI to access global scalar values."""
        return self.env.scalar_values

    # ==========================================
    # 1. Entry Points
    # ==========================================

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handles explicit annotated variable assignments."""

        if not isinstance(node.target, ast.Name):
            return

        var_name = node.target.id

        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, (int, float)
        ):
            self.env.set_scalar(var_name, node.value.value)

        annotated_shape = self._extract_annotation_shape(node.annotation)
        inferred_shape = self._infer_shape(node.value) if node.value else None

        self.env.set_shape(
            var_name, annotated_shape if annotated_shape is not None else inferred_shape
        )

        if (
            annotated_shape is not None
            and inferred_shape is not None
            and annotated_shape != inferred_shape
        ):
            self._log_error(
                node,
                ErrorCode.ANNOTATION,
                (
                    f"{var_name} annotated as {annotated_shape}, "
                    f"but expression has the shape {inferred_shape}. "
                ),
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles implicit variable assignments."""

        if (
            node.value
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env.set_scalar(target.id, node.value.value)

        inferred_shape = self._infer_shape(node.value) if node.value else None

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.env.set_shape(target.id, inferred_shape)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Tracks custom function definitions."""
        if self.env.parent is None:
            self.functions[node.name] = node

        # no generic visit, evaluate function only when called

    # ==========================================
    # 2. Core Inference
    # Routes AST nodes to the appropriate 3. Specialized Handlers.
    # ==========================================

    def _infer_shape(self, node: ast.AST | None) -> tuple[int | str, ...] | None:
        """Extracts and infers shape properties from an arbitrary AST expression.

        Args:
            node: The AST node to evaluate, or None.

        Returns:
            A tuple representing the shape dimensions, or None if uninferrable.
        """
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return self.env.get_shape(node.id)

        if isinstance(node, ast.Call):
            return self._infer_call_shape(node)

        if isinstance(node, ast.BinOp):
            return self._infer_binop_shape(node)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return ()

        if isinstance(node, ast.Attribute):
            return self._infer_attribute_shape(node)

        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)

        return None

    def _infer_call_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers the shape of a function or method call node.
        Handles both user-defined function and NumPy functions.

        Args:
            node: The `ast.Call` node representing the function invocation.

        Returns:
            The inferred shape tuple, or None if the function is uninferrable.
        """
        func_name = ast.unparse(node.func)

        # Check if it's a user-defined function call
        if func_name in self.functions:
            return self._infer_user_function_call(node, self.functions[func_name])

        for suffix in self.call_handlers:
            if func_name.endswith(suffix):
                handler = self.call_handlers[suffix]
                return handler(node)
        return None

    def _infer_attribute_shape(
        self, node: ast.Attribute
    ) -> tuple[int | str, ...] | None:
        """Handles attribute-based shape changes, such as `.T` (transpose).

        Args:
            node: The `ast.Attribute` node being evaluated.

        Returns:
            The transformed shape tuple, or None if uninferrable.
        """
        handler = self.attr_handlers.get(node.attr)
        if handler:
            return handler(node)
        return None

    def _infer_binop_shape(self, node: ast.BinOp) -> tuple[int | str, ...] | None:
        """Infers and validates shapes for binary operations like `+`, `-`, and `@`.

        Args:
            node: The `ast.BinOp` node containing left, right operands, and operator.

        Returns:
            The broadcasted or validated result shape tuple, or None on failure.
        """
        left_shape = self._infer_shape(node.left)
        right_shape = self._infer_shape(node.right)

        if left_shape is None or right_shape is None:
            return None

        handler = self.binop_handlers.get(type(node.op))
        if handler:
            return handler(node, left_shape, right_shape)
        return None

    def _infer_user_function_call(
        self, node: ast.Call, func_node: ast.FunctionDef
    ) -> tuple[int | str, ...] | None:
        """Infers the shape returned by a user-defined function call.
        Args:
            node: The `ast.Call` node representing the invocation.
            func_node: The `ast.FunctionDef` node of the target function.

        Returns:
            The inferred return shape tuple, or None if uninferrable.
        """

        func_name = func_node.name

        # does not handle recursion
        if func_name in self.active_calls:
            return None

        self.active_calls.add(func_name)

        # seperates local and gloabl variables
        previous_env = self.env
        self.env = self.env.create_child()

        try:
            self._bind_function_arguments(node, func_node)

            if func_node.returns:
                return self._extract_annotation_shape(func_node.returns)

            return self._infer_function_body_shape(func_node.body)
        finally:
            self.env = previous_env
            self.active_calls.remove(func_name)

    def _infer_function_body_shape(
        self, body: list[ast.stmt]
    ) -> tuple[int | str, ...] | None:
        """Walks through function body statements and tracks the final return shape.

        Args:
            body: A list of AST statement nodes inside the function definition.

        Returns:
            The inferred return shape tuple, or None.
        """
        inferred_return_shape = None
        for stmt in body:
            self.visit(stmt)
            if isinstance(stmt, ast.Return):
                inferred_return_shape = self._infer_shape(stmt.value)
        return inferred_return_shape

    # ==========================================
    # 3. Specialized Handlers
    # ==========================================

    def _infer_reshape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers and validates shape changes resulting from a reshape operation.

        Args:
            node: The `ast.Call` node representing the reshape operation.

        Returns:
            The new shape tuple after validation, or None if validation fails.
        """
        old_shape, args = self._extract_call_target(node)

        if not args or old_shape is None:
            receiver_node = (
                node.func.value if isinstance(node.func, ast.Attribute) else None
            )
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                (
                    f"could not infer original shape: "
                    f"{ast.unparse(receiver_node) if receiver_node else 'unknown'} "
                    f"shape not found and missing argument "
                ),
            )
            return None

        new_shape_arg = args[0]
        new_shape = self._extract_expr_shape(new_shape_arg)

        if new_shape is None:
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                f"could not infer new shape: {ast.unparse(new_shape_arg)} ",
            )
            return None

        if any(isinstance(d, str) or d < 0 for d in old_shape):
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                "All dimensions in the original shape must be positive. ",
            )
            return None
        old_total = math.prod([d for d in old_shape if isinstance(d, int)])

        # accepts -1 (NumPy wildcard) but rejects other negative values
        if any(isinstance(d, str) or d < -1 for d in new_shape):
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                "New shape dimensions cannot be -2 or smaller. ",
            )
            return None

        if new_shape.count(-1) > 1:
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                "New shape cannot have multiple -1 dimensions. ",
            )
            return None
        new_total = math.prod(d for d in new_shape if d != -1 and isinstance(d, int))

        # If a -1 is used, check if divisible
        if (
            new_shape.count(-1) == 1
            and isinstance(new_total, int)
            and new_total > 0
            and isinstance(old_total, int)
            and old_total % new_total == 0
        ) or old_total == new_total:
            inferred_dim = (
                old_total // new_total
                if isinstance(old_total, int) and isinstance(new_total, int)
                else -1
            )
            return tuple(inferred_dim if d == -1 else d for d in new_shape)

        self._log_error(
            node,
            ErrorCode.RESHAPE,
            f"Cannot reshape array of size {old_total} into shape {new_shape}. ",
        )
        return None

    def _infer_flatten(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers the resulting shape for flatten or ravel operations.

        Args:
            node: The `ast.Call` node representing the flatten/ravel call.

        Returns:
            A 1-dimensional tuple containing the product of the original shape, or None.
        """
        shape, _ = self._extract_call_target(node)
        if shape is not None:
            return (math.prod(shape),)
        return None

    def _infer_squeeze(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape changes from squeeze operations, optionally along specific axes.

        Args:
            node: The `ast.Call` node representing the squeeze call.

        Returns:
            The squeezed shape tuple, or None if invalid.
        """
        shape, args = self._extract_call_target(node)
        if shape is None:
            return None

        axis_node = args[0] if args else None
        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        if axis_node is None:
            return tuple(dimension for dimension in shape if dimension != 1)

        axis_shape = self._extract_expr_shape(axis_node)
        if axis_shape is None:
            return None

        axes: list[int] = []
        for axis in axis_shape:
            if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
                return None
            normalized_axis = axis % len(shape)
            if normalized_axis in axes or shape[normalized_axis] != 1:
                return None
            axes.append(normalized_axis)

        return tuple(
            dimension for index, dimension in enumerate(shape) if index not in axes
        )

    def _infer_expand_dims(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape changes from expanding array dimensions.

        Args:
            node: The `ast.Call` node representing the expand_dims call.

        Returns:
            The expanded shape tuple with inserted dimensions, or None.
        """
        shape, args = self._extract_call_target(node)
        if shape is None:
            return None

        axis_node = args[0] if args else None
        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        axis_shape = (
            self._extract_expr_shape(axis_node) if axis_node is not None else None
        )
        if axis_shape is None or len(axis_shape) != 1:
            return None

        axis = axis_shape[0]
        if not isinstance(axis, int) or not -len(shape) - 1 <= axis <= len(shape):
            return None

        insert_at = axis if axis >= 0 else len(shape) + axis + 1
        return (*shape[:insert_at], 1, *shape[insert_at:])

    def _infer_swapaxes(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape changes from swapping two axes of an array.

        Args:
            node: The `ast.Call` node representing the swapaxes call.

        Returns:
            The shape tuple with swapped axes, or None if axes are invalid.
        """
        shape, args = self._extract_call_target(node)
        if shape is None or len(args) < 2:
            return None

        first_axis_shape = self._extract_expr_shape(args[0])
        second_axis_shape = self._extract_expr_shape(args[1])
        if (
            first_axis_shape is None
            or second_axis_shape is None
            or len(first_axis_shape) != 1
            or len(second_axis_shape) != 1
        ):
            return None

        first_axis = first_axis_shape[0]
        second_axis = second_axis_shape[0]
        dimension_count = len(shape)
        if not (
            isinstance(first_axis, int)
            and isinstance(second_axis, int)
            and -dimension_count <= first_axis < dimension_count
            and -dimension_count <= second_axis < dimension_count
        ):
            return None

        first_axis %= dimension_count
        second_axis %= dimension_count
        swapped_shape = list(shape)
        swapped_shape[first_axis], swapped_shape[second_axis] = (
            swapped_shape[second_axis],
            swapped_shape[first_axis],
        )
        return tuple(swapped_shape)

    def _infer_resize(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape for resize operations using call target extraction.

        Args:
            node: The `ast.Call` node representing the resize invocation.

        Returns:
            The target resized shape tuple, or None.
        """
        shape, args = self._extract_call_target(node)
        if shape is None or not args:
            return None
        return self._extract_expr_shape(args[0])

    def _infer_transpose(self, node: ast.Attribute) -> tuple[int | str, ...] | None:
        """Infers shape for attribute-based transposition (`.T`).

        Args:
            node: The `ast.Attribute` node referencing the transpose attribute.

        Returns:
            The reversed shape tuple, or None if uninferrable.
        """
        shape = self._infer_shape(node.value)
        if shape is not None:
            return tuple(reversed(shape))
        return None

    def _infer_matmult_shape(
        self,
        node: ast.BinOp,
        left_shape: tuple[int | str, ...],
        right_shape: tuple[int | str, ...],
    ) -> tuple[int | str, ...] | None:
        """Infers and validates the shape resulting from matrix multiplication (`@`).

        Args:
            node: The `ast.BinOp` node for the multiplication.
            left_shape: Shape tuple of the left operand.
            right_shape: Shape tuple of the right operand.

        Returns:
            The resulting matrix multiplication shape, or None if dimensions mismatch.
        """

        # promote 1D shapes to 2D for uniform handling
        left_is_1d = len(left_shape) == 1
        right_is_1d = len(right_shape) == 1

        work_left = (1, *left_shape) if left_is_1d else left_shape
        work_right = (*right_shape, 1) if right_is_1d else right_shape

        batch_left, (m, n1) = work_left[:-2], work_left[-2:]
        batch_right, (n2, k) = work_right[:-2], work_right[-2:]

        if n1 != n2:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ErrorCode.MATMUL,
                (
                    f"cannot multiply {left_name} {left_shape} and {right_name}. "
                    f"{right_shape}: inner dimensions must match ({n1} != {n2}). "
                ),
            )
            return None

        broadcast_batch = self._broadcast_shapes(batch_left, batch_right)
        if broadcast_batch is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ErrorCode.MATMUL,
                (
                    f"cannot multiply {left_name} {left_shape} and {right_name}. "
                    f"{right_shape}: batch dimensions {batch_left} and {batch_right} "
                    f"are incompatible. "
                ),
            )
            return None

        result = (*broadcast_batch, m, k)

        if left_is_1d:
            result = result[1:]
        if right_is_1d:
            result = result[:-1]

        return result

    def _infer_elementwise_shape(
        self,
        node: ast.BinOp,
        left_shape: tuple[int | str, ...],
        right_shape: tuple[int | str, ...],
    ) -> tuple[int | str, ...] | None:
        """Infers and validates the shape resulting from matrix multiplication (`@`).

        Args:
            node: The `ast.BinOp` node for the multiplication.
            left_shape: Shape tuple of the left operand.
            right_shape: Shape tuple of the right operand.

        Returns:
            The resulting matrix multiplication shape, or None if dimensions mismatch.
        """
        broadcasted = self._broadcast_shapes(left_shape, right_shape)
        if broadcasted is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ErrorCode.ELEMENTWISE,
                (
                    f"cannot combine {left_name} {left_shape} and {right_name}. "
                    f"{right_shape} with element-wise operator. "
                ),
            )
            return None
        return broadcasted

    def _infer_random_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape for random generation calls.

        Args:
            node: The `ast.Call` node representing the random function call.

        Returns:
            The inferred shape tuple, or None.
        """
        if len(node.args) == 1:
            extracted = self._extract_expr_shape(node.args[0])
            if extracted is not None:
                return extracted

        shape: list[int | str] = []
        for arg in node.args:
            extracted = self._extract_expr_shape(arg)
            if extracted and len(extracted) == 1:
                shape.append(extracted[0])
            else:
                return None
        return tuple(shape) if shape else None

    def _infer_randint_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape for integer random generation functions via keyword or position.

        Args:
            node: The `ast.Call` node representing the randint call.

        Returns:
            The inferred shape tuple, or None.
        """
        size_node = next(
            (kw.value for kw in node.keywords if kw.arg == "size"),
            None,
        )
        if size_node is None:
            if len(node.args) >= 3:
                size_node = node.args[2]
            elif (
                len(node.args) == 1 and not isinstance(node.func, ast.Attribute)
            ) or len(node.args) == 2:
                return None

        if size_node is not None:
            return self._extract_expr_shape(size_node)
        return None

    def _infer_random_distribution_shape(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | None:
        """Infers shape for continuous distribution functions using the size keyword.

        Args:
            node: The `ast.Call` node representing the distribution call.

        Returns:
            The inferred shape tuple, or None.
        """
        size_node = next(
            (kw.value for kw in node.keywords if kw.arg == "size"),
            None,
        )
        if size_node is not None:
            return self._extract_expr_shape(size_node)

        return None

    def _infer_reduction(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape changes resulting from reduction operations (sum, mean, max).

        Args:
            node: The `ast.Call` node representing the reduction function.

        Returns:
            The reduced shape tuple, or None.
        """
        shape, args = self._extract_call_target(node)
        if shape is None:
            return None

        axis_node = next(
            (kw.value for kw in node.keywords if kw.arg == "axis"),
            args[0] if args else None,
        )

        has_axis_kw = any(kw.arg == "axis" for kw in node.keywords)

        if axis_node is None and not has_axis_kw:
            return ()

        axis_shape = (
            self._extract_expr_shape(axis_node) if axis_node is not None else None
        )
        if axis_shape is None:
            return None

        axes: list[int] = []
        for axis in axis_shape:
            if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
                return shape
            normalized_axis = axis + len(shape) if axis < 0 else axis
            if normalized_axis in axes:
                return shape
            axes.append(normalized_axis)

        keepdims = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "keepdims"),
            None,
        )
        is_keepdims = isinstance(keepdims, ast.Constant) and keepdims.value is True

        if is_keepdims:
            return tuple(
                1 if index in axes else dimension
                for index, dimension in enumerate(shape)
            )

        return tuple(
            dimension for index, dimension in enumerate(shape) if index not in axes
        )

    def _infer_subscript(self, node: ast.Subscript) -> tuple[int | str, ...] | None:
        """Infers shape resulting from array slicing and indexing operations (`a[1:3]`).

        Args:
            node: The `ast.Subscript` node containing the slice/index expression.

        Returns:
            The sliced shape tuple, or None.
        """
        shape = self._infer_shape(node.value)
        if shape is None:
            return None

        slice_node = node.slice
        slices = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]

        result_shape: list[int | str] = []
        shape_idx = 0

        for s in slices:
            if isinstance(s, ast.Constant) and s.value is Ellipsis:
                ellipsis_count = len(shape) - len(slices) + 1
                for _ in range(max(0, ellipsis_count)):
                    if shape_idx < len(shape):
                        result_shape.append(shape[shape_idx])
                        shape_idx += 1
                continue

            if shape_idx >= len(shape):
                break

            current_dim = shape[shape_idx]

            if isinstance(s, ast.Slice):
                dim_len = self._infer_slice_dimension_length(current_dim, s)
                result_shape.append(dim_len)
                shape_idx += 1
            elif isinstance(s, ast.Constant) and isinstance(s.value, int):
                shape_idx += 1
            else:
                shape_idx += 1

        while shape_idx < len(shape):
            result_shape.append(shape[shape_idx])
            shape_idx += 1

        return tuple(result_shape)

    def _infer_arange(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape for range-based array creation functions like `np.arange`.

        Args:
            node: The `ast.Call` node representing the arange invocation.

        Returns:
            A 1-dimensional shape tuple reflecting the calculated length, or None.
        """
        if len(node.args) < 2 and not any(
            kw.arg in ("start", "stop") for kw in node.keywords
        ):
            return None

        # If positional args are given as scalars: arange(start, stop, step)
        if len(node.args) >= 3:
            start_val = self._extract_dim_value(node.args[0])
            stop_val = self._extract_dim_value(node.args[1])
            step_val = self._extract_dim_value(node.args[2])
            if (
                isinstance(start_val, (int, float))
                and isinstance(stop_val, (int, float))
                and isinstance(step_val, (int, float))
                and step_val != 0
            ):
                length = math.ceil((stop_val - start_val) / step_val)
                return (max(0, int(length)),)
        elif len(node.args) == 2:
            start_val = self._extract_dim_value(node.args[0])
            stop_val = self._extract_dim_value(node.args[1])
            if isinstance(start_val, (int, float)) and isinstance(
                stop_val, (int, float)
            ):
                length = math.ceil(stop_val - start_val)
                return (max(0, int(length)),)
        elif len(node.args) == 1:
            stop_val = self._extract_dim_value(node.args[0])
            if isinstance(stop_val, (int, float)):
                return (max(0, int(stop_val)),)

        return None

    def _infer_linspace(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape for logarithmically or linearly spaced sequence functions.

        Args:
            node: The `ast.Call` node representing the invocation.

        Returns:
            A 1-dimensional tuple containing the size determined by `num`, or None.
        """
        # Check for keyword 'num'
        for kw in node.keywords:
            if kw.arg == "num":
                val = self._extract_expr_shape(kw.value)
                if val and isinstance(val[0], int):
                    return val

        # Fallback : linspace(start, stop, num=50) -> num is usually 3rd arg
        if len(node.args) >= 3:
            val = self._extract_expr_shape(node.args[2])
            if val and isinstance(val[0], int):
                return val

        # Default num for linspace/logspace/geomspace is 50 if omitted
        return (50,)

    def _infer_meshgrid(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape coordinates resulting from coordinate matrix creation.

        Args:
            node: The `ast.Call` node representing the meshgrid call.

        Returns:
            The generated shape tuple, factoring in the chosen indexing style, or None.
        """
        input_shapes = []
        for arg in node.args:
            shape = self._infer_shape(arg)
            if shape is not None and len(shape) == 1:
                input_shapes.append(shape[0])
            else:
                return None

        if input_shapes is None:
            return None

        # Check indexing style (default is 'xy' which swaps the first two dimensions)
        indexing = "xy"
        for kw in node.keywords:
            if (
                kw.arg == "indexing"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                indexing = kw.value.value

        if indexing == "xy" and len(input_shapes) >= 2:
            input_shapes[0], input_shapes[1] = input_shapes[1], input_shapes[0]

        return tuple(input_shapes)

    # ==========================================
    # 4. Helpers
    # ==========================================

    def _extract_call_target(
        self, node: ast.Call
    ) -> tuple[tuple[int | str, ...] | None, list[ast.AST]]:
        """Normalizes a call node, returning target shape and remaining arguments.

        Handles both method calls (`arr.op(arg)`) and func calls (`np.op(arr, arg)`).

        Args:
            node: The `ast.Call` node to normalize.

        Returns:
            A tuple of (target_shape, remaining_args).
        """
        # Check if method call
        if isinstance(node.func, ast.Attribute):
            receiver_shape = self._infer_shape(node.func.value)
            if receiver_shape is not None:
                return receiver_shape, list(node.args)

        # Else function call
        if node.args:
            target_shape = self._infer_shape(node.args[0])
            return target_shape, list(node.args[1:])

        return None, []

    def _extract_dim_value(self, node: ast.AST | None) -> int | str | None:
        """Extracts a scalar integer/float value from an AST node for dim calculations.

        Args:
            node: The AST node containing a scalar value or variable name.

        Returns:
            The extracted integer/string value, or None if unextractable.
        """
        value = None

        if node is None:
            return None

        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        # Negative integer via unary minus
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            value = node.operand.value
            if value.is_integer():
                return -int(value)

        if isinstance(node, ast.Name):
            value = self.env.get_scalar(node.id)

            if value is None or not float(value).is_integer():
                self._log_error(
                    node,
                    ErrorCode.VALUE,
                    f"Scalar variable '{node.id}' must be an integer, got {value}.",
                )
                return None
            return int(value)

        return None

    def _extract_annotation_shape(self, node: ast.AST) -> tuple[int | str, ...] | None:
        """Extracts explicit shape tuples from type annotations using `Annotated`.

        Args:
            node: The AST node representing the type annotation.

        Returns:
            The extracted shape tuple, or None.
        """
        if not isinstance(node, ast.Subscript):
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id == "Annotated"):
            return None

        slice_node = node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
            return None

        elt = slice_node.elts[1]
        if not isinstance(elt, (ast.Tuple, ast.List)):
            return None

        shape: list[int | str] = []
        for item in elt.elts:
            val = self._extract_dim_value(item)
            if val is None:
                return None
            shape.append(val)

        return tuple(shape)

    def _extract_expr_shape(self, node: ast.AST) -> tuple[int | str, ...] | None:
        """Extracts shape from a literal list/tuple of dim or a scalar expression.

        Args:
            node: The AST expression node representing dimensions.

        Returns:
            The shape tuple, or None.
        """
        top_val = self._extract_dim_value(node)
        if top_val is not None:
            return (top_val,)

        if not isinstance(node, (ast.List, ast.Tuple)):
            return None

        if not node.elts:
            return ()

        shape: list[int | str] = []
        for elt in node.elts:
            val = self._extract_dim_value(elt)
            if val is not None:
                shape.append(val)
            else:
                return None

        return tuple(shape) if shape is not None else None

    def _extract_literal_shape(self, node: ast.AST) -> tuple[int, ...]:
        """Calculates the recursive shape of nested literal lists or tuples.
        Example: np.array([[1, 2], [3, 4]]) has a shape of (2, 2).

        Args:
            node: The AST node representing literal collections.

        Returns:
            A tuple representing the multi-dimensional structure.
        """
        if not isinstance(node, (ast.List, ast.Tuple)):
            return ()
        if not node.elts:
            return (0,)

        current_dim = len(node.elts)
        sub_shape = self._extract_literal_shape(node.elts[0])
        return (current_dim, *sub_shape)

    def _broadcast_shapes(
        self, shape1: tuple[int | str, ...], shape2: tuple[int | str, ...]
    ) -> tuple[int | str, ...] | None:
        """Broadcasts two shape tuples following standard NumPy right-to-left rules.

        Args:
            shape1: The first shape tuple.
            shape2: The second shape tuple.

        Returns:
            The broadcasted result shape tuple, or None if shapes are incompatible.
        """
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        p1 = (1,) * (max_len - len1) + shape1
        p2 = (1,) * (max_len - len2) + shape2

        result = []
        for d1, d2 in zip(p1, p2, strict=False):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                return None
        return tuple(result)

    def _bind_function_arguments(
        self, node: ast.Call, func_node: ast.FunctionDef
    ) -> None:
        """Binds positional, default, and annotated arg shapes to the local func scope.

        Args:
            node: The `ast.Call` node supplying the argument values.
            func_node: The `ast.FunctionDef` node being invoked.
        """
        args_args = func_node.args.args
        defaults = func_node.args.defaults
        num_required = len(args_args) - len(defaults)
        provided_shapes = [self._infer_shape(arg) for arg in node.args]

        for i, param in enumerate(args_args):
            arg_shape = provided_shapes[i] if i < len(provided_shapes) else None

            if arg_shape is None and i >= num_required:
                arg_shape = self._infer_shape(defaults[i - num_required])

            if arg_shape is not None:
                self.env.set_shape(param.arg, arg_shape)

            if param.annotation and arg_shape:
                expected_shape = self._extract_annotation_shape(param.annotation)
                if expected_shape is not None and expected_shape != arg_shape:
                    self._log_error(
                        node,
                        ErrorCode.ANNOTATION,
                        (
                            f"Argument annotated as {expected_shape}, "
                            f"but expression has the shape {arg_shape}. "
                        ),
                    )

    def _infer_slice_dimension_length(
        self, current_dim: int | str, s: ast.Slice
    ) -> int | str:
        """Calculates the resulting length of a single dim under a slice operation.

        Args:
            current_dim: The current dimension size.
            s: The `ast.Slice` node defining bounds and steps.

        Returns:
            The new dimension size after slicing.
        """
        if not isinstance(current_dim, int):
            return current_dim

        step_val = self._extract_dim_value(s.step) if s.step is not None else 1
        if not isinstance(step_val, int) or step_val == 0:
            return current_dim

        if step_val > 0:
            start_val = self._extract_dim_value(s.lower) if s.lower is not None else 0
            stop_val = (
                self._extract_dim_value(s.upper) if s.upper is not None else current_dim
            )

            if not isinstance(start_val, int) or not isinstance(stop_val, int):
                return current_dim

            start = max(
                0,
                current_dim + start_val
                if start_val < 0
                else min(start_val, current_dim),
            )
            stop = max(
                0,
                current_dim + stop_val if stop_val < 0 else min(stop_val, current_dim),
            )
            effective_len = math.ceil((stop - start) / step_val)
        else:
            start_val = (
                self._extract_dim_value(s.lower)
                if s.lower is not None
                else current_dim - 1
            )
            stop_val = self._extract_dim_value(s.upper) if s.upper is not None else -1

            if not isinstance(start_val, int) or not isinstance(stop_val, int):
                return current_dim

            start = (
                max(-1, current_dim + start_val)
                if start_val < 0
                else min(start_val, current_dim - 1)
            )
            stop = (
                max(-1, current_dim + stop_val)
                if stop_val < -1
                else min(stop_val, current_dim)
            )
            effective_len = math.ceil((start - stop) / abs(step_val))

        return max(0, effective_len)

    def _log_error(self, node: ast.AST, err_type: ErrorCode, message: str) -> None:
        """Logs structured diagnostic errors matching test assertion requirements.

        Args:
            node: The AST node where the error occurred.
            err_type: The `ErrorCode` enum classifying the error type.
            message: A descriptive error message string.
        """
        line_no = getattr(node, "lineno", 0)
        self.errors.append(
            {
                "line": line_no,
                "col": getattr(node, "col_offset", 0),
                "code": err_type.value,
                "message": message,
            }
        )
