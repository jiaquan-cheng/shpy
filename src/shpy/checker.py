import ast
import math
from enum import Enum
from typing import Any


class ErrorCode(Enum):
    ANNOTATION = "Annotation"
    ELEMENTWISE = "Elementwise"
    MATMUL = "MatMul"
    RESHAPE = "Reshape"


class Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.shapes: dict[
            str, tuple[Any, ...] | None
        ] = {}  # tracks variable shapes, None if unknown
        self.scalar_values: dict[
            str, int | float
        ] = {}  # tracks variable values of scalars, in case they are used in shape definitions
        self.errors: list[dict[str, Any]] = []

        self.call_handlers = {
            "array": lambda node: (
                self._traverse_literal_node(node.args[0]) if node.args else None
            ),
            "zeros": lambda node: (
                self._extract_shape_from_list_or_tuple_or_constant(node.args[0])
                if node.args
                else None
            ),
            "ones": lambda node: (
                self._extract_shape_from_list_or_tuple_or_constant(node.args[0])
                if node.args
                else None
            ),
            "empty": lambda node: (
                self._extract_shape_from_list_or_tuple_or_constant(node.args[0])
                if node.args
                else None
            ),
            "full": lambda node: (
                self._extract_shape_from_list_or_tuple_or_constant(node.args[0])
                if node.args
                else None
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
            self.scalar_values[var_name] = node.value.value

        annotated_shape = self._extract_annotation(node.annotation)
        inferred_shape = self._infer_shape(node.value) if node.value else None

        self.shapes[var_name] = annotated_shape if annotated_shape else inferred_shape

        if annotated_shape and inferred_shape and annotated_shape != inferred_shape:
            self._log_error(
                node,
                ErrorCode.ANNOTATION,
                f"{var_name} annotated as {annotated_shape}, but expression has the shape {inferred_shape}. ",
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
                    self.scalar_values[target.id] = node.value.value

        inferred_shape = self._infer_shape(node.value) if node.value else None

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.shapes[target.id] = inferred_shape

    # ==========================================
    # 2. Core Inference
    # ==========================================

    def _infer_shape(self, node: ast.AST | None) -> tuple[int | str, ...] | None:
        """Extracts shape from right-hand side expression."""
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return self.shapes.get(node.id)

        if isinstance(node, ast.Call):
            return self._infer_call_shape(node)

        if isinstance(node, ast.BinOp):
            return self._infer_binop_shape(node)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return (1,)

        if isinstance(node, ast.Attribute):
            return self._infer_attribute_shape(node)

        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)

        return None

    def _infer_call_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape from function calls like np.array, np.zeros, np.ones."""
        func_name = ast.unparse(node.func)
        for suffix in self.call_handlers:
            if func_name.endswith(suffix):
                handler = self.call_handlers[suffix]
                return handler(node)
        return None

    def _infer_attribute_shape(
        self, node: ast.Attribute
    ) -> tuple[int | str, ...] | None:
        """Handles attribute-based shape changes, such as .T (transpose)."""
        handler = self.attr_handlers.get(node.attr)
        if handler:
            return handler(node)
        return None

    def _infer_binop_shape(self, node: ast.BinOp) -> tuple[int | str, ...] | None:
        """Infers and validates shapes for binary operations (+, -, *, @)."""
        left_shape = self._infer_shape(node.left)
        right_shape = self._infer_shape(node.right)

        if left_shape is None or right_shape is None:
            return None

        handler = self.binop_handlers.get(type(node.op))
        if handler:
            return handler(node, left_shape, right_shape)
        return None

    # ==========================================
    # 3. Specialized Handlers
    # ==========================================

    def _infer_reshape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles reshape operations."""
        receiver_node = (
            node.func.value if isinstance(node.func, ast.Attribute) else None
        )
        receiver_shape = (
            self._infer_shape(receiver_node) if receiver_node is not None else None
        )
        has_receiver = 1 if receiver_shape is not None else 0

        # checks if receiver has shape in receiver.reshape(new_shape) else np.reshape(old_shape, new_shape)
        if not node.args or has_receiver + len(node.args) < 2:
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                f"could not infer original shape: {ast.unparse(receiver_node) if receiver_node else 'unknown'} shape not found and missing argument ",
            )
            return None

        old_shape = (
            self._infer_shape(node.args[0])
            if receiver_shape is None
            else receiver_shape
        )
        new_shape_idx = 1 if receiver_shape is None else 0

        new_shape_arg = node.args[new_shape_idx]
        new_shape = self._extract_shape_from_list_or_tuple_or_constant(new_shape_arg)

        if old_shape is None:
            return None
        if new_shape is None:
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                f"could not infer new shape: {ast.unparse(new_shape_arg)} ",
            )
            return None

        # old shape must be positive
        if any(isinstance(d, str) or d < 0 for d in old_shape):
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                "All dimensions in the original shape must be positive. ",
            )
            return None
        old_total = math.prod([d for d in old_shape if isinstance(d, int)])

        # smallest dimension in new shape must be -1 or positive
        if any(isinstance(d, str) or d < -1 for d in new_shape):
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                "New shape dimensions cannot be -2 or smaller. ",
            )
            return None

        # at most one -1 in new shape
        if new_shape.count(-1) > 1:
            self._log_error(
                node,
                ErrorCode.RESHAPE,
                "New shape cannot have multiple -1 dimensions. ",
            )
            return None
        new_total = math.prod(d for d in new_shape if d != -1 and isinstance(d, int))

        # if -1 then old_total must be divisible by new_total
        if (
            new_shape.count(-1) == 1
            and isinstance(new_total, int)
            and new_total > 0
            and isinstance(old_total, int)
            and old_total % new_total == 0
            or old_total == new_total
        ):
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
        """Handles flatten and ravel operations."""
        target_node = None
        if isinstance(node, ast.Call):
            if node.args:  # np.flatten(array)
                target_node = node.args[0]
            elif isinstance(node.func, ast.Attribute):  # receiver.flatten()
                target_node = node.func.value
        shape = self._infer_shape(target_node)
        if shape is not None:
            return (math.prod(shape),)
        return None

    def _infer_squeeze(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles squeeze operations, including an optional axis."""
        target_node = node.func.value if isinstance(node.func, ast.Attribute) else None
        shape = self._infer_shape(target_node)
        if target_node is not None and shape is not None:
            axis_node = node.args[0] if node.args else None
        else:
            target_node = node.args[0] if node.args else None
            axis_node = node.args[1] if len(node.args) > 1 else None

        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        shape = self._infer_shape(target_node)
        if shape is None:
            return None
        if axis_node is None:
            return tuple(dimension for dimension in shape if dimension != 1)

        axis_shape = self._extract_shape_from_list_or_tuple_or_constant(axis_node)
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
        """Handles expand_dims operations."""

        target_node = node.func.value if isinstance(node.func, ast.Attribute) else None
        shape = self._infer_shape(target_node)
        if target_node is not None and shape is not None:
            axis_node = node.args[0] if node.args else None
        else:
            target_node = node.args[0] if node.args else None
            axis_node = node.args[1] if len(node.args) > 1 else None

        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        shape = self._infer_shape(target_node)
        axis_shape = (
            self._extract_shape_from_list_or_tuple_or_constant(axis_node)
            if axis_node is not None
            else None
        )
        if shape is None or axis_shape is None or len(axis_shape) != 1:
            return None

        axis = axis_shape[0]
        if not isinstance(axis, int) or not -len(shape) - 1 <= axis <= len(shape):
            return None
        insert_at = axis if axis >= 0 else len(shape) + axis + 1
        return shape[:insert_at] + (1,) + shape[insert_at:]

    def _infer_swapaxes(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles swapaxes operations."""
        target_node = node.func.value if isinstance(node.func, ast.Attribute) else None
        shape = self._infer_shape(target_node)
        if target_node is not None and shape is not None:
            first_axis_node = node.args[0] if node.args else None
            second_axis_node = node.args[1] if len(node.args) > 1 else None
        else:
            target_node = node.args[0] if node.args else None
            first_axis_node = node.args[1] if len(node.args) > 1 else None
            second_axis_node = node.args[2] if len(node.args) > 2 else None

        shape = self._infer_shape(target_node)
        first_axis_shape = (
            self._extract_shape_from_list_or_tuple_or_constant(first_axis_node)
            if first_axis_node is not None
            else None
        )
        second_axis_shape = (
            self._extract_shape_from_list_or_tuple_or_constant(second_axis_node)
            if second_axis_node is not None
            else None
        )
        if (
            shape is None
            or first_axis_shape is None
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
        """Handles resize operations."""
        is_method_call = (
            isinstance(node.func, ast.Attribute)
            and self._infer_shape(node.func.value) is not None
        )
        shape_node = (
            node.args[0]
            if is_method_call and node.args
            else node.args[1]
            if not is_method_call and len(node.args) > 1
            else None
        )
        if shape_node is None:
            return None
        return self._extract_shape_from_list_or_tuple_or_constant(shape_node)

    def _infer_transpose(self, node: ast.Attribute) -> tuple[int | str, ...] | None:
        """Handles transpose operations."""
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
        """Infer and validate shape of matrix multiplication."""

        # promote 1D shapes to 2D for uniform handling
        left_is_1d = len(left_shape) == 1
        right_is_1d = len(right_shape) == 1

        work_left = (1,) + left_shape if left_is_1d else left_shape
        work_right = right_shape + (1,) if right_is_1d else right_shape

        batch_left, (m, n1) = work_left[:-2], work_left[-2:]
        batch_right, (n2, k) = work_right[:-2], work_right[-2:]

        if n1 != n2:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ErrorCode.MATMUL,
                f"cannot multiply {left_name} {left_shape} and {right_name} {right_shape}: inner dimensions must match ({n1} != {n2}). ",
            )
            return None

        broadcast_batch = self._broadcast_shapes(batch_left, batch_right)
        if broadcast_batch is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ErrorCode.MATMUL,
                f"cannot multiply {left_name} {left_shape} and {right_name} {right_shape}: batch dimensions {batch_left} and {batch_right} are incompatible. ",
            )
            return None

        result = broadcast_batch + (m, k)

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
        """Infer and validate shape of element-wise operations."""
        broadcasted = self._broadcast_shapes(left_shape, right_shape)
        if broadcasted is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ErrorCode.ELEMENTWISE,
                f"cannot combine {left_name} {left_shape} and {right_name} {right_shape} with element-wise operator. ",
            )
            return None
        return broadcasted

    def _infer_random_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles random functions that take dimensions either as separate positional arguments or a tuple/list size."""
        if len(node.args) == 1:
            extracted = self._extract_shape_from_list_or_tuple_or_constant(node.args[0])
            if extracted is not None:
                return extracted

        shape: list[int | str] = []
        for arg in node.args:
            extracted = self._extract_shape_from_list_or_tuple_or_constant(arg)
            if extracted and len(extracted) == 1:
                shape.append(extracted[0])
            else:
                shape.append(ast.unparse(arg))
        return tuple(shape) if shape else (1,)

    def _infer_randint_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles randint and similar functions where shape can be passed via a size keyword or positional argument."""
        size_node = next(
            (kw.value for kw in node.keywords if kw.arg == "size"),
            None,
        )
        if size_node is None:
            # Check if size is passed as a positional argument (usually 3rd arg for randint(low, high, size))
            if len(node.args) >= 3:
                size_node = node.args[2]
            elif (
                len(node.args) == 1
                and not isinstance(node.func, ast.Attribute)
                or len(node.args) == 2
            ):
                # If only high or low,high are given without size, it returns a scalar
                return (1,)

        if size_node is not None:
            return self._extract_shape_from_list_or_tuple_or_constant(size_node)
        return (1,)

    def _infer_random_distribution_shape(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | None:
        """Handles continuous distributions like uniform/normal where shape is passed via 'size' keyword."""
        size_node = next(
            (kw.value for kw in node.keywords if kw.arg == "size"),
            None,
        )
        if size_node is not None:
            return self._extract_shape_from_list_or_tuple_or_constant(size_node)

        # Fallback to positional arguments if size isn't specified (e.g., trailing args can represent parameters or size depending on function)
        return None

    def _infer_reduction(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles reduction operations like sum, mean, prod, min, max."""
        target_node = (
            node.func.value
            if isinstance(node.func, ast.Attribute)
            and self._infer_shape(node.func.value) is not None
            else None
        )

        if target_node is not None:
            axis_node = node.args[0] if node.args else None
        else:
            target_node = node.args[0] if node.args else None
            axis_node = node.args[1] if len(node.args) > 1 else None

        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        shape = self._infer_shape(target_node)
        if shape is None:
            return None

        if axis_node is None:
            return (1,)

        axis_shape = self._extract_shape_from_list_or_tuple_or_constant(axis_node)
        if axis_shape is None:
            return None

        axes: list[int] = []
        for axis in axis_shape:
            if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
                return None
            normalized_axis = axis + len(shape) if axis < 0 else axis
            if normalized_axis in axes:
                return None
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
        """Handles array slicing and indexing operations (e.g., a[0], a[1:3, :])."""
        shape = self._infer_shape(node.value)
        if shape is None:
            return None

        slice_node = node.slice
        # Normalize slice into a tuple of slices/indices
        if isinstance(slice_node, ast.Tuple):
            slices = slice_node.elts
        else:
            slices = [slice_node]

        result_shape: list[int | str] = []
        shape_idx = 0

        for s in slices:
            if isinstance(s, ast.Constant) and s.value is Ellipsis:
                # Ellipsis matches as many dimensions as needed
                # For simplicity in fixed rank shapes, count remaining dimensions
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
                # Slice operation: start:stop:step
                start = self._eval_index_constant(s.lower) if s.lower is not None else 0
                stop = (
                    self._eval_index_constant(s.upper)
                    if s.upper is not None
                    else (current_dim if isinstance(current_dim, int) else None)
                )
                step = self._eval_index_constant(s.step) if s.step is not None else 1

                if (
                    isinstance(current_dim, int)
                    and isinstance(start, int)
                    and isinstance(stop, int)
                    and isinstance(step, int)
                ):
                    # Handle bounds and steps roughly
                    start = max(0, min(start, current_dim))
                    stop = max(0, min(stop, current_dim))
                    effective_len = math.ceil((stop - start) / step) if step != 0 else 0
                    result_shape.append(max(0, effective_len))
                else:
                    result_shape.append(current_dim)
                shape_idx += 1
            elif isinstance(s, ast.Constant) and isinstance(s.value, int):
                # Integer indexing drops the dimension
                shape_idx += 1
            else:
                # Other indexing expressions (like lists or advanced indexing)
                shape_idx += 1

        # Append any remaining dimensions if not fully consumed by slices/ellipsis
        while shape_idx < len(shape):
            result_shape.append(shape[shape_idx])
            shape_idx += 1

        return tuple(result_shape)

    def _eval_index_constant(self, node: ast.AST | None) -> int | None:
        """Helper to evaluate simple constant integer indices or negative sign expressions."""
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        ):
            return -node.operand.value
        return None

    def _infer_arange(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles np.arange(start, stop, step) or similar variations."""
        if len(node.args) < 2 and not any(
            kw.arg in ("start", "stop") for kw in node.keywords
        ):
            return None

        # If positional args are given as scalars: arange(start, stop, step)
        if len(node.args) >= 3:
            start_val = self._eval_index_constant(node.args[0])
            stop_val = self._eval_index_constant(node.args[1])
            step_val = self._eval_index_constant(node.args[2])
            if (
                isinstance(start_val, (int, float))
                and isinstance(stop_val, (int, float))
                and isinstance(step_val, (int, float))
                and step_val != 0
            ):
                length = math.ceil((stop_val - start_val) / step_val)
                return (max(0, int(length)),)
        elif len(node.args) == 2:
            start_val = self._eval_index_constant(node.args[0])
            stop_val = self._eval_index_constant(node.args[1])
            if isinstance(start_val, (int, float)) and isinstance(
                stop_val, (int, float)
            ):
                length = math.ceil(stop_val - start_val)
                return (max(0, int(length)),)
        elif len(node.args) == 1:
            stop_val = self._eval_index_constant(node.args[0])
            if isinstance(stop_val, (int, float)):
                return (max(0, int(stop_val)),)

        return None

    def _infer_linspace(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles np.linspace, np.logspace, np.geomspace where 'num' dictates size."""
        # Check for keyword 'num'
        for kw in node.keywords:
            if kw.arg == "num":
                val = self._extract_shape_from_list_or_tuple_or_constant(kw.value)
                if val and isinstance(val[0], int):
                    return val

        # Fallback positionally: linspace(start, stop, num=50) -> num is usually 3rd arg
        if len(node.args) >= 3:
            val = self._extract_shape_from_list_or_tuple_or_constant(node.args[2])
            if val and isinstance(val[0], int):
                return val

        # Default num for linspace/logspace/geomspace is 50 if omitted
        return (50,)

    def _infer_meshgrid(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Handles np.meshgrid(*xi, indexing='xy'/'ij')."""
        input_shapes = []
        for arg in node.args:
            shape = self._infer_shape(arg)
            if shape and len(shape) == 1:
                input_shapes.append(shape[0])
            else:
                return None

        if not input_shapes:
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

        # Meshgrid returns a list/tuple of arrays, but in typical static annotation checks,
        # assigning it or returning it matches a tuple of those dimensions or the first broadcasted shape.
        # Returning the multi-dimensional shape of each output array:
        return tuple(input_shapes)

    # ==========================================
    # 4. Helpers
    # ==========================================

    def _extract_annotation(self, node: ast.AST) -> tuple[int | str, ...] | None:
        """Extracts shape from explicit annotations using native literal tuples/lists."""
        if not isinstance(node, ast.Subscript):
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id == "Annotated"):
            return None

        slice_node = node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
            return None

        elt = slice_node.elts[1]

        if isinstance(elt, (ast.Tuple, ast.List)):
            shape: list[int | str] = []
            for item in elt.elts:
                if isinstance(item, ast.Constant) and isinstance(
                    item.value, (int, str)
                ):
                    shape.append(item.value)
                elif isinstance(item, ast.Name):
                    shape.append(item.id)
                else:
                    shape.append(ast.unparse(item))
            return tuple(shape)

        return None

    def _extract_shape_from_list_or_tuple_or_constant(
        self, node: ast.AST
    ) -> tuple[int | str, ...] | None:
        """Extracts shape from a list or tuple of constants or an expression."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return (node.value,)

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return (-int(node.operand.value),)

        if isinstance(node, (ast.List, ast.Tuple)):
            if not node.elts:
                return (0,)

            shape: list[int | str] = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    shape.append(elt.value)
                elif isinstance(elt, ast.Name) and elt.id in self.scalar_values:
                    val = self.scalar_values[elt.id]
                    if isinstance(val, (int, float)):
                        shape.append(int(val))
                elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                    if isinstance(elt.operand, ast.Constant) and isinstance(
                        elt.operand.value, (int, float)
                    ):
                        shape.append(-int(elt.operand.value))
                elif isinstance(elt, ast.Name):
                    shape.append(elt.id)
                else:
                    shape.append(ast.unparse(elt))
            if len(shape) == 0:
                return None
            return tuple(shape)

        return None

    def _traverse_literal_node(self, node: ast.AST) -> tuple[int, ...]:
        """Calculates shape of lists/tuples recursively."""
        if not isinstance(node, (ast.List, ast.Tuple)):
            return ()
        if not node.elts:
            return (0,)

        current_dim = len(node.elts)
        sub_shape = self._traverse_literal_node(node.elts[0])
        return (current_dim,) + sub_shape

    def _broadcast_shapes(
        self, shape1: tuple[int | str, ...], shape2: tuple[int | str, ...]
    ) -> tuple[int | str, ...] | None:
        """Broadcasts two shapes following standard NumPy right-to-left rules."""
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        p1 = (1,) * (max_len - len1) + shape1
        p2 = (1,) * (max_len - len2) + shape2

        result = []
        for d1, d2 in zip(p1, p2):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                return None
        return tuple(result)

    def _log_error(self, node: ast.AST, err_type: ErrorCode, message: str) -> None:
        """Logs structured errors matching test assertion requirements."""
        line_no = getattr(node, "lineno", 0)
        self.errors.append(
            {
                "line": line_no,
                "col": getattr(node, "col_offset", 0),
                "code": err_type.value,
                "message": message,
            }
        )
