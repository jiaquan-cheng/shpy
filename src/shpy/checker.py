import ast
from collections.abc import Sequence
from enum import Enum


class ShapeError(Enum):
    ANNOTATION_MISMATCH = "AnnotationMismatch"
    ELEMENTWISE_MISMATCH = "ElementwiseMismatch"
    MATMUL_MISMATCH = "MatMulMismatch"


class ShapeChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbol_table: dict[str, tuple[int | str, ...] | None] = {}
        self.errors: list[dict[str, int | ShapeError | str]] = []

    def _log_error(self, node: ast.AST, err_type: ShapeError, message: str) -> None:
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

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handles explicit annotated variable assignments."""
        self.generic_visit(node)

        if not isinstance(node.target, ast.Name):
            return

        var_name = node.target.id
        annotated_shape = self._extract_annotation(node.annotation)
        inferred_shape = self._infer_shape(node.value) if node.value else None

        # Explicit annotation takes precedence for the symbol table contract
        self.symbol_table[var_name] = (
            annotated_shape if annotated_shape else inferred_shape
        )

        if annotated_shape and inferred_shape and annotated_shape != inferred_shape:
            self._log_error(
                node,
                ShapeError.ANNOTATION_MISMATCH,
                f"{var_name} annotated as {annotated_shape}, but expression has the shape {inferred_shape}. ",
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles implicit variable assignments."""
        self.generic_visit(node)
        inferred_shape = self._infer_shape(node.value) if node.value else None

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbol_table[target.id] = inferred_shape

    def _extract_annotation(self, node: ast.AST) -> tuple[int | str, ...] | None:
        """Extracts shape from explicit annotations using native literal tuples/lists
        (e.g. Annotated[np.ndarray, (1, 2)] or Annotated[np.ndarray, ('batch', 2)])."""
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

    def _infer_shape(self, node: ast.AST) -> tuple[int | str, ...] | None:
        """Extracts shape from right-hand side expression."""
        if isinstance(node, ast.Name):
            return self.symbol_table.get(node.id)

        if isinstance(node, ast.Call):
            return self._infer_call_shape(node)

        if isinstance(node, ast.BinOp):
            return self._infer_binop_shape(node)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return (1,)

        return None

    def _infer_call_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape from function calls like np.array, np.zeros, np.ones."""
        func_name = ast.unparse(node.func)

        if func_name.endswith("array") and node.args:
            return self._traverse_literal_node(node.args[0])

        if any(func_name.endswith(c) for c in ("zeros", "ones", "empty", "full")):
            return self._extract_shape_from_args(node.args)

        return None

    def _infer_binop_shape(self, node: ast.BinOp) -> tuple[int | str, ...] | None:
        """Infers and validates shapes for binary operations (+, -, *, @)."""
        left_shape = self._infer_shape(node.left)
        right_shape = self._infer_shape(node.right)

        if left_shape is None or right_shape is None:
            return None

        if isinstance(node.op, ast.MatMult):
            return self._infer_matmult_shape(node, left_shape, right_shape)

        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return self._infer_elementwise_shape(node, left_shape, right_shape)

        return None

    def _infer_matmult_shape(
        self,
        node: ast.BinOp,
        left_shape: tuple[int | str, ...],
        right_shape: tuple[int | str, ...],
    ) -> tuple[int | str, ...] | None:
        """Handles matrix multiplication shape inference."""
        # Promote 1D vector to 2D for alignment
        left_is_1d = len(left_shape) == 1
        right_is_1d = len(right_shape) == 1

        work_left = (1,) + left_shape if left_is_1d else left_shape
        work_right = right_shape + (1,) if right_is_1d else right_shape

        batch_left, (m, n1) = work_left[:-2], work_left[-2:]
        batch_right, (n2, k) = work_right[:-2], work_right[-2:]

        # Validate inner dimensions
        if n1 != n2:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ShapeError.MATMUL_MISMATCH,
                f"cannot multiply {left_name} {left_shape} and {right_name} {right_shape}: inner dimensions must match ({n1} != {n2}). ",
            )
            return None

        # Validate and broadcast batch dimensions
        broadcast_batch = self._broadcast_shapes(batch_left, batch_right)
        if broadcast_batch is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ShapeError.MATMUL_MISMATCH,
                f"cannot multiply {left_name} {left_shape} and {right_name} {right_shape}: batch dimensions {batch_left} and {batch_right} are incompatible. ",
            )
            return None

        result = broadcast_batch + (m, k)

        # Restore dimensions for 1D vectors
        if left_is_1d:
            result = result[1:]
        if right_is_1d:
            result = result[:-1]

        return result

    def _broadcast_shapes(
        self, shape1: tuple[int | str, ...], shape2: tuple[int | str, ...]
    ) -> tuple[int | str, ...] | None:
        """Broadcasts two shapes following standard NumPy right-to-left rules. None if incompatible."""
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        # Pad shorter shape with 1s
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

    def _infer_elementwise_shape(
        self,
        node: ast.BinOp,
        left_shape: tuple[int | str, ...],
        right_shape: tuple[int | str, ...],
    ) -> tuple[int | str, ...] | None:
        """Handles element-wise operation shape inference via broadcasting."""
        broadcasted = self._broadcast_shapes(left_shape, right_shape)
        if broadcasted is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self._log_error(
                node,
                ShapeError.ELEMENTWISE_MISMATCH,
                f"cannot combine {left_name} {left_shape} and {right_name} {right_shape} with element-wise operator. ",
            )
            return None
        return broadcasted

    def _extract_shape_from_args(
        self, args: Sequence[ast.AST]
    ) -> tuple[int | str, ...] | None:
        """Extracts shape from np.zeros, np.ones, etc. based on arguments."""
        if not args:
            return None

        first_arg = args[0]
        if isinstance(first_arg, (ast.Tuple, ast.List)):
            return tuple(
                elt.value
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int)
                else ast.unparse(elt)
                for elt in first_arg.elts
            )

        return tuple(
            arg.value
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int)
            else ast.unparse(arg)
            for arg in args
        )

    def _traverse_literal_node(self, node: ast.AST) -> tuple[int, ...]:
        """Calculates shape of lists/tuples."""
        if not isinstance(node, (ast.List, ast.Tuple)):
            return ()
        if not node.elts:
            return (0,)

        current_dim = len(node.elts)
        sub_shape = self._traverse_literal_node(node.elts[0])
        return (current_dim,) + sub_shape
