import ast
from enum import Enum
from typing import Any


class ShapeError(Enum):
    ANNOTATION_MISMATCH = "AnnotationMismatch"
    ELEMENTWISE_MISMATCH = "ElementwiseMismatch"
    MATMUL_MISMATCH = "MatMulMismatch"


class ShapeChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbol_table: dict[
            str, tuple[Any, ...] | None
        ] = {}  # tracks variable shapes, None if unknown
        self.value_table: dict[
            str, int | float
        ] = {}  # tracks variable values of scalars, in case they are used in shape definitions
        self.errors: list[dict[str, Any]] = []

        self.call_handlers = {
            "array": lambda node: (
                self._traverse_literal_node(node.args[0]) if node.args else None
            ),
            "zeros": lambda node: (
                self._extract_shape_from_list_or_tuple(node.args[0])
                if node.args
                else None
            ),
            "ones": lambda node: (
                self._extract_shape_from_list_or_tuple(node.args[0])
                if node.args
                else None
            ),
            "empty": lambda node: (
                self._extract_shape_from_list_or_tuple(node.args[0])
                if node.args
                else None
            ),
            "full": lambda node: (
                self._extract_shape_from_list_or_tuple(node.args[0])
                if node.args
                else None
            ),
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
        self.generic_visit(node)

        if not isinstance(node.target, ast.Name):
            return

        var_name = node.target.id

        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, (int, float)
        ):
            self.value_table[var_name] = node.value.value

        annotated_shape = self._extract_annotation(node.annotation)
        inferred_shape = self._infer_shape(node.value) if node.value else None

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

        if (
            node.value
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.value_table[target.id] = node.value.value

        inferred_shape = self._infer_shape(node.value) if node.value else None

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbol_table[target.id] = inferred_shape

    # ==========================================
    # 2. Core Inference
    # ==========================================

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

        if isinstance(node, ast.Attribute):
            return self._infer_attribute_shape(node)

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
                ShapeError.MATMUL_MISMATCH,
                f"cannot multiply {left_name} {left_shape} and {right_name} {right_shape}: inner dimensions must match ({n1} != {n2}). ",
            )
            return None

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
                ShapeError.ELEMENTWISE_MISMATCH,
                f"cannot combine {left_name} {left_shape} and {right_name} {right_shape} with element-wise operator. ",
            )
            return None
        return broadcasted

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

    def _extract_shape_from_list_or_tuple(
        self, node: ast.List | ast.Tuple
    ) -> tuple[int | str, ...] | None:
        """Extracts shape from a list or tuple of constants."""
        if not node.elts:
            return (0,)

        shape: list[int | str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                shape.append(elt.value)
            elif isinstance(elt, ast.Name) and elt.id in self.value_table:
                shape.append(int(self.value_table[elt.id]))
            else:
                shape.append(ast.unparse(elt))
        if len(shape) == 0:
            return None
        return tuple(shape)

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
