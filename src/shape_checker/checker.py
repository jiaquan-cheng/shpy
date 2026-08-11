import ast
from collections.abc import Sequence
from enum import Enum


class ShapeError(Enum):
    ANNOTATION_MISMATCH = "ANNOTATION_MISMATCH"


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
                "code": err_type,
                "message": f"[line {line_no}] [{err_type}] {message}",
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
                f"{var_name} annotated as {annotated_shape}, but expression has the shape {inferred_shape}",
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

        return None

    def _infer_call_shape(self, node: ast.Call) -> tuple[int | str, ...] | None:
        """Infers shape from function calls like np.array, np.zeros, np.ones."""
        func_name = ast.unparse(node.func)

        if func_name.endswith("array") and node.args:
            return self._traverse_literal_node(node.args[0])

        if any(func_name.endswith(c) for c in ("zeros", "ones", "empty", "full")):
            return self._extract_shape_from_args(node.args)

        return None

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
