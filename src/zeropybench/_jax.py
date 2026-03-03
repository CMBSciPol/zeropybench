import ast
import sys
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['CodeASTParser']


class CodeASTParser:
    """Parse and transform code for JAX benchmarking."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree

    @classmethod
    def from_code(cls, code: str) -> Self:
        """Instantiate an AST parser from a JAX code."""
        tree = ast.parse(code, mode='exec')
        return cls(tree)

    def transform_jax_code(
        self, locals: dict[str, Any], globals: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Transform code into a benchmarkable jitted function call.

        A simple expression such as ``x + y`` is transformed into::

            __rv = __bench_func(x, y)
            jax.block_until_ready(__rv)

        with ``__bench_func`` defined as::

            @jax.jit
            def __bench_func(x, y):
                return x + y

        For a function call ``func(x, y)``, ``__bench_func`` is set to ``jax.jit(func)``
        if ``func`` is not already jitted, or to ``func`` directly otherwise.

        For compound or multiple statements, ``__bench_func`` returns a tuple of all
        assigned variables (excluding those starting with ``_``) so that
        ``block_until_ready`` can synchronize all computations.
        For example, ``a = x + y; b = a * 2`` becomes::

            @jax.jit
            def __bench_func(x, y):
                a = x + y
                b = a * 2
                return a, b

        Args:
            locals: Local variables from the execution context.
            globals: Global variables from the execution context.

        Returns:
            A tuple ``(new_code, new_globals)`` where ``new_code`` is the transformed
            code string and ``new_globals`` contains ``__bench_func``.
        """

        jax = sys.modules['jax']
        combined = locals | globals

        # Check if it's a simple call to an already jitted function
        if self._is_simple_jitted_call(combined):
            stmt = self.tree.body[0]
            expr = stmt.value  # type: ignore[attr-defined]
            args = [arg.id for arg in expr.args]
            func_name = expr.func.id
            bench_func = combined[func_name]
        else:
            # Create a jitted function from the code
            args = sorted(self._collect_used_names(locals, globals))
            bench_func = jax.jit(self._create_bench_func(args, combined))

        new_globals = combined | {'__bench_func': bench_func, 'jax': jax}
        new_code = f"""__rv = __bench_func({', '.join(args)})
jax.block_until_ready(__rv)"""
        return new_code, new_globals

    def _is_simple_jitted_call(self, combined: dict[str, Any]) -> bool:
        """Check if code is a simple call to an already jitted function.

        Returns True if:
        - There is exactly one statement
        - It's a simple statement (Expr, Assign, AnnAssign)
        - The value is a function call with simple Name arguments
        - The function is already jitted (has 'lower' attribute)
        """
        if len(self.tree.body) != 1:
            return False
        stmt = self.tree.body[0]
        if not isinstance(stmt, ast.Expr | ast.Assign | ast.AnnAssign):
            return False
        expr = stmt.value
        if not isinstance(expr, ast.Call):
            return False
        if not isinstance(expr.func, ast.Name):
            return False
        if not all(isinstance(arg, ast.Name) for arg in expr.args):
            return False
        if expr.keywords:
            return False
        func = combined.get(expr.func.id)
        return hasattr(func, 'lower')

    def _create_bench_func(self, args: list[str], combined: dict[str, Any]) -> Any:
        """Create a function from the tree body.

        The function takes the used variables as arguments and returns a tuple
        of all variables created in the code scope, so that ``block_until_ready``
        can be called on each of them.
        """
        body = list(self.tree.body)

        # Collect all assigned variable names
        assigned_names = self._collect_assigned_names()

        # Build return value
        return_value: ast.expr
        if len(assigned_names) == 1:
            # Single assigned variable: return it directly
            return_value = ast.Name(id=next(iter(assigned_names)), ctx=ast.Load())
            body.append(ast.Return(value=return_value))
        elif len(assigned_names) > 1:
            # Multiple assigned variables: return tuple
            return_value = ast.Tuple(
                elts=[ast.Name(id=name, ctx=ast.Load()) for name in sorted(assigned_names)],
                ctx=ast.Load(),
            )
            body.append(ast.Return(value=return_value))
        elif len(self.tree.body) == 1 and isinstance(self.tree.body[0], ast.Expr):
            # Single expression without assignment: return its value
            body = [ast.Return(value=self.tree.body[0].value)]

        # Create function def: def __bench_func(x, y, ...): <body>
        func_def = ast.FunctionDef(
            name='__bench_func',
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in args],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            type_params=[],
        )
        ast.fix_missing_locations(func_def)

        # Compile and execute to get the function
        module = ast.Module(body=[func_def], type_ignores=[])
        code = compile(module, '<benchmark>', 'exec')
        exec(code, combined)
        return combined['__bench_func']

    def _collect_assigned_names(self) -> set[str]:
        """Collect variable names assigned in the AST (Store context)."""

        class NameCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.names: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Store) and not node.id.startswith('_'):
                    self.names.add(node.id)
                self.generic_visit(node)

        collector = NameCollector()
        collector.visit(self.tree)
        return collector.names

    def _collect_used_names(self, locals: dict[str, Any], globals: dict[str, Any]) -> set[str]:
        """Collect variable names used in the AST."""

        class NameCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.names: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load):
                    self.names.add(node.id)
                self.generic_visit(node)

        collector = NameCollector()
        collector.visit(self.tree)

        # Filter: keep only names in locals/globals, exclude builtins and callables
        combined = locals | globals
        return {
            name for name in collector.names if name in combined and not callable(combined[name])
        }
