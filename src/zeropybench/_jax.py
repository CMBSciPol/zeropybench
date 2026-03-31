import ast
import builtins
import inspect
import sys
from collections.abc import Callable
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['CodeASTParser']


class CodeASTParser:
    """Parse and transform code for JAX benchmarking."""

    def __init__(self, tree: ast.Module, globals: dict[str, Any]) -> None:
        self.tree = tree
        self.globals = globals

    @classmethod
    def from_code(cls, code: str, globals: dict[str, Any]) -> Self:
        """Instantiate an AST parser from a JAX code."""
        tree = ast.parse(code, mode='exec')
        return cls(tree, globals)

    def is_jax_context(self) -> bool:
        """Returns true if a JAX variable, module, or jitted function is used."""
        jax = sys.modules.get('jax')
        if jax is None:
            return False

        names = self._collect_loaded_names()
        for name in names:
            obj = self.globals.get(name)
            if obj is None:
                continue
            if self.is_jitted(obj):
                return True
            if self._contains_jax_arrays(obj, jax):
                return True
            if self._is_jax_module(obj):
                return True
        return False

    @staticmethod
    def _is_jax_module(obj: Any) -> bool:
        """Check if an object is a JAX module (jax, jax.numpy, etc.)."""
        module_name: str | None = getattr(obj, '__name__', None)
        if module_name is None:
            return False
        return module_name == 'jax' or module_name.startswith('jax.')

    @staticmethod
    def _contains_jax_arrays(obj: Any, jax: Any) -> bool:
        """Check if an object is a pytree containing JAX arrays."""
        leaves = jax.tree.leaves(obj)
        return any(isinstance(leaf, jax.Array) for leaf in leaves)

    def transform_jax_code(self) -> tuple[str, list[str], dict[str, Any]]:
        """Transform code into a benchmarkable jitted function call.

        A simple expression such as ``x + y`` is wrapped into a jitted function::

            @jax.jit
            def __bench_func(x, y):
                return x + y

        And the benchmark is performed on::

            __bench_func(x, y).block_until_ready()

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

        Bare expressions (without assignment) are converted into ``__exprN``
        assignments, unless they are calls to functions known to return ``None``
        (e.g., ``print``), which are left as-is.
        For example, ``a + 1; a + 2`` becomes::

            @jax.jit
            def __bench_func(a):
                __expr1 = a + 1
                __expr2 = a + 2
                return (__expr1, __expr2)

        Returns:
            A tuple ``(setup_code, args, new_globals)`` where ``setup_code`` is the
            code that defines ``__bench_func``, ``args`` is the list of argument names
            for the function call, and ``new_globals`` contains ``__bench_func``.
        """

        jax = sys.modules['jax']

        if self._is_simple_call():
            # Check if it's a simple call, in which case the function is reused and jitted if needed
            stmt = self.tree.body[0]
            expr = stmt.value  # type: ignore[attr-defined]
            args = [arg.id for arg in expr.args]
            bench_func, func_str = self._get_func_from_call(expr)
            if self.is_jitted(bench_func):
                setup_code = f'__bench_func = {func_str}'
            else:
                bench_func = jax.jit(bench_func)
                setup_code = f'__bench_func = jax.jit({func_str})'

        else:
            # Wrap the benchmarked code inside a jitted function
            args = sorted(self._collect_used_names())
            bench_func, setup_code = self._create_bench_func(args, self.globals)
            bench_func = jax.jit(bench_func)

        new_globals = self.globals | {
            '__bench_func': bench_func,
            'jax': jax,
        }
        return setup_code, args, new_globals

    def _is_simple_call(self) -> bool:
        """Check if code is a simple call to an already jitted function.

        We don't want to re-jit a function because some of its arguments may be static.

        Returns True if:
        - There is exactly one statement
        - It's a simple statement (Expr, Assign, AnnAssign)
        - The value is a function call with simple Name arguments
        - The function is either a Name (func) or Attribute (obj.method)
        """
        if len(self.tree.body) != 1:
            return False
        stmt = self.tree.body[0]
        if not isinstance(stmt, ast.Expr | ast.Assign | ast.AnnAssign):
            return False
        expr = stmt.value
        if not isinstance(expr, ast.Call):
            return False
        # Accept func(x, y) or obj.method(x, y)
        if not isinstance(expr.func, ast.Name | ast.Attribute):
            return False
        if not all(isinstance(arg, ast.Name) for arg in expr.args):
            return False
        if expr.keywords:
            return False
        return True

    def _get_func_from_call(self, expr: ast.Call) -> tuple[Callable[..., Any], str]:
        """Extract function and its string representation from a Call node.

        Returns:
            A tuple (func, func_str) where func is the callable and func_str
            is its string representation (e.g., "func" or "obj.method").
        """
        if isinstance(expr.func, ast.Name):
            func_name = expr.func.id
            return self.globals[func_name], func_name
        elif isinstance(expr.func, ast.Attribute):
            # obj.method -> get obj from globals, then get method
            func_str = ast.unparse(expr.func)  # "obj.method"
            obj_name = expr.func.value.id  # type: ignore[attr-defined]  # "obj" (only simple case)
            obj = self.globals[obj_name]
            method = getattr(obj, expr.func.attr)
            return method, func_str
        else:
            raise NotImplementedError(f'Unsupported function type: {type(expr.func)}')

    @staticmethod
    def is_jitted(func: Callable[..., Any]) -> bool:
        """Returns ``True`` if ``func`` is jitted."""
        return callable(func) and hasattr(func, 'lower')

    def _create_bench_func(self, args: list[str], combined: dict[str, Any]) -> tuple[Any, str]:
        """Create a function from the tree body.

        The function takes the used variables as arguments and returns a tuple
        of all variables created in the code scope, so that ``block_until_ready``
        can be called on each of them. A single bare expression is returned
        directly. Multiple bare expressions are converted into ``__exprN``
        assignments before building the return statement. Calls to functions
        known to return ``None`` are left as-is.

        Returns:
            A tuple ``(func, source)`` where ``func`` is the created function
            and ``source`` is the source code of the function.
        """
        body = list(self.tree.body)

        # Collect all user-assigned variable names (excludes _ prefixed)
        assigned_names = self._collect_assigned_names()

        # Identify bare expressions that are not None-returning calls
        bare_exprs = [
            stmt
            for stmt in body
            if isinstance(stmt, ast.Expr) and not self._returns_none(stmt.value)
        ]

        # If there is a single bare expression and no assigned variables,
        # return the expression directly without an intermediate assignment.
        if len(bare_exprs) == 1 and not assigned_names:
            target = bare_exprs[0]
            body = [stmt if stmt is not target else ast.Return(value=target.value) for stmt in body]
        else:
            # Convert bare expressions into assignments: __expr1 = ..., __expr2 = ...
            expr_names: list[str] = []
            expr_count = 0
            rewritten: list[ast.stmt] = []
            for stmt in body:
                if isinstance(stmt, ast.Expr) and not self._returns_none(stmt.value):
                    expr_count += 1
                    name = f'__expr{expr_count}'
                    expr_names.append(name)
                    rewritten.append(
                        ast.Assign(
                            targets=[ast.Name(id=name, ctx=ast.Store())],
                            value=stmt.value,
                        )
                    )
                else:
                    rewritten.append(stmt)
            body = rewritten

            # All names to return: user-assigned (sorted) + expression results (in order)
            return_names = sorted(assigned_names) + expr_names

            # Build return statement
            if len(return_names) == 1:
                body.append(ast.Return(value=ast.Name(id=return_names[0], ctx=ast.Load())))
            elif len(return_names) > 1:
                body.append(
                    ast.Return(
                        value=ast.Tuple(
                            elts=[ast.Name(id=n, ctx=ast.Load()) for n in return_names],
                            ctx=ast.Load(),
                        )
                    )
                )

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

        source = '@jax.jit\n' + ast.unparse(func_def)

        return combined['__bench_func'], source

    def _returns_none(self, expr: ast.expr) -> bool:
        """Check if an expression is a call to a function known to return None.

        Detects:
        - Builtins known to return None (e.g., ``print``)
        - Functions/methods with a ``-> None`` return annotation
        """
        _NONE_BUILTINS = {print}

        if not isinstance(expr, ast.Call):
            return False

        func = self._resolve_callable(expr.func)
        if func is None:
            return False

        if func in _NONE_BUILTINS:
            return True

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return False

        return sig.return_annotation is None

    def _resolve_callable(self, node: ast.expr) -> Any | None:
        """Resolve an AST function node to the actual callable from globals/builtins.

        Supports ``ast.Name`` (``func``) and ``ast.Attribute`` (``obj.method``).
        Returns ``None`` if the callable cannot be resolved.
        """
        if isinstance(node, ast.Name):
            result = self.globals.get(node.id)
            if result is None:
                result = getattr(builtins, node.id, None)
            return result
        if isinstance(node, ast.Attribute):
            obj = self._resolve_callable(node.value)
            if obj is not None:
                return getattr(obj, node.attr, None)
        return None

    def _collect_used_names(self) -> set[str]:
        """Collect variable names used in the AST.

        Keep only names in locals/globals, exclude builtins, callables, modules,
        and names that are assigned in the code (local variables).
        """
        from types import ModuleType

        loaded_names = self._collect_loaded_names()
        assigned_names = self._collect_assigned_names()
        return {
            name
            for name in loaded_names - assigned_names
            if name in self.globals
            and not callable(self.globals[name])
            and not isinstance(self.globals[name], ModuleType)
        }

    def _collect_loaded_names(self) -> set[str]:
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
        return collector.names

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
