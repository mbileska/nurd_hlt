import ast
from pathlib import Path


def test_train_hlt_calls_main_from_script_entrypoint():
    source_path = Path(__file__).resolve().parents[1] / "train_hlt.py"
    module = ast.parse(source_path.read_text())

    main_guards = [
        node
        for node in module.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(value, ast.Constant) and value.value == "__main__"
            for value in node.test.comparators
        )
    ]

    assert main_guards, "train_hlt.py must execute main() when run as a script"
    assert any(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "main"
        for guard in main_guards
        for statement in guard.body
    )
