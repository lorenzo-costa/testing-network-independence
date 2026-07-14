import importlib


def test_solver_namespace_and_modules_are_importable():
    package = importlib.import_module("src.solvers")

    assert package.__name__ == "src.solvers"
    for module_name in (
        "weighted_network",
        "binary_network",
        "placeholder",
        "MaMa_uuuuu",
    ):
        module = importlib.import_module(f"src.solvers.{module_name}")
        assert module.__name__ == f"src.solvers.{module_name}"

