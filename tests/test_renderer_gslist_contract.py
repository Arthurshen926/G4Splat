import ast
from pathlib import Path


def _append_count(function: ast.FunctionDef, list_name: str) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == list_name
        and node.func.attr == "append"
        for node in ast.walk(function)
    )


def test_render_gslist_keeps_one_parameter_row_per_input_gaussian():
    """The multi-model renderer must concatenate aligned parameter tables."""
    tree = ast.parse(Path("2d-gaussian-splatting/gaussian_renderer/__init__.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_gslist"
    )

    # Each table is appended once per input point cloud. A second xyz append
    # would make means3D longer than opacity/scale/rotation/SH at rasterization.
    for list_name in (
        "means3D_list",
        "opacity_list",
        "scales_list",
        "rotations_list",
        "features_list",
    ):
        assert _append_count(function, list_name) == 1
