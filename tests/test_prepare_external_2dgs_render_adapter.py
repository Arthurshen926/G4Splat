import numpy as np

from scripts.prepare_external_2dgs_render_adapter import adapt_vertex_for_external_2dgs


def test_external_2dgs_renderer_adapter_preserves_local_parameters_and_adds_only_loc():
    vertex = np.zeros(
        2,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
        ],
    )
    vertex["x"] = [1.0, 2.0]
    vertex["rot_3"] = [3.0, 4.0]

    adapted = adapt_vertex_for_external_2dgs(vertex)

    assert adapted.dtype.names == (*vertex.dtype.names, "loc_0")
    for name in vertex.dtype.names:
        assert np.array_equal(adapted[name], vertex[name])
    assert np.array_equal(adapted["loc_0"], np.zeros(2, dtype=np.float32))
