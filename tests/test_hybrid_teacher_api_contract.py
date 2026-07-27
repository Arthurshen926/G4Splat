from outdoor.hybrid_teacher_api import SUPPORTED_TEACHER_PROTOCOLS


def test_causal_repair_teacher_protocol_is_publicly_loadable():
    assert (
        "cambridge_native_hybrid_teacher_v4_causal_repair"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
