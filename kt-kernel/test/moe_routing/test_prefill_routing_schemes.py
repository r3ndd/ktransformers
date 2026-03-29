import importlib.util
import os

_this_dir = os.path.dirname(__file__)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_dir)))
_python_dir = os.path.join(_repo_root, "kt-kernel", "python")

spec = importlib.util.spec_from_file_location(
    "routing_schemes", os.path.join(_python_dir, "moe_routing", "routing_schemes.py")
)
routing_schemes = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(routing_schemes)


PrefillBlockMeanRouting = routing_schemes.PrefillBlockMeanRouting
PrefillFullMeanRouting = routing_schemes.PrefillFullMeanRouting


def test_prefill_block_mean_progressive_average():
    s = PrefillBlockMeanRouting(window_size=2)
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]

    out1 = s.smooth_scores(0, v1)
    s.observe(0, v1)
    out2 = s.smooth_scores(0, v2)

    assert out1 == v1
    assert out2[0] == 0.5
    assert out2[1] == 0.5


def test_prefill_full_mean_running_mean():
    s = PrefillFullMeanRouting()
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]

    out1 = s.smooth_scores(0, v1)
    s.observe(0, v1)
    out2 = s.smooth_scores(0, v2)
    s.observe(0, v2)
    out3 = s.smooth_scores(0, [0.2, 0.8])

    assert out1 == v1
    assert out2 == [1.0, 0.0]
    assert out3 == [0.5, 0.5]
