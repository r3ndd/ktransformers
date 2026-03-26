"""Test package exports - works without full kt_kernel installation."""

import sys
import os

# Add the python directory to path for standalone testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from moe_routing import (
    EMAScoreAveragingRouting,
    RoutingTraceCollector,
    simulate_routing_scheme,
    temporal_reuse_curve,
    TwoTimescaleSoftmaxRouting,
    TwoTimescalePlusCurrentSoftmaxRouting,
    TwoTimescaleEMARouting,
)


def test_exports_available():
    assert RoutingTraceCollector is not None
    assert temporal_reuse_curve is not None
    assert simulate_routing_scheme is not None
    assert EMAScoreAveragingRouting is not None
    assert TwoTimescaleEMARouting is not None
    assert TwoTimescaleSoftmaxRouting is not None
    assert TwoTimescalePlusCurrentSoftmaxRouting is not None
