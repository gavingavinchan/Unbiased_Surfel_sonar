"""Explicit GPU-only skips in CPU CI; unexpected import failures remain errors."""
import os
import pytest


def pytest_collection_modifyitems(items):
    if os.environ.get("SONAR_CPU_CI") == "1":
        skip = pytest.mark.skip(reason="CPU CI: GPU/CUDA coverage requires serial desktop evidence")
        for item in items:
            if item.get_closest_marker("cuda") or item.get_closest_marker("gpu"):
                item.add_marker(skip)
