"""409 visibility contract with conservative 416 acceleration.

The original implementation remains executable in ray_depth_oracle_409.py.
SONAR_VISIBILITY_ACCEL=oracle selects it; conservative is the default.
"""
import os
from .ray_depth_oracle_409 import quaternion_tangents, representation_coincidence


def ray_depth_visibility(*args, **kwargs):
    mode = os.environ.get('SONAR_VISIBILITY_ACCEL', 'conservative').strip().lower()
    if mode == 'oracle':
        from .ray_depth_oracle_409 import ray_depth_visibility as impl
    elif mode == 'conservative':
        from .ray_depth_accel import ray_depth_visibility as impl
    else:
        raise ValueError('SONAR_VISIBILITY_ACCEL must be oracle or conservative')
    return impl(*args, **kwargs)
