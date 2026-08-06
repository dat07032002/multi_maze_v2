"""Control layer: policy-facing commands down to servo counts.

Above the drivers in ``tag_vision.hardware``, below any policy. Everything here
is deterministic feedforward -- see ``tilt.py`` for why feedback is the wrong
tool on this rig.
"""
