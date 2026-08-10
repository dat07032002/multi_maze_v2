"""Policy-facing control layer: estimate, predict, act.

Distinct from ``tag_vision.control.tilt``, which sits one level lower and turns
a desired board angle into servo counts. Everything here runs above that, and
runs identically in simulation and on the rig.
"""
