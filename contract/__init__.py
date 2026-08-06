"""Hardware contracts for this project.

Currently one: `servo_contract`, the mapping between policy actions, board
angles, and FEETECH STS3215 servo counts.

The previous contract here was a vendored copy of an external simulator's file,
guarded by drift checks against that checkout. Both are gone. This project no
longer depends on that repository, and a contract describing hardware we do not
have is worse than none -- its servo constants (`servo_home=500`,
`servo_limits=(100, 900)`) described a 0-1000 device, while the STS3215 is
0-4095.

The policy observation spec is deliberately absent. It belongs with the
simulator and training work, and writing it before those decisions exist would
repeat exactly the mistake above: constants pinned to assumptions nobody has
tested yet.
"""
from __future__ import annotations

from .servo_contract import (  # noqa: F401
    CONTRACT_VERSION,
    AxisCalibration,
    CalibrationError,
    ServoContract,
)
