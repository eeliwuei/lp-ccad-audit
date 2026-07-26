"""LP-CCAD audit release: the three code units the paper's claims rest on.

* ``projection_loss``    -- the label-space projection losses (the method under audit).
* ``evaluation_adapter`` -- teacher/GT alignment, including the integer letterbox
                            padding convention that the E1 correction fixed.
* ``schedule``           -- the dose-matched 2x2 factorial schedule builder.

Nothing here imports the training framework; ``projection_loss`` needs only
PyTorch. See docs/REPRODUCIBILITY.md for what is and is not runnable from this
repository alone.
"""

__all__ = ["projection_loss", "evaluation_adapter", "schedule"]
__version__ = "0.1.0"
