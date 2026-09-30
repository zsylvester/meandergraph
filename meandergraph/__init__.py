"""
meandergraph: describe, analyze, and visualize the migration of meandering
river channels through time using directed graphs.

The implementation is split across submodules by concern (see
IMPROVEMENT_PLAN.md Phase 4.1):

- geometry     -- low-level shapely helpers shared by the rest of the package
- dtw          -- dynamic time warping (numba; exact and coarse-to-fine)
- correlation  -- dynamic-time-warping correlation of successive lines
- graph        -- the line graph (channel + radial edges)
- polygons     -- polygon graphs built from a line graph
- bars         -- Scroll / Bar objects and the bar-building pipeline
- plot         -- plotting functions

Everything public is re-exported here, so `import meandergraph as mg` and
`mg.correlate_curves(...)` etc. keep working exactly as when this was a
single flat module.
"""
from .geometry import *
from .dtw import *
from .correlation import *
from .graph import *
from .polygons import *
from .plot import *
from .bars import *
