from pyxu.info.plugin import _load_entry_points
from pyxu.opt.solver.adam import *
from pyxu.opt.solver.cg import *
from pyxu.opt.solver.nlcg import *
from pyxu.opt.solver.pds import *
from pyxu.opt.solver.pgd import *

_load_entry_points(globals(), group="pyxu.opt.solver")
