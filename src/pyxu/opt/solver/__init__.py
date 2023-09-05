from pyxu.opt.solver.adam import *
from pyxu.opt.solver.cg import *
from pyxu.opt.solver.nlcg import *
from pyxu.opt.solver.pds import *
from pyxu.opt.solver.pgd import *
from pyxu.info.plugin import _load_entry_points

_load_entry_points(globals(), group="pyxu.solver")