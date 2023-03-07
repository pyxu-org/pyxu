from pycsou.opt.solver.cg import *
from pycsou.opt.solver.nlcg import *
from pycsou.opt.solver.pds import *
from pycsou.opt.solver.pgd import *
from pycsou.opt.solver.prox_adam import *
from pycsou.util.plugin import _load_entry_points

_load_entry_points(globals(), group="pycsou.solver")
