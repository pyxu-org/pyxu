from pyxu.operator.blocks import *
from pyxu.operator.func import *
from pyxu.operator.linop import *
from pyxu.operator.map import *
from pyxu.info.plugin import _load_entry_points

_load_entry_points(globals(), group="pyxu.operator")