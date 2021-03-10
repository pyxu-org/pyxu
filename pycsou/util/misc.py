# #############################################################################
# misc.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Miscellaneous functions.
"""

from typing import Tuple, Optional
import numpy as np
import re

def is_range_broadcastable(shape1: Tuple[int, int], shape2: Tuple[int, int]) -> bool:
    r"""
    Check if two shapes satisfy Numpy's broadcasting rules.

    Parameters
    ----------
    shape1: Tuple[int, int]
    shape2: Tuple[int, int]

    Returns
    -------
    bool
         ``True`` if broadcastable, ``False`` otherwise.

    Examples
    --------

    .. testsetup::

       from pycsou.util.misc import is_range_broadcastable

    .. doctest::

       >>> is_range_broadcastable((3,2), (1,2))
       True
       >>> is_range_broadcastable((3,2), (4,2))
       False
    """
    if shape1[1] != shape2[1]:
        return False
    elif shape1[0] == shape2[0]:
        return True
    elif shape1[0] == 1 or shape2[0] == 1:
        return True
    else:
        return False


def range_broadcast_shape(shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Tuple[int, int]:
    r"""
    Given two shapes, determine broadcasting shape.

    Parameters
    ----------
    shape1: Tuple[int, int]
    shape2: Tuple[int, int]

    Returns
    -------
    Tuple[int, int]
        Broadcasting shape.

    Raises
    ------
    ValueError
        If the two shapes cannot be broadcasted.

    Examples
    --------

    .. testsetup::

       from pycsou.util.misc import range_broadcast_shape

    .. doctest::

       >>> range_broadcast_shape((3,2), (1,2))
       (3, 2)

    """
    if not is_range_broadcastable(shape1, shape2):
        raise ValueError('Shapes are not (range) broadcastable.')
    shape = tuple(np.fmax(shape1, shape2).tolist())
    return shape


def peaks(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Matlab 2D peaks function.

    Peaks is a function of two variables, obtained by translating and scaling Gaussian distributions (see `Matlab's peaks function <https://www.mathworks.com/help/matlab/ref/peaks.html>`).
    This function is useful for testing purposes.

    Parameters
    ----------
    x: np.ndarray
        X coordinates.
    y: np.ndarray
        Y coordinates.

    Returns
    -------
    np.ndarray
        Values of the 2D function ``peaks`` at the points specified by the entries of ``x`` and ``y``.

    Examples
    --------
    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.util.misc import peaks

       x = np.linspace(-3,3, 1000)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X,Y)
       plt.figure()
       plt.imshow(Z)

    """
    z = 3 * ((1 - x) ** 2) * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - (1 / 3) * np.exp(-(x + 1) ** 2 - y ** 2)
    return z


def beamer2rst(input_file, output_file: Optional[str] = None):
    if output_file is None:
        output_file = f'{input_file.split(".")[0]}.rst'
    with open(output_file, 'w') as out_file:
        with open(input_file, 'r') as in_file:
            file_content = in_file.read()
        frames = re.findall(r'\\begin\{frame\}(.*?)\\end\{frame\}', file_content, flags=re.DOTALL)
        for frame in frames:
            frame = re.sub(r'\$\$(?P<EQ>.*?)\$\$', r'\n.. math::\n\n   \g<EQ>', frame, flags=re.DOTALL)
            frame = re.sub(r'\$(?P<EQ>.*?)\$', r':math:`\g<EQ>`', frame, flags=re.DOTALL)
            frame = re.sub(r'\\begin\{equation\}(?P<EQ>.*?)\\end\{equation\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{equation\*\}(?P<EQ>.*?)\\end\{equation\*\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{align\*\}(?P<EQ>.*?)\\end\{align\*\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{align\}(?P<EQ>.*?)\\end\{align\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{itemize\}(?P<items>.*?)\\end\{itemize\}', r'\n\g<items>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{enumerate\}(?P<items>.*?)\\end\{enumerate\}', r'\n\g<items>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\item\s*(?P<item>\S.*?)', r'* \g<item>', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\cite\[(?P<details>.*?)\]\{(?P<citation>.*?)\}', r'\g<details> of [\g<citation>]_',
                           frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])cal', r'\\mathcal{\g<symbol>}', frame)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])scr', r'\\mathcal{\g<symbol>}', frame)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])bf', r'\\mathbf{\g<symbol>}', frame)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])bb', r'\\mathbb{\g<symbol>}', frame)
            frame = re.sub(r'\\bb(?P<symbol>[a-zA-Z])', r'\\mathbf{\g<symbol>}', frame)
            frame = re.sub(r'\\R', r'\\mathbb{R}', frame)
            frame = re.sub(r'\\N', r'\\mathbb{N}', frame)
            frame = re.sub(r'\\Q', r'\\mathbb{Q}', frame)
            frame = re.sub(r'\\bm\{(?P<symbol>.*?)\}', r'\\mathbf{\g<symbol>}', frame)
            frame = re.sub(r'\\emph\{(?P<expression>.*?)\}', r'*\g<expression>*', frame)
            frame = re.sub(r'\\textbf\{(?P<expression>.*?)\}', r'**\g<expression>**', frame)
            frame = re.sub(r'(\\green|\\blue|\\red|\\orange|\\purple)\{(?P<expression>.*?)\}', r'\g<expression>',
                           frame)
            out_file.write(frame)


if __name__ == '__main__':
    beamer2rst('pycsou/util/part1.tex')
