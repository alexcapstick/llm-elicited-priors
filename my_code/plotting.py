import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import matplotlib
import numpy as np
from cycler import cycler
import typing as t


def save_fig(fig: plt.figure, file_name: str, **kwargs) -> None:
    """
    This function saves a pdf, png, and svg of the figure,
    with :code:`dpi=300`.


    Arguments
    ---------

    - fig: plt.figure:
        The figure to save.

    - file_name: str:
        The file name, including path, to save the figure at.
        This should not include the extension, which will
        be added when each file is saved.

    """

    fig.savefig(f"{file_name}.pdf", **kwargs)
    fig.savefig(f"{file_name}.png", dpi=300, **kwargs)
    fig.savefig(f"{file_name}.svg", **kwargs)


# colours
tol_muted = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
]

ibm = [
    "#648fff",
    "#fe6100",
    "#dc267f",
    "#785ef0",
    "#ffb000",
]


# colour map
def set_colour_map(colours: list = tol_muted):
    """
    Sets the default colour map for all plots.



    Examples
    ---------

    The following sets the colourmap to :code:`tol_muted`:

    .. code-block::

        >>> set_colour_map(colours=tol_muted)


    Arguments
    ---------

    - colours: list, optional:
        Format that is accepted by
        :code:`cycler.cycler`.
        Defaults to :code:`tol_muted`.

    """
    custom_params = {"axes.prop_cycle": cycler(color=colours)}
    matplotlib.rcParams.update(**custom_params)


# context functions
@contextlib.contextmanager
def temp_colour_map(colours=tol_muted):
    """
    Temporarily sets the default colour map for all plots.


    Examples
    ---------

    The following sets the colourmap to :code:`tol_muted` for
    the plotting done within the context:

    .. code-block::

        >>> with set_colour_map(colours=tol_muted):
        ...     plt.plot(x,y)


    Arguments
    ---------

    - colours: list, optional:
        Format that is accepted by
        :code:`cycler.cycler`.
        Defaults to :code:`tol_muted`.

    """
    set_colour_map(colours=colours)


@contextlib.contextmanager
def graph_theme(colours: t.List[str] = ibm, **kwargs):
    """
    Temporarily sets the default theme for all plots.


    Examples
    ---------

    .. code-block::

        >>> with graph_theme():
        ...     plt.plot(x,y)


    Arguments
    ---------

    - colours: t.List[str], optional:
        Any acceptable list to :code:`cycler`.
        Defaults to :code:`ibm`.


    """
    with matplotlib.rc_context():
        sns.set(context="paper", style="whitegrid")
        custom_params = {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=colours),
            "grid.alpha": 0.5,
            "grid.color": "#b0b0b0",
            "grid.linestyle": "--",
            "grid.linewidth": 1,
            # following requres latex
            "font.family": "serif",
            "font.serif": "Computer Modern",
            "text.usetex": True,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "boxplot.whiskerprops.linestyle": "-",
            "boxplot.whiskerprops.linewidth": 1,
            "boxplot.whiskerprops.color": "black",
            "boxplot.boxprops.linestyle": "-",
            "boxplot.boxprops.linewidth": 1,
            "boxplot.boxprops.color": "black",
            "boxplot.meanprops.markeredgecolor": "black",
            "boxplot.capprops.color": "black",
            "boxplot.capprops.linestyle": "-",
            "boxplot.capprops.linewidth": 1.0,
        }

        matplotlib.rcParams.update(**custom_params)
        matplotlib.rcParams.update(**kwargs)

        yield
