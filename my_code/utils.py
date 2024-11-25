import copy
import typing as t
import numpy as np
import pandas as pd
import difflib
from pathlib import Path


def inv_logistic(x: np.array) -> np.array:
    """
    The inverse of the logistic function.

    Arguments
    ----------

    x : array-like
        The input to the inverse logistic function.

    Returns
    ---------

    y : array-like
        The output of the inverse logistic function.

    """
    x = np.array(x)
    return np.log(x / (1 - x))


def load_prompts(path: Path, delim="\n\n") -> t.List[str]:
    """
    This function loads a text file of prompts into a list.
    In the text file, each prompt should be separated by
    a delimiter that is specified by the :code:`delim` argument.

    Arguments
    ----------
    path : Path
        The path to the text file containing the prompts.

    delim : str
        The delimiter that separates the prompts in the text file.
        Defaults to :code:`\n\n`.

    Returns
    --------

    prompts : list of str
        A list of prompts loaded from the text file.


    """
    return open(path).read().split(delim)


def load_nested_dict_to_pandas(
    results_dict: t.Dict[t.Any, t.Any],
    level_names: t.List[t.Any] = None,
) -> pd.DataFrame:
    """
    This function loads a nested dictionary into a pandas DataFrame.

    Arguments
    ----------

    results_dict : dict
        The nested dictionary to load.

    level_names : list
        The names of the levels in the dictionary.
        This can be used to name the columns of the
        outputted :code:`DataFrame`.
        Defaults to :code:`None`.


    Returns
    --------

    df : DataFrame
        The pandas DataFrame with the nested dictionary loaded.


    """

    # internal function to call recursively
    def internal_load_nested_dict_to_pandas(results_dict, level, level_names):
        df = pd.DataFrame()
        level = level + 1
        for key in results_dict.keys():
            if isinstance(results_dict[key], dict):
                df_output = internal_load_nested_dict_to_pandas(
                    results_dict[key], level, level_names
                )
                if level_names is None:
                    df_output.insert(0, f"level_{level}", key)
                else:
                    df_output.insert(0, level_names[level], key)
                df = pd.concat([df, df_output])
            else:
                try:
                    df[key] = [results_dict[key]]
                except:
                    df[key] = np.nan
        return df

    return internal_load_nested_dict_to_pandas(
        results_dict=results_dict, level=-1, level_names=level_names
    )


def cut_end_points(
    x: np.array,
    lower_percentile: float,
    upper_percentile: float,
) -> np.array:
    """
    Return an array that is a subset of :code:`x` that
    only contains data from with in the :code:`lower_percentile`
    and :code:`upper_percentile`.


    Arguments
    ----------

    - x : array-like
        The input array.

    - lower_percentile : float
        The lower percentile to cut the data at.
        This should be in the range [0, 100].

    - upper_percentile : float
        The upper percentile to cut the data at.
        This should be in the range [0, 100].

    Returns
    ---------
    - x_subset : array-like
        The subset of the input array that is within the
        :code:`lower_percentile` and :code:`upper_percentile`.

    """
    lower = np.percentile(x, lower_percentile)
    upper = np.percentile(x, upper_percentile)
    return x[(lower < x) & (x < upper)]


def cut_end_points_groupby(
    df: pd.DataFrame,
    groupby: str,
    cut_on: str,
    lower_percentile: float,
    upper_percentile: float,
) -> pd.DataFrame:
    """
    Return a dataframe that is a subset of :code:`df` that
    only contains data from with in the :code:`lower_percentile`
    and :code:`upper_percentile` in the :code:`cut_on` column
    grouped by the :code:`groupby` column.


    Arguments
    ----------

    - df : pd.DataFrame
        The input dataframe.

    - groupby : str
        The column to group the data by.

    - cut_on : str
        The column to cut the data on.

    - lower_percentile : float
        The lower percentile to cut the data at.
        This should be in the range [0, 100].

    - upper_percentile : float
        The upper percentile to cut the data at.
        This should be in the range [0, 100].

    Returns
    ---------
    - df_subset : array-like
        The subset of the input df that is within the
        :code:`lower_percentile` and :code:`upper_percentile`.

    """
    return df.loc[
        lambda x: x.groupby(groupby)[cut_on]
        .apply(
            lambda l: l.loc[
                (l > l.quantile(lower_percentile / 100))
                & (l < l.quantile(upper_percentile / 100))
            ]
        )
        .index.get_level_values(-1)
    ]


def make_list(x: t.Iterable) -> t.List[t.Any]:
    """
    This function will take an input and
    if any of its nested objects are arrays, they will be
    converted to lists.

    Arguments
    ----------

    x : any
        The input to convert to a list.

    Returns
    ---------

    x : list
        The input as a list.

    """

    # if x is an array then we can convert it straight away
    if isinstance(x, np.ndarray):
        x_new = x.tolist()
        return x_new

    # otherwise we need to find all the arrays in the object
    x_new = []
    for xi in x:
        if isinstance(xi, np.ndarray):
            xi = xi.tolist()
        if isinstance(xi, list):
            xi = make_list(xi)
        x_new.append(xi)
    return x_new


def find_best_matches(list1, list2):
    list1 = copy.deepcopy(list1)
    list2 = copy.deepcopy(list2)
    matches = []
    for item1 in list1:
        best_match = None
        highest_similarity = 0
        for item2 in list2:
            similarity = difflib.SequenceMatcher(None, item1, item2).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = item2
        matches.append((item1, best_match))
        list2.remove(best_match)  # Ensure each item is matched only once
    return matches
