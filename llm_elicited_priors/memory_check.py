"""
In this file, we implement two of the tests from:

Elephants Never Forget: Memorization and Learning 
of Tabular Data in Large Language Models 
(https://doi.org/10.48550/arXiv.2404.06209)

In particular, we implement the header completion test
and the row completion test. These tests are used to
determine if the language model has memorised the
header of the dataset and rows of the dataset
respectively.

We have adapted code from the original repository to
work with our implementation of the GPT-3 API and 
with pandas dataframes instead of csv files. We have
also provided further documentation for the functions and
removed one of the few-shot datasets from the header 
completion test as we use it as part of our experiments.

(https://github.com/interpretml/LLM-Tabular-Memorization-Checker)

The original repository is provided under the MIT license.

"""

import io
import typing as t
import numpy as np
import pandas as pd
import tqdm
from .datasets import load_raw_dataset_frame
from .gpt import LLMOutputs

# dataframe testing functions

header_system_prompt = """
You are an autocomplete bot for tabular datasets.
You will be prompted with parts of a tabular dataset.
Your task is to complete the dataset.
"""

row_system_prompt = """You are a helpful autocomplete 
bot for tabular datasets. Your task is to provide rows 
as they are contained in tabular datasets. The user 
provides a number of contiguous rows from a tabular 
dataset. You then provide the next row from the dataset.
"""


def df_to_string(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a string.
    """
    output = io.StringIO()
    df.to_csv(output, index=False)
    df_string = output.getvalue()
    output.close()
    return df_string


def header_completion_test(
    df: pd.DataFrame,
    client: LLMOutputs,
    split_rows: t.List[int] = [2, 4, 6, 8],
    completion_length: int = 500,
    system_prompt: str = header_system_prompt.replace("\n", " "),
    few_shot_datasets: t.List[str] = [
        "iris",
        "adult",
        "diabetes",
        "wine",
    ],
    rng: np.random.Generator = np.random.default_rng(),
) -> t.Tuple[str, str, str]:
    """
    Based on code given in:
    https://github.com/interpretml/LLM-Tabular-Memorization-Checker

    This function checks if the language model has
    memorised the first few rows of the dataset.

    The test is as follows:
    - The given dataframe will be converted into a string.
    - The string will be tructed at the :code:`split_row`
    row for each :code:`split_row` in :code:`split_rows`.
    - The language model will be asked to complete the next
    tokens, which will be measured against the next true
    :code:`completion_length` tokens.

    Few shot examples from the given datasets in :code:`few_shot_datasets`.

    Arguments
    ---------

    df: pd.DataFrame
        The dataset to check.

    client: LLMOutputs
        The client to use to generate completions.

    split_rows: List[int]
        The rows at which to split the dataset.
        Each integer is used for a separate experiment and
        determines the row at which the text is truncated
        and the language model asked to complete from.
        Defaults to :code:`[2, 4, 6, 8]`.

    completion_length: int
        The length of the completion to generate.
        Defaults to :code:`500`.

    system_prompt: str
        The prompt to use for the system.
        Defaults to the header_system_prompt.

    few_shot_files: List[str]
        The few-shot files to use. For the strings,
        the files will be loaded using the function
        :code:`load_raw_dataset_frame`. For the dataframes,
        they will be used directly.
        Defaults to :code:`["iris", "adult", "diabetes", "wine"]`.

    rng: np.random.Generator
        The random number generator to use.
        Defaults to :code:`np.random.default_rng()`.

    Returns
    -------

    Tuple[str, str, str]
        A tuple with the header prompt, the true completion, and the LLM completion.

    """

    few_shot_data = [
        (
            df_to_string(load_raw_dataset_frame(ds))
            if type(ds) == str  # only load the data if it is a string
            else df_to_string(ds)  # otherwise use the dataframe directly
        )
        for ds in few_shot_datasets
    ]

    df_string = df_to_string(df)
    df_rows_string = df_string.split("\n")

    # perform the test multiple times, cutting the dataset at
    # random positions in rows split_rows
    num_completions = -1
    header_prompt, llm_completion = None, None

    # for each of the row cuts
    for i_row in tqdm.tqdm(split_rows, desc="Testing header completion"):

        # find this row in the string
        offset = np.sum([len(row) for row in df_rows_string[: i_row - 1]])
        offset += rng.integers(
            len(df_rows_string[i_row]) // 3, 2 * len(df_rows_string[i_row]) // 3
        )
        # split the dataset at this row
        prefixes = [df_string[:offset]]
        # the completion is the next completion_length characters
        suffixes = [df_string[offset : offset + completion_length]]

        # construct the few-shot data: 1 example from each dataset
        few_shot = [
            ([fs_data[:offset]], [fs_data[offset : offset + completion_length]])
            for fs_data in few_shot_data
        ]

        test_prefixes = []
        test_suffixes = []
        responses = []

        # system prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # add the few-shot examples to the prompt
        for fs_prefixes, fs_suffixes in few_shot:
            fs_prefix, fs_suffix = None, None
            retries = 0
            # select a random prefix/suffix pair
            while (
                fs_prefix is None
                # assert that the test suffix is not contained in the
                # few-shot prefixes or suffixes
                or suffixes[0] in fs_prefix
                or suffixes[0] in fs_suffix
            ):
                fs_idx = rng.choice(len(fs_prefixes))
                fs_prefix = fs_prefixes[fs_idx]
                fs_suffix = fs_suffixes[fs_idx]
                retries += 1
                if retries > 100:
                    raise Exception(
                        """
                        Unable to construct a query where the desired output
                        is not contained in the few-shot data.
                        Did you provide the test dataset as few-shot example?
                        """
                    )
            messages.append({"role": "user", "content": fs_prefix})
            messages.append({"role": "assistant", "content": fs_suffix})

        # test observation
        test_prefix = prefixes[0]
        test_suffix = suffixes[0]
        messages.append({"role": "user", "content": test_prefix})

        # get the completion
        response = client.get_result(messages)

        # save the test data
        test_prefixes.append(test_prefix)
        test_suffixes.append(test_suffix)
        responses.append(response)

        # find the first digit where the response and the completion disagree
        idx = -1000
        for idx, (c, r) in enumerate(zip(df_string[offset:], response)):
            if c != r:
                break
        if idx == len(response) - 1 and response[idx] == df_string[offset + idx]:
            idx += 1  # no disagreement found, set idx to length of the response

        # is this the best completion so far?
        if idx > num_completions:
            num_completions = idx
            header_prompt = prefixes[0]
            llm_completion = response
            header_completion = df_string[offset : offset + len(llm_completion)]

    # return the best completion found
    test_triplet = header_prompt, header_completion, llm_completion

    return test_triplet


def row_completion_test(
    df: pd.DataFrame,
    client: LLMOutputs,
    n_prompt_rows: int = 10,
    n_tests: int = 25,
    n_fewshot_examples: int = 7,
    system_prompt: str = row_system_prompt.replace("\n", " "),
    rng: np.random.Generator = np.random.default_rng(),
) -> t.Tuple[str, str, str]:
    """
    Based on code given in:
    https://github.com/interpretml/LLM-Tabular-Memorization-Checker

    This function checks if the language model has
    memorised the header of the dataset.

    The test is as follows:
    - The given dataframe will be converted into a string.
    - :code:`n_tests` times:
        - Random chunks of :code:`n_prompt_rows` rows will be provided to the
        language model.
        - The language model will be asked to complete the next row.

    Few shot examples (in total :code:`n_fewshot_examples` examples)
    from the same dataset will be provided.

    Arguments
    ---------

    df: pd.DataFrame
        The dataset to check.

    client: LLMOutputs
        The client to use to generate completions.

    n_prompt_rows: int
        The number of rows to use as a prompt when
        testing whether the language model can complete the
        next row.
        Defaults to :code:`10`.

    n_tests: int
        The number of tests to run. For each
        test, a new row will be selected as the test row.
        Defaults to :code:`25`.

    n_fewshot_examples: int
        The number of few-shot examples to provide to the model.
        This will be chosen randomly from the dataset.
        Defaults to :code:`7`.

    system_prompt: str
        The prompt to use for the system.
        Defaults to the header_system_prompt.

    rng: np.random.Generator
        The random number generator to use.
        Defaults to :code:`np.random.default_rng()`.

    Returns
    -------

    test_triplet: Tuple[str, str, str]
        A tuple containing the test prefix, test suffix and the
        completion generated by the language model.

    """

    # getting df as string
    df_string = df_to_string(df)
    df_rows_string = df_string.split("\n")

    # calculate the proportion of duplicates
    duplicate_proportion = 1 - len(set(df_rows_string)) / len(df_rows_string)

    # prepare data
    prefixes = []
    suffixes = []

    # for each row in the df, make the prefixes from a sliding window
    # of n_prompt_rows rows and the suffixes the next row
    for idx in range(len(df_rows_string) - n_prompt_rows):
        prefixes.append("\n".join(df_rows_string[idx : idx + n_prompt_rows]))
        suffixes.append(df_rows_string[idx + n_prompt_rows])

    # shuffle the data
    idx = rng.permutation(len(prefixes))
    prefixes = [prefixes[i] for i in idx]
    suffixes = [suffixes[i] for i in idx]

    # the number of points to evaluate
    num_points = min(n_tests, len(prefixes))

    test_prefixes = []
    test_suffixes = []
    llm_completion = []

    # testing each point one at a time
    for i_testpoint in tqdm.trange(num_points, desc="Testing row completion"):
        # system prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # add the few-shot examples
        for _ in range(n_fewshot_examples):
            idx = None
            retries = 0
            # select a random prefix/suffix pair
            while (
                idx is None
                # assert that the demonstration
                # is not the test point
                or idx == i_testpoint
                # assert that the current test suffix is not
                # contained in the few-shot prefixes or suffixes
                or suffixes[i_testpoint] in prefixes[idx]
                or suffixes[i_testpoint] in suffixes[idx]
            ):
                idx = rng.choice(len(prefixes))
                retries += 1
                if retries > 100:
                    raise Exception(
                        """Unable to construct a query where the desired 
                        output is not contained in the few-shot data.
                        Did you provide the test dataset as few-shot example?"""
                    )

            # if the above is met then add the few-shot example
            # to the prompt
            prefix = prefixes[idx]
            suffix = suffixes[idx]
            messages.append({"role": "user", "content": prefix})
            messages.append({"role": "assistant", "content": suffix})

        # test observation
        test_prefix = prefixes[i_testpoint]
        test_suffix = suffixes[i_testpoint]
        messages.append({"role": "user", "content": test_prefix})

        # get the completion
        response = client.get_result(messages)

        # store prefix, suffix and response
        test_prefixes.append(test_prefix)
        test_suffixes.append(test_suffix)
        llm_completion.append(response)

    test_triplet = list(zip(test_prefixes, test_suffixes, llm_completion))

    return test_triplet
