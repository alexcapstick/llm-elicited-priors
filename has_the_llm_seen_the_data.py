import json
import argparse
import tqdm
import numpy as np
from pathlib import Path

from llm_elicited_priors.datasets import load_raw_dataset_frame
from llm_elicited_priors.gpt import GPTOutputs
from llm_elicited_priors.memory_check import header_completion_test, row_completion_test
from llm_elicited_priors.metrics import levenshtein_score

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--dataset",
    type=str,
    default=[
        "breast_cancer",
        "california_housing",
        "heart_disease",
        "wine_quality",
        "fake_data",
    ],
    nargs="+",
)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument(
    "--save_path",
    type=str,
    default="./results/table_tests",
)
args = parser.parse_args()

result_save_path = Path(args.save_path)
result_save_path.mkdir(parents=True, exist_ok=True)

CLIENT_CLASS = GPTOutputs
CLIENT_KWARGS = dict(
    temperature=args.temperature,
    model_id="gpt-3.5-turbo-0125",
)

DATASETS = args.dataset
for d in DATASETS:
    assert d in [
        "breast_cancer",
        "california_housing",
        "heart_disease",
        "wine_quality",
        "fake_data",
    ]

for df_name in tqdm.tqdm(DATASETS, desc="Iterating over datasets"):
    print("\n", "\n", f"Starting tests for {df_name}", "\n", "\n")
    # setting random seed for reproducibility no matter
    # which datasets are being used
    RNG = np.random.default_rng(args.seed)

    # get data
    df = load_raw_dataset_frame(df_name)

    # get client
    client = CLIENT_CLASS(rng=RNG, **CLIENT_KWARGS)

    # ==== header test ====

    # run header test
    header_prompt, header_completion, llm_completion = header_completion_test(
        df, client, rng=RNG
    )

    # collating header results
    header_test_results_this_df = {
        df_name: {
            "header_prompt": header_prompt,
            "header_completion": header_completion,
            "llm_completion": llm_completion,
            "levenshtein_score": levenshtein_score(header_completion, llm_completion),
        }
    }

    # saving the header test for this df to file
    header_test_file_name = f"header_test_{df_name}.json"
    header_test_file_path = result_save_path.joinpath(header_test_file_name)
    with open(header_test_file_path, "w") as f:
        json.dump(header_test_results_this_df, f)

    # ==== row test ====

    # run row test
    row_test_result = row_completion_test(df, client, rng=RNG)

    # collating row results
    row_test_results_this_df = {
        df_name: {
            run: {
                "row_prompt": test_prefix,
                "row_completion": test_suffix,
                "llm_completion": llm_completion,
                "levenshtein_score": levenshtein_score(test_suffix, llm_completion),
            }
            for run, (test_prefix, test_suffix, llm_completion) in enumerate(
                row_test_result
            )
        }
    }

    # saving the row test for this df to file
    row_test_file_name = f"row_test_{df_name}.json"
    row_test_file_path = result_save_path.joinpath(row_test_file_name)
    with open(row_test_file_path, "w") as f:
        json.dump(row_test_results_this_df, f)

    print("\n", "\n", f"Completed tests for {df_name}", "\n", "\n")
