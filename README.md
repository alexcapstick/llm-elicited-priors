# Using Large Language Models for Expert Prior Elicitation in Predictive Modelling


This is the corresponding to the paper **Using Large Language Models for Expert Prior Elicitation in Predictive Modelling**.

This repository contains the experimental code for reproducing the results in the paper and implementing the proposed method for new tasks.




## Installation


The code in this repository is written and tested in `Python 3.11.9`. 

After cloning this repository, to install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

This will install the following packages:
```python
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2 
tqdm==4.66.5
matplotlib==3.9.2
seaborn==0.13.2
pymc==5.17.0
pyarrow==17.0.0
ucimlrepo==0.0.7
openai==1.51.2
```

To run the code for performing Monte Carlo sampling on the approximated in-context prior, you must additionally [install JAX](https://jax.readthedocs.io/en/latest/installation.html) and [blackjax](https://blackjax-devs.github.io/blackjax/#installation) by following the instructions in the official documentation. 
Our code was tested using the JAX version `0.4.31` and blackJAX version `1.2.3`.


Further, to test open source models such as Llama and Qwen, you must [install torch](https://pytorch.org/get-started/locally/) by following the instructions in the official documentation, and the [Hugging Face Transformers library](https://huggingface.co/transformers/installation.html) by running the following command:

```bash
pip install transformers
```

Our code was run using the torch version `2.5.1` and the transformers version `4.46.0`.



## Usage

To elicit priors from a language model given information about a dataset, you can primarily use the following functions:

```python
from my_code.gpt import get_llm_elicitation
from my_code.gpt import get_llm_elicitation_for_dataset
```


`get_llm_elicitation` allows you to elicit a single prior distribution given a single user role and system role, as well as information about the dataset. The function returns a dictionary containing the elicited prior distribution.


`get_llm_elicitation_for_dataset` allows you to elicit a prior distribution for each user role and system role combination given information about the dataset. The function returns a numpy array containing all of the elicited prior distributions for each user role and system role combination.


Many examples of system roles and user roles can be found in the `prompts/elicitation` folder. 
The `get_llm_elicitation` function can be used to elicit priors for any of these roles, which can be adapted for any new task.
These prompts can be loaded using the function `my_code.utils import load_prompts`.

To elicit prior distributions from the wine quality dataset using GPT-3.5 turbo, we can use the following code (also found in `example_dataset_elicitation.ipynb`):

```python
# import the necessary functions and classes
from my_code.utils import load_prompts
from my_code.gpt import GPTOutputs, get_llm_elicitation_for_dataset
from my_code.datasets import load_wine_quality

# wrapper for language models
# see my_code.gpt for more details
CLIENT_CLASS = GPTOutputs
CLIENT_KWARGS = dict(
    temperature=0.1,
    model_id="gpt-3.5-turbo-0125",
    result_args=dict(
        response_format={"type": "json_object"},
    ),
)

# load the dataset which contains information
# about the feature names, target names, and 
# the dataset itself
wine_quality = load_wine_quality()

# load the prompts for the system and user roles
system_roles = load_prompts("prompts/elicitation/system_roles_wine_quality.txt")
user_roles = load_prompts("prompts/elicitation/user_roles_wine_quality.txt")

# create the llm client
client = CLIENT_CLASS(**CLIENT_KWARGS)

#### elicit the priors for the dataset ####
expert_priors = get_llm_elicitation_for_dataset(
    # the language model client
    client=client,
    # the prompts
    system_roles=system_roles,
    user_roles=user_roles,
    # the dataset contains the feature names as an attribute
    feature_names=wine_quality.feature_names.tolist(),
    # the dataset contains the target names as an attribute
    target_map={k: v for v, k in enumerate(wine_quality.target_names)},
    # print the prompts before passing them to the language model
    verbose=True,
)
```

The arguments in the `get_llm_elicitation_for_dataset` function can be adapted to any new task. 

The arguments to the `get_llm_elicitation` function are similar, except that it accepts a single system role and user role, to provide a single prior distribution.

Similarly, we provide a simple wrapper for using other language models as our elicitation function only requires that the client object has a `get_result` method than accepts the `messages` argument with a list of dictionaries containing the roles and content, for example:

```python
messages = [
    {
        "role" : "system",
        "content" : "You are an expert in predicting ..."
    },
    {
        "role" : "user",
        "content" : "I am a data scientist who wants a prior for ..."
    }
]
```

Please see `my_code.gpt.GPTOutputs`, `my_code.gpt.LlamaOutputs`, and `my_code.gpt.QwenOutputs` for examples of how to wrap different language models.


## Reproducing the experiments

All of the results presented in the paper are saved within the `results` folder, all of the elicited and approximated priors are saved within the `priors` folder, and all of the approximated posteriors are saved within the `posteriors` folder.

These results are used to produce the figures and tables presented in the paper, which can be generated using the notebooks in the root directory.

Note that when reproducing the experiments, the UTI dataset is private and cannot be shared. Therefore, some of the scripts and notebooks will error when trying to load the UTI dataset.

### Notebooks

The following describes each of the notebooks, ordered by the section in the paper. These notebooks load the results (already saved in the corresponding folders) and generate the figures and tables presented in the paper:

- `dataset_descriptions.ipynb`: This notebook describes the datasets used in the experiments.
- `example_dataset_elicitation.ipynb`: This notebook demonstrates how to elicit priors from a language model for the wine quality dataset.
- `llm_prior_vs_no_prior.ipynb`: This notebook compares the performance of a linear model with and without a prior elicited from a language model.
- `llm_prior_vs_no_prior_other_models.ipynb`: This notebook compares the performance of a linear model with and without priors elicited from different language models.
- `uti_prior_performance_vs_label_collection.ipynb`: Focusing on the UTI dataset, this notebook compares the performance of a linear model with and without priors elicited from a language model compared to the time taken to collect labels. This notebook will error, as the UTI data is private.
- `effects_of_bad_descriptions.ipynb`: This notebook compares the performance of a linear model with priors elicited from a language model with descriptions of increasing levels of information.
- `how_many_task_descriptions_is_enough.ipynb`: This notebook studies the difference in prior distributions as the number of task descriptions increases.
- `extracting_prior_internal_model.ipynb`: This notebook studies the approximate in-context prior distributions.
- `extracting_posterior_internal_model.ipynb`: This notebook studies the approximate in-context posterior distributions.
- `performing_mc_on_extracted_prior.ipynb`: This notebook studies the difference between the Monte Carlo sampling of the approximated in-context prior and the approximated in-context posterior.
- `measuring_bayes_factor.ipynb`: This notebook studies the Bayes factor between prior elicitation and in-context learning.
- `has_the_llm_seen_the_data.ipynb`: This notebook explores the extent to which the datasets have been memorised by the language model.
- `plots_for_demonstration.ipynb`: A simple plot used in Figure 1.

### Scripts

The following describes each of the scripts used to generate the results that are loaded by the notebooks:

- `llm_prior_vs_no_prior.py`: This script calculates the posterior performance of a linear model with and without a prior elicited from a language model.
- `llm_prior_vs_no_prior_with_expert_information.py`: This script calculates the posterior performance of a linear model with and without a prior elicited from a language model with additional expert information.
- `effects_of_bad_descriptions.py`: This script calculates the posterior performance of a linear model with priors elicited from a language model with descriptions of increasing levels of information.
- `how_many_task_descriptions_is_enough.py`: This script calculates the difference in prior distributions as the number of task descriptions increases.
- `extracting_prior_internal_model.py`: This script approximates the in-context prior distributions.
- `extracting_posterior_internal_model.py`: This script approximates the in-context posterior distributions.
- `performing_mc_on_extracted_prior.py`: This script calculates the difference between the Monte Carlo sampling of the approximated in-context prior and the approximated in-context posterior.
- `measuring_bayes_factor.py`: This script calculates the log-likelihood of the posterior prior predictions under prior elicitation and in-context learning.
- `has_the_llm_seen_the_data.py`: This script calculates the extent to which the datasets have been memorised by the language model.
- `make_fake_data.py`: This script generates the fake data used in the experiments.



## Citation

If you find this work useful, please consider citing our paper.

```
@article{capstick2024leveraging,
  title={Using Large Language Models for Expert Prior Elicitation in Predictive Modelling},
  author={Capstick, Alexander and Krishnan, Rahul and Barnaghi, Payam},
  year={2024}
}
```

