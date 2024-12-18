import os
import re
import io
import itertools
import typing as t
import numpy as np
import pandas as pd
import tqdm
from .datasets import load_raw_dataset_frame
import sklearn.metrics as skmetrics
import sklearn.linear_model as sklm
from openai import OpenAI

from .utils import inv_logistic, find_best_matches


class LLMOutputs(object):
    """
    Base class that allows you to interact with a language model.
    """


class LlamaOutputs(LLMOutputs):
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        top_p: float = 0.9,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        result_args: t.Optional[t.Dict[str, t.Any]] = {},
        quantisation: str = "none",
    ):
        """
        This class allows you to interact with the Meta-Llama model.
        If the language model is not already downloaded in the
        default hugging face hub directory, then it will be downloaded
        when this class is instantiated.

        Arguments
        ---------

        model_id : str
            The model ID to use.
            Defaults to :code:`"meta-llama/Meta-Llama-3-8B-Instruct"`.

        top_p : float
            The top-p value to use.
            Defaults to :code:`0.9`.

        temperature : float
            The temperature to use.
            Defaults to :code:`1.0`.

        max_new_tokens : int
            The maximum number of new tokens to generate.
            Defaults to :code:`2048`.

        result_args : Dict[str, Any]
            Additional arguments to pass to the model chat
            completion.
            Defaults to :code:`{}`.

        quantisation : str
            The quantisation to use.
            This can be one of :code:`"none"`, :code:`"bfloat16"`,
            or :code:`"int8"`, or :code:`"int4"`.
            Defaults to :code:`"none"`.

        """
        try:
            import transformers
            import torch
        except ImportError:
            raise ImportError(
                "Please ensure that the transformers and torch libraries are installed if using Llama."
            )

        print("model_id:", model_id)
        print("quantisation:", quantisation)

        model_pipeline_args = {}
        if quantisation in ["bfloat16", "int8", "int4"]:
            model_pipeline_args["torch_dtype"] = torch.bfloat16
            if quantisation == "int8":
                model_pipeline_args["quantization_config"] = (
                    transformers.BitsAndBytesConfig(load_in_8bit=True)
                )
            if quantisation == "int4":
                model_pipeline_args["quantization_config"] = (
                    transformers.BitsAndBytesConfig(load_in_4bit=True)
                )
        elif quantisation == "none":
            pass
        else:
            raise ValueError(
                "Please ensure that the quantisation is one of 'none', 'bfloat16', 'int8', or 'int4'."
            )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_pipeline_args,
            device_map="auto",
            # device_map="balanced_low_0",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.result_args = result_args

    def get_result(self, messages: t.List[t.Dict[str, str]]) -> str:
        """
        Arguments
        ---------

        messages : List[Dict[str, str]]
            A list of dictionaries with the role and content of the messages.


        Returns
        -------

        str
            The generated text.

        """

        result_args = self.result_args.copy()
        if "response_format" in result_args:
            response_format = result_args["response_format"]
            del result_args["response_format"]
            if response_format == {"type": "json_object"}:
                messages.append(
                    {
                        "role": "system",
                        "content": "I am only returning a JSON object with "
                        + "'mean' and 'std' for each feature and nothing else: {",
                    }
                )

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            do_sample=True,
            return_full_text=False,
            temperature=self.temperature,
            top_p=self.top_p,
            **result_args,
        )
        return outputs[0]["generated_text"]


class QwenOutputs(LLMOutputs):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-14B-Instruct",
        top_p: float = 0.9,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        result_args: t.Optional[t.Dict[str, t.Any]] = {},
        quantisation: str = "none",
    ):
        """
        This class allows you to interact with the Meta-Llama model.
        If the language model is not already downloaded in the
        default hugging face hub directory, then it will be downloaded
        when this class is instantiated.

        Arguments
        ---------

        model_id : str
            The model ID to use.
            Defaults to :code:`"meta-llama/Meta-Llama-3-8B-Instruct"`.

        top_p : float
            The top-p value to use.
            Defaults to :code:`0.9`.

        temperature : float
            The temperature to use.
            Defaults to :code:`1.0`.

        max_new_tokens : int
            The maximum number of new tokens to generate.
            Defaults to :code:`2048`.

        result_args : Dict[str, Any]
            Additional arguments to pass to the model chat
            completion.
            Defaults to :code:`{}`.

        quantisation : str
            The quantisation to use.
            This can be one of :code:`"none"`, :code:`"bfloat16"`,
            or :code:`"int8"`, or :code:`"int4"`.
            Defaults to :code:`"none"`.

        """
        try:
            import transformers
            import torch
        except ImportError:
            raise ImportError(
                "Please ensure that the transformers and torch libraries are installed if using Llama."
            )

        print("model_id:", model_id)
        print("quantisation:", quantisation)

        model_pipeline_args = {}
        if quantisation in ["bfloat16", "int8", "int4"]:
            model_pipeline_args["torch_dtype"] = torch.bfloat16
            if quantisation == "int8":
                model_pipeline_args["quantization_config"] = (
                    transformers.BitsAndBytesConfig(load_in_8bit=True)
                )
            if quantisation == "int4":
                model_pipeline_args["quantization_config"] = (
                    transformers.BitsAndBytesConfig(load_in_4bit=True)
                )
        elif quantisation == "none":
            pass
        else:
            raise ValueError(
                "Please ensure that the quantisation is one of 'none', 'bfloat16', 'int8', or 'int4'."
            )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_pipeline_args,
            device_map="auto",
            # device_map="balanced_low_0",
        )
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.result_args = result_args

    def get_result(self, messages: t.List[t.Dict[str, str]]) -> str:
        """
        Arguments
        ---------

        messages : List[Dict[str, str]]
            A list of dictionaries with the role and content of the messages.


        Returns
        -------

        str
            The generated text.

        """

        result_args = self.result_args.copy()
        if "response_format" in result_args:
            response_format = result_args["response_format"]
            del result_args["response_format"]
            if response_format == {"type": "json_object"}:
                messages.append(
                    {
                        "role": "system",
                        "content": "I am only returning a JSON object with "
                        + "'mean' and 'std' for each feature and nothing else: {",
                    }
                )

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            do_sample=True,
            return_full_text=False,
            temperature=self.temperature,
            top_p=self.top_p,
            **result_args,
        )
        return outputs[0]["generated_text"]


class GPTOutputs(LLMOutputs):
    def __init__(
        self,
        model_id: str = "gpt-4-turbo",
        temperature: float = 1.0,
        result_args: t.Optional[t.Dict[str, t.Any]] = {},
        rng: np.random._generator.Generator = None,
    ):
        """
        This class allows you to interact with the OpenAI models.

        Arguments
        ---------

        model_id : str
            The model ID to use.
            Defaults to :code:`"gpt-4-turbo"`.

        temperature : float
            The temperature to use.
            Defaults to :code:`0.6`.

        result_args : Dict[str, Any]
            Additional arguments to pass to the model chat
            completion.
            Defaults to :code:`{}`.

        rng : np.random._generator.Generator
            The random number generator to use
            to allow for reproducibility.
            Defaults to :code:`None`.

        """
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id
        self.temperature = temperature
        self.result_args = result_args
        self.rng = np.random.default_rng() if rng is None else rng

    def get_result(self, messages: t.List[t.Dict[str, str]]) -> str:
        """
        Arguments
        ---------

        messages : List[Dict[str, str]]
            A list of dictionaries with the role and content of the messages.


        Returns
        -------

        str
            The generated text.

        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            seed=int(self.rng.integers(1e9)),
            **self.result_args,
        )

        return response.choices[0].message.content


def get_llm_elicitation(
    client: LLMOutputs,
    system_role: t.Optional[str] = None,
    user_role: t.Optional[str] = None,
    task_title: t.Optional[str] = None,
    feature_names: t.Optional[t.List[str]] = None,
    target_map: t.Optional[t.Dict[str, int]] = None,
    verbose: bool = True,
    dry_run: bool = False,
) -> t.Dict[str, float]:
    """
    Given a task description, model, and feature names, this
    function will return a language model's guess of the weights
    of a linear model.


    Arguments
    ---------

    client: LLMOutputs
        An instance of the LLMOutputs class.

    system_role: str
        The role of the system.
        If this is not provided, then the :code:`task_title`,
        :code:`feature_names`, and :code:`target_map` must be provided.
        Additionally, even if this argument is provided, the :code:`task_title`,
        :code:`feature_names`, and :code:`target_map` can also be provided
        and the string :code:`sytem_role` can contain the following placeholders:
        :code:`'{task_title}'`, :code:`'{feature_names}'`, :code:`'{unique_targets}'`,
        and :code:`'{target_map}'` which will be filled in before prompting the
        language model.
        Defaults to :code:`None`.

    user_role: str
        The role of the user.
        If this is not provided, then the :code:`task_title`,
        :code:`feature_names`, and :code:`target_map` must be provided.
        Additionally, even if this argument is provided, the :code:`task_title`,
        :code:`feature_names`, and :code:`target_map` can also be provided
        and the string :code:`user_role` can contain the following placeholders:
        :code:`'{task_title}'`, :code:`'{feature_names}'`, :code:`'{unique_targets}'`,
        and :code:`'{target_map}'` which will be filled in before prompting the
        language model.
        Defaults to :code:`None`.

    task_title: str
        The title of the task that the model is being asked to
        predict.
        Defaults to :code:`None`.

    feature_names: List[str]
        A list of the feature names that the model is being asked
        to predict.
        Defaults to :code:`None`.

    target_map: Dict[str, int]
        A dictionary with the target names as keys and the values
        as the target values.
        Defaults to :code:`None`.

    verbose: bool
        Whether or not to print the system and user roles.
        Defaults to :code:`True`.

    dry_run: bool
        Whether or not to run the function in dry run mode.
        When in dry run mode, the function will not make any API
        requests and will return a mock response.
        Defaults to :code:`False`.


    Returns
    -------

    Dict[str, float]
        Hopefully, a dictionary with the feature names as keys and the weights
        as values.

    """

    if not isinstance(feature_names, list):
        raise ValueError("Please ensure that the feature_names is a list")
    if not isinstance(target_map, dict):
        raise ValueError(
            "Please ensure that the target_map is a dictionary with the target names as keys "
            "and the values as the target values"
        )

    example_response = {
        "feature_1": {"mean": "mean1", "std": "std1"},
        "feature_2": {"mean": "mean2", "std": "std2"},
        "feature_3": {"mean": "mean3", "std": "std3"},
    }

    if system_role is None:
        system_role = f"""
        You are an expert in 
        {task_title}. 
        You have access to an internal predictive model of 
        {task_title} 
        and are great at guessing the prior distribution of weights of a linear model.
        """

    else:
        system_role = system_role.format(
            task_title=task_title,
            feature_names="[" + "'" + "', '".join(feature_names) + "'" + "]",
            unique_targets=" or ".join(target_map.keys()),
            target_map=" and ".join([f"'{k}' = {v}" for k, v in target_map.items()]),
        )

    if user_role is None:
        user_role = f"""
        I am a data scientist with a dataset of {task_title} samples. 
        I would like to use your model to predict the diagnosis of my samples.
        I have a dataset that is made up of the following features:
        [ {"[" +  "'" + "', '".join(feature_names) +  "'" + "]"} ].
        All of the feature values are standardized using the z-score.
        By thinking about how each feature might be related to a diagnosis of 
        {' or '.join(target_map.keys())}, 
        and whether each feature is positively or negatively correlated with the 
        outcome of 
        {' and '.join([f"'{k}' = {v}" for k,v in target_map.items()])},
        I would like you to guess the 
        mean and standard deviation for a normal distribution prior for each feature
        for a logistic regression model that predicts the 
        {task_title}.
        Please respond with a JSON object with the feature names as keys 
        and a nested dictionary of mean and standard deviation as values.
        A positive mean indicates a positive correlation with the outcome,
        a negative mean indicates a negative correlation with the outcome,
        whilst a small standard deviation indicates that you are confident in your guess.
        Please only respond with a JSON, no other text.
        """

    else:
        user_role = user_role.format(
            task_title=task_title,
            feature_names="[" + "'" + "', '".join(feature_names) + "'" + "]",
            unique_targets=" or ".join(target_map.keys()),
            target_map=" and ".join([f"'{k}' = {v}" for k, v in target_map.items()]),
        )

    if verbose:
        print("System role", "\n", "---------", "\n", system_role)
        print("User query", "\n", "---------", "\n", user_role)

    system_role = system_role.replace("\n", " ")
    user_role = user_role.replace("\n", " ")

    if dry_run:
        return None

    result = client.get_result(
        [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_role},
        ]
    )

    processed_result = result.replace("\n", "").replace("```", "").replace("\\", "")
    if processed_result.startswith("json"):
        processed_result = processed_result[4:]

    if processed_result.startswith('"'):
        processed_result = processed_result[1:]

    if processed_result.endswith('"'):
        processed_result = processed_result[:-1]

    if not processed_result.replace(" ", "").startswith("{"):
        processed_result = "{" + processed_result

    if not processed_result.replace(" ", "").endswith("}"):
        processed_result = processed_result + "}"

    if not processed_result.replace(" ", "").endswith("}}"):
        processed_result = processed_result + "}}"
    try:
        llm_weights = {key: value for key, value in eval(processed_result).items()}
    except:
        print("tried the processed result:", processed_result)
        print("the original was:", result)
        raise ValueError("Could not evaluate the response from the language model.")

    return llm_weights


def data_points_to_sentence(
    x: np.array,
    feature_names: t.List[str] = None,
    y: t.Optional[np.array] = None,
) -> str:
    """
    This converts a given array of data points and
    optionally the target variable into a sentence.
    The sentence will have the following form:

    .. code-block::
        '''
        # data point 1
        {
            'feature_1' = value_1,
            'feature_2' = value_2,
        },
        # data point 2
        {
            'feature_1' = value_1,
            'feature_2' = value_2,
        },
        # data point 3
        {
            'feature_1' = value_1,
            'feature_2' = value_2,
        }"
        '''

    Arguments
    ---------

    x : np.array
        The data points to convert to a sentence.

    feature_names : t.List[str]
        The names of the features.
        This should be a list of strings.
        Defaults to :code:`None`.

    y : t.Optional[np.array]
        The target variable.
        Defaults to :code:`None`.

    """
    sentence_of_data_points = ""

    if feature_names is None:
        raise ValueError("Feature names must be provided.")

    for index, x_i in enumerate(x):
        sentence_of_data_points += "{"
        sentence_of_data_points += "\n"
        for feature_name, value in zip(feature_names, x_i):
            sentence_of_data_points += f"'{feature_name}' = {value}, "
            sentence_of_data_points += "\n"
        sentence_of_data_points += "}"
        sentence_of_data_points += "\n"
        if y is not None:
            sentence_of_data_points += f"gives y = {y[index]}, "
        sentence_of_data_points += "\n"

    return sentence_of_data_points


def get_llm_predictions(
    client: LLMOutputs,
    x: np.array,
    system_role: str = None,
    final_message: str = None,
    feature_names: t.List[str] = None,
    demonstration: t.Optional[t.List[np.array]] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> np.array:
    """
    Get predictions for a linear model from a language model.

    Arguments
    ---------

    client: LLMOutputs
        An instance of the LLMOutputs class.

    x : np.array
        The data to be used to predict the target variable.

    system_role : str
        The system role in the conversation.
        Defaults to :code:`None`.

    final_message : str
        The final message to display to the language model.
        Defaults to :code:`None`.

    feature_names : t.List[str]
        The names of the features.
        This should be a list of strings.
        Defaults to :code:`None`.

    demonstration : t.List[np.array]
        The demonstration data to use.
        This should be a list of numpy arrays
        with the first array containing the features
        and the second array containing the target variable.
        Defaults to :code:`None`.

    dry_run : bool
        Whether to run the function in dry run mode.
        Defaults to :code:`False`.

    verbose : bool
        Whether to print information.
        Defaults to :code:`False`.

    Returns
    -------

    np.array
        The predictions for the target variable.

    """

    assert isinstance(x, np.ndarray), "x must be a numpy array."
    if system_role is None:
        raise ValueError("System role must be provided.")
    if feature_names is None:
        raise ValueError("Feature names must be provided.")

    user_role = f"""
    I will give you {x.shape[1]} features, which are: 
    [ '{"', '".join(feature_names)}' ], 
    You then need to make predictions y from 
    the data.
    """

    if demonstration is not None:
        # turn demonstrations into text for in-context learning
        demonstration_sentence = data_points_to_sentence(
            x=demonstration[0], feature_names=feature_names, y=demonstration[1]
        )

        some_example_data = (
            "Here are some examples of data points and their predictions: \n"
            + demonstration_sentence
        )

    sentence_of_data_points = data_points_to_sentence(x=x, feature_names=feature_names)

    some_new_data = (
        "Here are some data points, please make predictions for y: \n"
        + sentence_of_data_points
    )

    if final_message is None:
        final_message = ""

    final_message = (
        final_message
        + f"""
    You should return {x.shape[0]} predictions.
    """
    )

    if verbose:
        print("System role", "\n", "---------", "\n", system_role)
        print("User query", "\n", "---------", "\n", user_role)
        if demonstration is not None:
            print("Demonstration", "\n", "---------", "\n", some_example_data)
        print("New data", "\n", "---------", "\n", some_new_data)
        print("Final message", "\n", "---------", "\n", final_message)

    messages = [
        {"role": "system", "content": system_role.replace("\n", " ")},
        {"role": "user", "content": user_role.replace("\n", " ")},
    ]
    if demonstration is not None:
        messages.append(
            {"role": "system", "content": some_example_data.replace("\n", " ")}
        )

    messages.extend(
        [
            {"role": "user", "content": some_new_data.replace("\n", " ")},
            {"role": "user", "content": final_message.replace("\n", " ")},
        ]
    )

    if dry_run:
        print(messages)
        output = None
    else:
        result = client.get_result(
            messages=messages,
        )
        # try to turn output into an array, if that fails try [ output ]
        # and if that fails, raise error

        result

        ### function with things to try and:
        possible_process_funcs = [
            lambda x: np.array(eval(x)),
            lambda x: np.array(eval("[" + x + "]")),
            lambda x: np.array(eval(x.replace("\n", ", "))),
            lambda x: np.array(eval(x.replace(" ", ", "))),
            lambda x: np.array(eval("[" + x.replace("\n", ", ") + "]")),
            lambda x: np.array(eval("[" + x.replace(" ", ", ") + "]")),
            # regex to match inly floats and integers in the string
            lambda x: np.array([float(d) for d in re.findall(r"\d+\.\d+|\d+", x)]),
        ]

        not_working = True
        i = -1
        while not_working:
            try:
                i += 1
                output = possible_process_funcs[i](result)
                not_working = False
            except:
                if i == len(possible_process_funcs):
                    not_working = False
                    print(result)
                    raise ValueError("The LLM did not return a valid response.")

    if verbose:
        print("Output", "\n", "---------", "\n", output)

    return output


def get_llm_elicitation_for_dataset(
    client: LLMOutputs,
    system_roles: t.List[str],
    user_roles: t.List[str],
    feature_names: t.List[str],
    target_map: t.Dict[str, int],
    verbose=True,
    std_lower_clip=1e-3,
) -> t.List[np.array]:
    """
    Given a task description, model, and feature names, this
    function will return a language model's guess of the weights
    of a linear model.


    Arguments
    ---------

    client: LLMOutputs
        An instance of the LLMOutputs class.

    system_roles: List[str]
        The roles of the system.
        These are used to describe the language model's role.
        This should mention that the language model is an
        expert in the field of the task.
        The :code:`feature_names`, and :code:`target_map` should also be provided
        and the strings in :code:`sytem_roles` can contain the following placeholders:
        :code:`'{feature_names}'` and :code:`'{target_map}'`
        which will be filled in before prompting the language model.
        Defaults to :code:`None`.

    user_roles: List[str]
        The roles of the user.
        These are used to describe the users role.
        This should describe that the user wants to elicit a prior
        distribution for a given task.
        The :code:`feature_names`, and :code:`target_map` should also be provided
        and the strings in :code:`sytem_roles` can contain the following placeholders:
        :code:`'{feature_names}'` and :code:`'{target_map}'`
        which will be filled in before prompting the language model.
        Defaults to :code:`None`.

    feature_names: List[str]
        A list of the feature names that the model is being asked
        to predict.
        Defaults to :code:`None`.

    target_map: Dict[str, int]
        A dictionary with the target names as keys and the values
        as the target values.
        Defaults to :code:`None`.

    verbose: bool
        Whether or not to print the system and user roles.
        Defaults to :code:`True`.

    std_lower_clip: float
        The lower bound for the standard deviation.
        If the standard deviation is lower than this value,
        then it will be replaced with this value.
        Defaults to :code:`1e-3`.


    Returns
    -------

    t.List[np.array]
        The elicitation priors for the given system roles and user roles.

    """

    pbar = tqdm.tqdm(
        total=len(system_roles) * len(user_roles),
        desc=f"Getting priors for {len(system_roles) * len(user_roles)} combinations",
        disable=not verbose,
    )

    priors = []

    for i, (sr, ur) in enumerate(itertools.product(system_roles, user_roles)):

        gpt_elicitation = get_llm_elicitation(
            client=client,
            system_role=sr,
            user_role=ur,
            feature_names=feature_names,
            target_map=target_map,
            verbose=verbose,
        )

        try:

            if isinstance(list(gpt_elicitation.values())[0], dict):

                possible_mean_keys = [
                    "mean",
                    "mu",
                    "m",
                    "average",
                    "expected_value",
                    "expectedvalue",
                    "expected value",
                ]
                possible_std_keys = [
                    "std",
                    "sigma",
                    "s",
                    "standard deviation",
                    "standard_deviation",
                    "standarddeviation",
                    "stddev",
                    "std_dev",
                    "std dev",
                    "std_deviation",
                    "std deviation",
                    "stddeviation",
                    "standard_dev",
                    "standarddev",
                    "standard dev",
                    "standardDeviation",
                ]

                mean_key = None
                std_key = None
                gpt_4_keys = list(gpt_elicitation.values())[0].keys()

                for key in possible_mean_keys:
                    if key in gpt_4_keys:
                        mean_key = key
                        break

                for key in possible_std_keys:
                    if key in gpt_4_keys:
                        std_key = key
                        break

                if (mean_key is None) or (std_key is None):
                    print(gpt_4_keys)
                    raise ValueError("Could not find mean and std keys in GPT-4 output")

                gpt_elicitation = {
                    key: [value[mean_key], value[std_key]]
                    for key, value in gpt_elicitation.items()
                }

            gpt_bias = [[0.0, 1.0]]
            gpt_weights = []

            features_in_dataset = feature_names
            features_in_elicitation = list(gpt_elicitation.keys())

            matches = find_best_matches(features_in_elicitation, features_in_dataset)

            matches = {item2: item1 for item1, item2 in matches}

            print("\n")
            print("matched features:")
            print(
                *[f"{k}: {v}: {gpt_elicitation[v]}" for k, v in matches.items()],
                sep="\n",
            )
            print("\n")

            for f_dataset in feature_names:
                f_elicitation = matches[f_dataset]
                gpt_weights.append(gpt_elicitation[f_elicitation])

            gpt_weights = np.array(gpt_weights).astype(float)

            if np.any(gpt_weights[:, 1] < std_lower_clip):
                print(
                    "Zero standard deviation found in elicitation, repalced with",
                    std_lower_clip,
                )
                gpt_weights[gpt_weights[:, 1] < std_lower_clip, 1] = std_lower_clip
                print(gpt_weights)

            prior = np.concatenate([gpt_bias, gpt_weights], axis=0)
            print("elicitation prior:\n", prior)

        except Exception as e:
            print(f"Error getting prior: {e}")
            print(gpt_elicitation)
            raise e

        priors.append(prior)

        pbar.update(1)

    pbar.close()

    return priors


def sample_approximate_llm_internal_predictive_model_parameters(
    client: LLMOutputs,
    n_samples: int = 20,
    n_datapoints_in_sample: int = 25,
    required_model: str = "linear",
    system_role: str = None,
    final_message: str = None,
    feature_names: t.List[str] = None,
    rng: np.random._generator.Generator = None,
    demonstration: t.List[np.array] = None,
    x_sample_low: float = -5,
    x_sample_high: float = 5,
    return_mle_loss_and_samples: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> np.array:
    """
    Get predictions for a linear model from GPT-4.

    Arguments
    ---------

    client: LLMOutputs
        An instance of the LLMOutputs class.

    n_samples : int
        The number of times to sample the language model.

    n_datapoints_in_sample : int
        The number of data points in each sample.

    required_model : str
        The required internal model that the language model
        must use.

    system_role : str
        The system role in the conversation.
        Defaults to :code:`None`.

    feature_names : typing.List[str]
        The names of the features.
        This should be a list of strings.
        Defaults to :code:`None`.

    rng : np.random._generator.Generator
        The random number generator to use.

    demonstration : t.List[np.array]
        The demonstration data to use.
        This should be a list of numpy arrays
        with the first array containing the features
        and the second array containing the target variable.
        Defaults to :code:`None`.

    x_sample_low : float
        The lower bound for the samples. A uniform distribution
        is used to sample the values.
        Defaults to :code:`-5`.

    x_sample_high : float
        The upper bound for the samples. A uniform distribution
        is used to sample the values.
        Defaults to :code:`5`.

    return_mle_loss_and_samples : bool
        Whether to return the MLE loss for the approximated
        model and the samples. In this case, the MLE loss
        is calculated over the logits of the predictions.
        Defaults to :code:`False`.

    dry_run : bool
        If :code:`True`, then the function will not make any API calls.
        Defaults to :code:`False`.

    verbose : bool
        If :code:`True`, then the function will print progress bars
        and other information.
        Defaults to :code:`True`.

    Returns
    -------

    np.array:
        Samples of the parameterisation.

    np.array: optional:
        The MLE loss for the approximated model
        if :code:`return_mle_loss_and_samples` is :code:`True`.

    Tuple[np.array, np.array]: optional:
        The samples of the data points and the predictions
        if :code:`return_mle_loss_and_samples` is :code:`True`.

    """

    # check the inputs
    if feature_names is None:
        raise ValueError("Feature names must be provided.")

    possible_required_models = [
        "linear",
        "logistic",
    ]
    assert (
        required_model in possible_required_models
    ), f"required_model must be one of {possible_required_models}"

    if required_model not in system_role:
        raise ValueError(
            "The system role must contain the required model "
            "so that the language model knows what to use internally."
        )

    # sample the data
    x = rng.uniform(
        low=x_sample_low,
        high=x_sample_high,
        size=(n_samples, n_datapoints_in_sample, len(feature_names)),
    )

    # get the predictions from the gpt model
    # that we can use to build a distribition
    y_pred = [
        get_llm_predictions(
            client=client,
            x=np.round(x_sample, 3),
            system_role=system_role,
            final_message=final_message,
            feature_names=feature_names,
            demonstration=demonstration,
            dry_run=dry_run,
            verbose=verbose,
        )
        for x_sample in tqdm.tqdm(x, desc="Getting predictions", disable=not verbose)
    ]

    if dry_run:
        return None

    x_sample_out, y_sample_out = [], []

    if required_model == "linear":

        weights = []
        mle_loss = []
        # we use linear regression to get the weights
        # by performing MLE
        for x_sample, y_sample in zip(x, y_pred):
            y_sample = np.array(y_sample).ravel()

            try:
                mle_model = sklm.LinearRegression(fit_intercept=True).fit(
                    x_sample, y_sample
                )
                weights_sample = mle_model.coef_
                intercept_sample = mle_model.intercept_
                weights_sample = np.concatenate(
                    [np.array([intercept_sample]), weights_sample]
                )
                weights.append(weights_sample)
                if return_mle_loss_and_samples:
                    mle_loss.append(
                        skmetrics.mean_squared_error(
                            y_sample, mle_model.predict(x_sample)
                        )
                    )
                    x_sample_out.append(x_sample)
                    y_sample_out.append(y_sample)

            except ValueError as e:
                print("y:", y_pred)
                print("y:", y_sample)
                print("x:", x_sample)
                print(e)
                print("skipping this sample")
                weights.append(None)
                mle_loss.append(None)
                x_sample_out.append(x_sample)
                y_sample_out.append(y_sample)

            parameter_distribution = weights

    if required_model == "logistic":

        weights = []

        # we use linear regression to get the weights
        # by performing MLE

        # we will first perform the opposite of the sigmoid
        # function to get the logits

        mle_loss = []

        for x_sample, y_sample in zip(x, y_pred):
            y_sample = inv_logistic(y_sample)
            y_sample = np.array(y_sample).ravel()

            try:
                mle_model = sklm.LinearRegression(fit_intercept=True).fit(
                    x_sample, y_sample
                )
                weights_sample = mle_model.coef_
                intercept_sample = mle_model.intercept_
                weights_sample = np.concatenate(
                    [np.array([intercept_sample]), weights_sample]
                )
                weights.append(weights_sample)
                if return_mle_loss_and_samples:
                    mle_loss.append(
                        skmetrics.mean_squared_error(
                            y_sample, mle_model.predict(x_sample)
                        )
                    )
                    x_sample_out.append(x_sample)
                    y_sample_out.append(y_sample)

            except ValueError as e:
                print("y:", y_pred)
                print("inverse logistic y:", y_sample)
                print("x:", x_sample)
                print(e)
                print("skipping this sample")
                weights.append(None)
                mle_loss.append(None)
                x_sample_out.append(x_sample)
                y_sample_out.append(y_sample)

            parameter_distribution = weights

    if return_mle_loss_and_samples:
        return (
            parameter_distribution,
            mle_loss,
            (x_sample_out, y_sample_out),
        )

    return parameter_distribution
