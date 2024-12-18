#!/bin/bash
python llm_prior_vs_no_prior.py --dataset diabetes hypothyroid --get_priors --model gpt-4o-mini-2024-07-18 
&& python llm_prior_vs_no_prior.py --dataset diabetes hypothyroid --get_priors --model gpt-4-turbo-2024-04-09 
&& python llm_prior_vs_no_prior.py --dataset diabetes hypothyroid --get_priors --model meta-llama/Llama-3.2-3B-Instruct 
&& python llm_prior_vs_no_prior.py --dataset diabetes hypothyroid --get_priors --model meta-llama/Llama-3.1-8B-Instruct 
&& python llm_prior_vs_no_prior.py --dataset diabetes hypothyroid --get_priors --model Qwen/Qwen2.5-14B-Instruct