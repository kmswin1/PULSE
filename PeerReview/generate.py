from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets
from reward_models import aggregate_scores
import numpy as np
import json
import os
import sys
import tqdm
dataset = sys.argv[1]
actor_model = sys.argv[2]

assert dataset is not None

llm = LLM(model=actor_model, tokenizer=actor_model, gpu_memory_utilization=0.9)
tokenizer = llm.get_tokenizer()

if dataset == "ultrafeedback_all":
    ds1 = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")["train_iteration_1"]
    ds2 = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")["train_iteration_2"]
    ds3 = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")["train_iteration_3"]
    ds = concatenate_datasets([ds1, ds2, ds3])
elif dataset == "ultrafeedback_iter1":
    ds = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")["train_iteration_1"]
elif dataset == "ultrafeedback_iter2":
    ds = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")["train_iteration_2"]
elif dataset == "ultrafeedback_iter3":
    ds = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")["train_iteration_3"]

def get_general_prompt(instruction):
    prompt="""
    {}
    """.format(instruction)
    
    return prompt

num_samples=1
bsz=64
cnt = 0
instructions=[]
inp = []
file_name = actor_model.split("/")[-1]
with open(f"{file_name}_{dataset}.jsonl", "w") as wf:
    for meta in tqdm.tqdm(ds):
        instruction = meta["prompt"]
        if "gemma" not in actor_model:
            prompts = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": get_general_prompt(instruction)}]
        else:
            prompts = [{"role": "user", "content": get_general_prompt(instruction)}]

        if cnt < bsz:
            cnt += 1
            instructions.append(instruction)
            inp.append(prompts)
        else:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                n=1,
            )

            responses = llm.chat(
                inp,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            
            if len(responses) != bsz:
                raise ValueError(
                    f"Generated {len(responses)} responses instead of {bsz}"
                )

            completions = [
                output.text
                for r in responses
                for output in r.outputs
            ]
            completion_tokens = [
                len(output.token_ids)
                for r in responses
                for output in r.outputs
            ]

            # Check we generated the correct number of completions for each prompt
            if len(completions) != bsz:
                raise ValueError(f"Generated {len(completions)} completions instead of {bsz}")
            
            # Select the completion with the highest score

            for instruction, prompt, completion in zip(instructions, inp, completions):
                res = {}
                res["instruction"] = instruction
                res["prompt"] = prompt
                res["completion"] = completion
                
                wf.write(json.dumps(res)+"\n")
                
            cnt = 0
            inp = []
            instructions = []
    
    if inp:
        sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.9,
                    n=1,
                )

        responses = llm.chat(
            inp,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        

        completions = [
            output.text
            for r in responses
            for output in r.outputs
        ]
        completion_tokens = [
            len(output.token_ids)
            for r in responses
            for output in r.outputs
        ]
        # Select the completion with the highest score

        for instruction, prompt, completion in zip(instructions, inp, completions):
            res = {}
            res["instruction"] = instruction
            res["prompt"] = prompt
            res["completion"] = completion
            
            wf.write(json.dumps(res)+"\n")
