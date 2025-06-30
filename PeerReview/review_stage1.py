from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from datasets import load_dataset
from reward_models import aggregate_scores
import numpy as np
import json
import os
import sys
import tqdm

actor_data = sys.argv[1]
critic_model = sys.argv[2]

llm = LLM(model=critic_model, tokenizer=critic_model, gpu_memory_utilization=0.95)
tokenizer = llm.get_tokenizer()
ds = load_dataset("json", data_files=actor_data+".jsonl")["train"]


def get_review_prompt(question, response):
    prompt="""
    Your task is to review the peer's answer to the question below. After reading and deeply thinking about the answer, you can rate the grade between 1 and 5 whether the current answer is acceptable.
    For example, 1: reject, 2: weak reject, 3: borderline, 4: weak accept, 5: accept.
    You should write both the summary of strengths and weaknesses about answer including what needs to be revised. When you are ready to write, conclude using the format Score: "..."\nWeaknesses: "..."\nStrengths: "..."\nRecommended Suggestions: "...".
    Question: {}
    Answer: {}
    """.format(question, response)
    
    return prompt

num_samples=1
bsz=64
cnt = 0
instructions=[]
inp = []
answers = []
file_name = critic_model.split("/")[-1]
with open(f"{file_name}_review_{actor_data}.jsonl", "w") as wf:
    for meta in tqdm.tqdm(ds):
        instruction = meta["instruction"]
        answer = meta["completion"]
        prompts = [{"role": "user", "content": get_review_prompt(instruction, answer)}]

        if cnt < bsz:
            cnt += 1
            instructions.append(instruction)
            inp.append(prompts)
            answers.append(answer)
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

            for instruction, answer,  prompt, completion in zip(instructions, answers, inp, completions):
                res = {}
                res["instruction"] = instruction
                res["prompt"] = prompt
                res["answer"] = answer
                res["review"] = completion
                
                wf.write(json.dumps(res)+"\n")
                
            cnt = 0
            inp = []
            instructions = []
            answers = []
    
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

        for instruction, answer,  prompt, completion in zip(instructions, answers, inp, completions):
            res = {}
            res["instruction"] = instruction
            res["prompt"] = prompt
            res["answer"] = answer
            res["review"] = completion
            
            wf.write(json.dumps(res)+"\n")
