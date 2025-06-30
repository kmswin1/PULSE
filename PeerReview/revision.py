from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from datasets import load_dataset
from reward_models import aggregate_scores
import numpy as np
import json
import os
import sys
import tqdm
review_data = sys.argv[1]
actor_model = sys.argv[2]

llm = LLM(model=actor_model, tokenizer=actor_model, gpu_memory_utilization=0.95)
tokenizer = llm.get_tokenizer()

ds = load_dataset("json", data_files=review_data+".jsonl")["train"]

def get_revision_prompt(question, response, review):
    prompt="""
    Your task is to revise your previous answer based on the peer's feedback. After reading and deeply thinking about the review, you can correct your previous answer by referring to the weaknesses and recommended corrections while retaining the strengths.
    Question: {}
    Previous Your Answer: {}
    Total Feedback: {}
    """.format(question, response, review)
    
    return prompt

num_samples=1
bsz=64
cnt = 0
instructions=[]
inp = []
answers = []
scores=[]
reviews=[]
file_name = actor_model.split("/")[-1]
with open(f"{review_data}_revision.jsonl", "w") as wf:
    for meta in tqdm.tqdm(ds):
        instruction = meta["instruction"]
        answer = meta["answer"]
        review = meta["review"]
        score = meta["avg_score"]
        prompts = [{"role": "user", "content": get_revision_prompt(instruction, answer, review)}]


        if cnt < bsz:
            cnt += 1
            instructions.append(instruction)
            inp.append(prompts)
            answers.append(answer)
            reviews.append(review)
            scores.append(score)
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

            for instruction, answer,  prompt, completion, review, score in zip(instructions, answers, inp, completions, reviews, scores):
                res = {}
                res["instruction"] = instruction
                res["prompt"] = prompt
                res["answer"] = answer
                res["review"] = review
                res["revision"] = completion
                res["score"] = score
                
                wf.write(json.dumps(res)+"\n")
                
            cnt = 0
            inp = []
            instructions = []
            answers = []
            reviews = []
            scores = []
    
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

        for instruction, answer,  prompt, completion, review, score in zip(instructions, answers, inp, completions, reviews, scores):
            res = {}
            res["instruction"] = instruction
            res["prompt"] = prompt
            res["answer"] = answer
            res["review"] = review
            res["revision"] = completion
            res["score"] = score
            
            wf.write(json.dumps(res)+"\n")
