import json
import re

# Example when actor LLM is Mistral and critic LLMs are LLaMA3, Qwen2,5, and Gemma2
student="Mistral-7B-Instruct-v0.2"
peer1="Meta-Llama-3-8B-Instruct"
peer2="Qwen2.5-7B-Instruct"
peer3="gemma-2-9b-it"
iter="all"

with open(f"{peer1}_review_{student}_total_reviews_{iter}_revision.jsonl", "r") as f1:
    with open(f"{peer2}_review_{student}_total_reviews_{iter}_revision.jsonl", "r") as f2:
        with open(f"{peer3}_review_{student}_total_reviews_{iter}_revision.jsonl", "r") as f3:
                with open(f"{student}_total_reviews_{iter}.jsonl", "w") as wf:
                    for line1, line2, line3 in zip(f1, f2, f3):
                        line1 = json.loads(line1)
                        line2 = json.loads(line2)
                        line3 = json.loads(line3)
                        scores = []
                        reviews = []
                        paragraphs=line1["review"].split("\n")
                        for p in paragraphs:
                            if "score:" in p.lower():
                                score = re.findall(r'\d', p)
                                if not score:   
                                    continue
                                score = int(score[0])
                                scores.append(score)
                                reviews.append(line1["review2"])
                                break
                        paragraphs=line2["review"].split("\n")
                        for p in paragraphs:
                            if "score:" in p.lower():
                                score = re.findall(r'\d', p)
                                if not score:   
                                    continue
                                score = int(score[0])
                                scores.append(score)
                                reviews.append(line2["review2"])
                                break
                        paragraphs=line3["review"].split("\n")
                        for p in paragraphs:
                            if "score:" in p.lower():
                                score = re.findall(r'\d', p)
                                if not score:   
                                    continue
                                score = int(score[0])
                                scores.append(score)
                                reviews.append(line3["review"])
                                break
                        
                        if not scores:
                            continue
                        res = {}
                        res["instruction"] = line1["instruction"]
                        res["answer"] = line1["answer"]
                        res["revision"] = line1["revision"]
                        res["score"] = line1["score"]
                        res["avg_score"] = sum(scores)/len(scores)
                        res["review"] = "\n\n".join([f"Review {i+1}\n\n" + review.strip() for i, review in enumerate(reviews)])
                        wf.write(json.dumps(res)+"\n")