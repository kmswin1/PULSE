import json
student="Mistral-7B-Instruct-v0.2"
iter="all"

with open(f"{student}_total_final_reviews_{iter}.jsonl", "r") as f:
    with open(f"{student}_training_{iter}.jsonl", "w") as wf:
        for line in f:
            line = json.loads(line)
            res = {}
            if line["score_stage1"] >= line["score_stage2"]:
                res["chosen"] = [{"role": "user", "content": line["instruction"]}, {"role": "assistant", "content": line["answer"]}]
                res["rejected"] = [{"role": "user", "content": line["instruction"]}, {"role": "assistant", "content": line["revision"]}]
                res["score_chosen"] = line["score_stage1"]
                res["score_rejected"] = line["score_stage2"]

            elif line["score_stage2"] > line["score_stage1"]:
                res["chosen"] = [{"role": "user", "content": line["instruction"]}, {"role": "assistant", "content": line["revision"]}]
                res["rejected"] = [{"role": "user", "content": line["instruction"]}, {"role": "assistant", "content": line["answer"]}]
                res["score_chosen"] = line["score_stage2"]
                res["score_rejected"] = line["score_stage1"]
            wf.write(json.dumps(res)+"\n")