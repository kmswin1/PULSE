# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
from PULSE.PULSETrainer import PULSETrainer
import sys

model_name = sys.argv[1]
dataset = sys.argv[2]
local_rank = int(os.environ.get('LOCAL_RANK', '0'))

if "Mistral" not in model_name:
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to(f"cuda:{local_rank}")
    model.config.use_cache=False
    model_ref = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to(f"cuda:{local_rank}")
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(f"cuda:{local_rank}")
    model.config.use_cache=False
    model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(f"cuda:{local_rank}")
    
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side  = 'left'
ds = load_dataset({dataset})["train"].train_test_split(test_size=0.01)
train_ds = ds["train"]
test_ds = ds["test"]
print (train_ds)

fsdp_config={'limit_all_gathers': True, 'forward_prefetch': True, 'backward_prefetch': 'backward_pre'}
file_name = model_name.split("/")[-1]

lr = 1e-7
seq_len = 512
label_smoothing=0.1
if "mistral" in model_name:
    beta=0.05
else:
    beta=0.1

training_args = DPOConfig(output_dir=dataset+"_"+"PULSE",
                        logging_steps=10,
                        num_train_epochs=3,
                        per_device_train_batch_size=1,
                        warmup_ratio=0.1,
                        per_device_eval_batch_size=1, 
                        gradient_accumulation_steps=8,
                        gradient_checkpointing=True if "gemma" in model_name else False,
                        save_strategy="epoch",
                        metric_for_best_model="eval_loss",
                        load_best_model_at_end=True,
                        eval_strategy='epoch',
                        lr_scheduler_type="linear",
                        bf16=True,
                        max_prompt_length=128,
                        max_completion_length=seq_len,
                        save_total_limit=3,
                        fsdp="full_shard auto_wrap",
                        fsdp_config=fsdp_config,
                        report_to='tensorboard',
                        learning_rate=lr,
                        beta=beta,
                        optim="rmsprop",
                        loss_type="pulse",
                        remove_unused_columns=False,
                        label_smoothing=label_smoothing,
                        )

trainer = PULSETrainer(model=model,
                    ref_model=model_ref,
                    args=training_args,
                    tokenizer=tokenizer, 
                    train_dataset=train_ds,
                    eval_dataset=test_ds,
                    )

trainer.train()
