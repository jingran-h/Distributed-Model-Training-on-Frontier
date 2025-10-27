import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

SCRATCH_ROOT = "/lustre/orion/gen150/scratch/william_huang"
TOKENIZER_CACHE_DIR = os.path.join(SCRATCH_ROOT, "llama-tokenizers-cache")
MODEL_CACHE_DIR = os.path.join(SCRATCH_ROOT, "llama-models-cache")
DATA_CACHE_DIR = os.path.join(SCRATCH_ROOT, "data_cache")

if not os.path.isdir(MODEL_CACHE_DIR):
    raise FileNotFoundError(f"Expected model snapshot under {MODEL_CACHE_DIR}. Download it on a login node first.")
if not os.path.isdir(TOKENIZER_CACHE_DIR):
    raise FileNotFoundError(f"Expected tokenizer snapshot under {TOKENIZER_CACHE_DIR}. Download it on a login node first.")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat"
base_model_name = "meta-llama/Meta-Llama-3-8B"
new_model_name = "llama-3-8b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_CACHE_DIR,
    trust_remote_code=True,
    local_files_only=True
)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_DIR,
    device_map="cuda:0",
    local_files_only=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Data set
data_name = "mlabonne/guanaco-llama2-1k"
DATASET_LOCAL_DIRNAME = data_name.replace("/", "___")
LOCAL_ARROW_BASENAME = "guanaco-llama2-1k-train.arrow"


def load_training_split() -> Dataset:
    dataset_root = Path(DATA_CACHE_DIR) / DATASET_LOCAL_DIRNAME
    if dataset_root.is_dir():
        try:
            arrow_path = next(dataset_root.rglob(LOCAL_ARROW_BASENAME))
            print(f"Loading training data from local cache: {arrow_path}")
            return Dataset.from_file(str(arrow_path))
        except StopIteration:
            pass

    try:
        print("Local Arrow cache not found; falling back to HF cached dataset.")
        return load_dataset(
            data_name,
            split="train",
            cache_dir=DATA_CACHE_DIR,
            local_files_only=True
        )
    except Exception as offline_err:
        msg = (
            f"Unable to load dataset '{data_name}' from local cache under {DATA_CACHE_DIR}. "
            "Download it on a login node (with network access) before launching training."
        )
        raise RuntimeError(msg) from offline_err


training_data = load_training_split()
# check the data
print(training_data.shape)
# #11 is a QA sample in English
print(training_data[11])

# Training Params
train_params = SFTConfig(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    #optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
)

from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    processing_class=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()
model.save_pretrained("finetuned_llama")
