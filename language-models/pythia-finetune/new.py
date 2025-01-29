import os
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from utils import autotune_batch_size, get_device

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async kernel execution
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Prevent hangs
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer contention

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m"  # Lightweight model for quick training
BLOCK_SIZE = 1024  # Easily adjustable (512, 1024, etc)
USE_FLASH = False  # Enable if your GPU supports it (T4/V100/A100+)
DEVICE = get_device()

# Model setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2" if USE_FLASH else "eager",
    # torch_dtype=torch.float16 if not USE_FLASH else torch.bfloat16,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Auto-tune batch size for single GPU
BATCH_SIZE = 32  # Default fallback
if torch.cuda.device_count() == 1:
    BATCH_SIZE = autotune_batch_size(model, tokenizer, BLOCK_SIZE)
    print(
        f"\n🔥 Autotuned batch size: {BATCH_SIZE} (VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB)\n"
    )


# Optimized dataset processing
def chunk_tokenize(examples):
    # Process text in batches with parallel workers
    texts = [t.strip() for t in examples["text"] if t.strip()]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=BLOCK_SIZE,
        return_overflowing_tokens=True,
        return_length=True,
        return_tensors="pt",  # Process as tensors directly
    )

    # Convert to lists for Dataset compatibility
    return {k: [v[i] for i in range(v.size(0))] for k, v in tokenized.items()}


# Load and process dataset
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
dataset = dataset.map(
    chunk_tokenize,
    batched=True,
    remove_columns=["text"],
    num_proc=8,  # Maximize parallel processing
    batch_size=32,  # Larger batches for mapping efficiency
)


# Training setup
args = TrainingArguments(
    output_dir=f"./results-{MODEL_NAME}",
    # Batch configuration (core performance)
    per_device_train_batch_size=BATCH_SIZE,  # From autotuner
    gradient_accumulation_steps=1,  # Correct for pure batch focus
    dataloader_num_workers=max(4, os.cpu_count() // 2),  # Dynamic worker allocation
    dataloader_pin_memory=True,  # Faster GPU transfers
    # Precision configuration
    bf16=USE_FLASH,  # Keep first - modern Ampere+ GPUs
    fp16=not USE_FLASH,  # Explicit fallback
    # fp16_full_eval=False,  # Reduce memory during eval
    # tf32=True,  # Disable if not running on Ampere+ GPUs
    # gradient_checkpointing=True,  # Keep - enables 2x batch size
    # Optimization core
    learning_rate=5e-5,
    num_train_epochs=3,
    # lr_scheduler_type="cosine",
    # warmup_ratio=0.1,  # 10% of training for warmup
    optim="adamw_torch_fused",
    # torch_compile=True,
    # torch_compile_backend="inductor",  # Best for NVIDIA GPUs
    # torch_compile_mode="max-autotune",
    # deepspeed={
    #     "zero_optimization": {"stage": 1},
    #     "fp16": {"enabled": not USE_FLASH},
    #     "bf16": {"enabled": USE_FLASH},
    # }
    # if torch.cuda.device_count() != 1
    # else None,  # Disable DeepSpeed for single GPU (Why? DeepSpeed is designed for multi-GPU training)
    report_to="wandb",
    run_name=f"{MODEL_NAME}-wikitext-{datetime.now().strftime('%Y%m%d-%H%M')}",
    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,
    # max_grad_norm=1.0,  # Stability for large batches
)

# Efficient data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=32,  # For tensor core efficiency
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Launch training
trainer.train()
