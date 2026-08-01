"""
RoBERTa + LoRA training and inference for the LLM arm.

Not "QLoRA" - we do NOT use 4-bit quantization. Reasons:
  1. RoBERTa-base is only 125M params; it fits easily on a T4 in fp16.
  2. bitsandbytes on Colab has been unreliable (triton.ops missing,
     torchao version conflicts, etc.). Dropping it eliminates the whole
     class of dependency problems.

The parameter efficiency (~1% trainable) comes from LoRA, not from
quantization. So we keep the LoRA benefit and skip the fragile 4-bit step.

Required library versions (as of 2026-07):
    peft == 0.11.1
    transformers >= 4.40, < 4.45
    accelerate  (any recent)
    NO bitsandbytes  (uninstall it if present)
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, PeftModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from .config import (FEATURES, LABEL, LLM_MODEL_NAME, LLM_MAX_LEN,
                     LLM_BATCH_SIZE, LLM_EPOCHS, LLM_LR,
                     LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
                     DEFAULT_SEED)
from .flow_serialization import flow_to_text, FlowDataset


# ---------------------------------------------------------------------------
# BUILD A FRESH MODEL
# ---------------------------------------------------------------------------
def build_model_and_tokenizer(model_name=LLM_MODEL_NAME):
    """Load RoBERTa in fp32 with a fresh classification head + LoRA adapters.

    Load in fp32 because the Trainer manages fp16 mixed precision internally
    during training. Loading fp16 + trainer fp16 causes:
        ValueError: Attempting to unscale FP16 gradients.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to("cuda")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds), "acc": accuracy_score(labels, preds)}


def train_llm(model, tokenizer, df_train, output_dir,
              seed=DEFAULT_SEED,
              batch_size=LLM_BATCH_SIZE, epochs=LLM_EPOCHS, lr=LLM_LR,
              val_size=4000):
    """Fine-tune RoBERTa+LoRA on a flow dataset.

    Uses 70/30 train/val split from df_train. The val subset is capped to
    4000 rows to keep evaluation fast between epochs.
    """
    tr_df, val_df = train_test_split(
        df_train, test_size=0.3, random_state=seed, stratify=df_train[LABEL],
    )

    train_ds = FlowDataset(tr_df, tokenizer, seed=seed)
    val_ds = FlowDataset(val_df, tokenizer, seed=seed, n=val_size)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        optim="adamw_torch",
        fp16=True,             # Trainer manages this; model must load in fp32.
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, val_df


def load_trained(adapter_dir, model_name=LLM_MODEL_NAME):
    """Load a base model and attach a previously saved LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------
def llm_predict(model, tokenizer, df, batch_size=256, verbose=False):
    """Batched inference. Returns a numpy array of predicted labels.

    Same output shape as sklearn's `.predict()`, so it drops directly into
    the evaluation harness's `evaluate()` function.
    """
    model.eval()
    texts = [flow_to_text(df.iloc[i]) for i in range(len(df))]
    preds = []
    n = len(texts)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True, padding=True, max_length=LLM_MAX_LEN,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to("cuda")
            attn = enc["attention_mask"].to("cuda")
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            if verbose and start % (batch_size * 20) == 0:
                print(f"  {start:,}/{n:,} flows...", flush=True)
    return np.array(preds)


def timed_predict(model, tokenizer, df, batch_size=256):
    """Predict and also report flows-per-second for the efficiency table."""
    t0 = time.time()
    preds = llm_predict(model, tokenizer, df, batch_size=batch_size)
    elapsed = time.time() - t0
    fps = int(len(df) / max(elapsed, 1e-9))
    return preds, elapsed, fps
