import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load the tokenizer and model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Apply LoRA for memory-efficient training
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Ensure LoRA parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# Load and preprocess dataset
dataset = load_dataset("json", data_files="data.json")  # Update path to your dataset
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=15,
    per_device_train_batch_size=1,        # Small batch size for limited memory
    gradient_accumulation_steps=8,        # Accumulate to simulate larger batch size
    fp16=True,                            # Mixed precision for memory saving
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=True,          # Enable gradient checkpointing
    logging_steps=10,                     # Log regularly to monitor progress
    save_total_limit=1,                   # Limit saved models to save disk space
    logging_dir="./logs",
    metric_for_best_model="eval_loss",    # Choose the best model based on eval loss
    greater_is_better=False  
)

# Custom metric calculation function
def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Handle model output properly
    if isinstance(predictions, tuple):  # Some models return logits as a tuple
        predictions = predictions[0]

    # Ensure that the predictions are the logits
    pred_ids = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    accuracy = np.mean([pred.strip() == label.strip() for pred, label in zip(pred_texts, label_texts)])
    return {"accuracy": accuracy}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()
trainer.save_model("./finetuned_t5base_model")
tokenizer.save_pretrained("./finetuned_t5base_model")
