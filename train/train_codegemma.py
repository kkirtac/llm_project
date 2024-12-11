import numpy as np
from transformers import GemmaTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset

tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/codegemma-7b")

# Load and preprocess the dataset
dataset = load_dataset("json", data_files="data.json")

# Split the dataset
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Tokenize function for input (prompt) and output (nodes sequence)
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]  # Set target labels
    return inputs

# Apply to both training and validation datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=70,                # Can reduce if compute-constrained
    per_device_train_batch_size=1,      # Set smaller batch size due to model size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,      # Simulate a larger batch size
    learning_rate=5e-6,                 # Lower learning rate for larger models
    weight_decay=0.01,                  # Regularization to prevent overfitting
    eval_strategy="epoch",              # Evaluate at the end of each epoch
    save_strategy="epoch",
    logging_steps=10,                   # Log regularly to monitor progress
    save_total_limit=1,                 # Limit saved models to save disk space
    logging_dir="./logs",
    load_best_model_at_end=True,        # Load the best model based on eval loss
    metric_for_best_model="eval_loss",  # Choose the best model based on eval loss
    greater_is_better=False             # For loss, lower values are better
)

# Custom optimizer and scheduler setup
def custom_optimizer_scheduler(trainer=None):
    # AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    # Cosine annealing learning rate scheduler with warmup
    num_training_steps = len(train_dataset) * training_args.num_train_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps for warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

# Custom metric calculation function
def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):  # Some models return logits as a tuple
        predictions = predictions[0]

    pred_ids = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    accuracy = np.mean([pred.strip() == label.strip() for pred, label in zip(pred_texts, label_texts)])
    return {"accuracy": accuracy}

# Initialize optimizer and scheduler
optimizer, scheduler = custom_optimizer_scheduler()

# Initialize the Trainer with custom settings
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    optimizers=(optimizer, scheduler)
)

# Start training
trainer.train()

# Save the model and tokenizer
trainer.save_model("./finetuned_codegemma_model")
tokenizer.save_pretrained("./finetuned_codegemma_model")
