import numpy as np
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from transformers import AdamW, get_cosine_schedule_with_warmup

# Load the tokenizer and model
model_name = "google/flan-t5-small"  # can be fine-tuned with a 4Gig gpu
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize function for both input (prompt) and output (nodes sequence)
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]  # Set target labels
    return inputs

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

# Custom function to compute sequence-level accuracy
def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Ensure `predictions` has consistent shape
    if isinstance(predictions, tuple):  # Some models return logits as a tuple
        predictions = predictions[0]

    # Convert logits to token ids
    pred_ids = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions

    # Decode token IDs for predictions and labels
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate sequence-level accuracy
    accuracy = np.mean([pred.strip() == label.strip() for pred, label in zip(pred_texts, label_texts)])
    return {"accuracy": accuracy}


# Main entry point for training
if __name__ == "__main__":

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # load and preprocess the dataset
    dataset = load_dataset("json", data_files="../data.json")

    # split the dataset
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    # Apply to both training and validation datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)


    #Define the training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=100,            # Still keep it relatively high to give room for early stopping
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,              # Lower learning rate for more precise updates
        weight_decay=0.01,               # Adds regularization to prevent overfitting
        eval_strategy="epoch",           # Evaluate at the end of each epoch
        save_strategy="epoch",
        logging_steps=10,                # Frequent logging to monitor training
        save_total_limit=1,              # Limit the number of saved models
        logging_dir="./logs",
        load_best_model_at_end=True,        # Load the best model based on eval loss
        metric_for_best_model="eval_loss",  # Metric to choose the best model
        greater_is_better=False	        # For loss, lower values are better
    )

    # Initialize the optimizer and scheduler
    optimizer, scheduler = custom_optimizer_scheduler()

    # Initialize the Trainer with the custom early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # Validation dataset for early stopping to monitor eval_loss
        compute_metrics=compute_metrics, # Include the custom metric
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add the early stopping callback
        optimizers=(optimizer, scheduler) #Use custom optimizer and scheduler
    )

    # Start training
    trainer.train()

    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

    # Run evaluation and print metrics
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)
