import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, Dataset, DatasetDict
from transformers.trainer_utils import EvalPrediction
from typing import Dict, Tuple

# Load the tokenizer and model
model_name = "google/flan-t5-small"  # can be fine-tuned with a 4Gig GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples: Dict[str, str]) -> Dict[str, list]:
    """
    Tokenizes both input (prompt) and output (nodes sequence) for training.
    
    Args:
        examples (Dict[str, str]): Dictionary containing the keys 'prompt' and 'output'.
    
    Returns:
        Dict[str, list]: Dictionary with tokenized input_ids and labels.
    """
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]  # Set target labels
    return inputs

def custom_optimizer_scheduler(trainer=None) -> Tuple[AdamW, get_cosine_schedule_with_warmup]:
    """
    Sets up a custom optimizer (AdamW) and scheduler (cosine schedule with warmup) for training.
    
    Args:
        trainer: Trainer object (optional, default=None)
    
    Returns:
        Tuple[AdamW, get_cosine_schedule_with_warmup]: Configured optimizer and scheduler.
    """
    # AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    # Cosine annealing learning rate scheduler with warmup
    num_training_steps = len(train_dataset) * training_args.num_train_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps for warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler

def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Computes the sequence-level accuracy between predictions and labels.
    
    Args:
        eval_preds (EvalPrediction): Tuple containing prediction logits and label ids.
    
    Returns:
        Dict[str, float]: Dictionary with calculated accuracy.
    """
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
    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and preprocess the dataset
    dataset: DatasetDict = load_dataset("json", data_files="../data.json")

    # Split the dataset
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset: Dataset = dataset['train']
    val_dataset: Dataset = dataset['test']

    # Apply tokenization to both training and validation datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=100,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize the optimizer and scheduler
    optimizer, scheduler = custom_optimizer_scheduler()

    # Initialize the Trainer with the custom early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(optimizer, scheduler),
    )

    # Start training
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

    # Run evaluation and print metrics
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)
