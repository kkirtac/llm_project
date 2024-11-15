import numpy as np
import sacrebleu
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from typing import Dict
from transformers.trainer_utils import EvalPrediction

# Load the tokenizer and model
model_name = "google/flan-t5-small"  # Use the same model name as in your training script
model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_model")  # Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the tokenization function
def tokenize_function(examples: Dict[str, str]) -> Dict[str, list]:
    """
    Tokenizes both input (prompt) and output (nodes sequence) for evaluation.
    
    Args:
        examples (Dict[str, str]): Dictionary containing the keys 'prompt' and 'output'.
    
    Returns:
        Dict[str, list]: Dictionary with tokenized input_ids and labels.
    """
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]  # Set target labels
    return inputs

# Define compute metrics function
def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Computes BLEU and Exact Match scores for evaluation.
    
    Args:
        eval_preds (EvalPrediction): Tuple containing prediction logits and label ids.
    
    Returns:
        Dict[str, float]: Dictionary with BLEU and Exact Match.
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

    # Prepare references for SacreBLEU
    references = [[ref] for ref in label_texts]  # Each reference as a single-item list
    preds = pred_texts  # Predictions as is

    # Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(preds, references)

    # Calculate Exact Match
    exact_match = np.mean([pred.strip() == label.strip() for pred, label in zip(pred_texts, label_texts)])

    print(f"bleu_score {bleu_score.score}")
    print(f"exact_match {exact_match}")

    return {
        "bleu": bleu_score.score,
        "exact_match": exact_match
    }

def main():
    # Load dataset
    dataset = load_dataset("json", data_files="../data.json")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)  # Split into train and test sets

    # Tokenize dataset
    test_dataset = dataset['test'].map(tokenize_function, batched=True)

    # Define training arguments (you can keep them similar to the ones used during training)
    training_args = TrainingArguments(
        output_dir="./eval_output",            # Output directory (optional)
        per_device_eval_batch_size=1,     # Evaluation batch size
        logging_dir="./eval_logs",             # Log directory (optional)
    )

    # Initialize the Trainer with the evaluation dataset and compute_metrics function
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model and get metrics
    metrics = trainer.evaluate()

    # Print the evaluation metrics
    print("Evaluation Metrics:")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")

if __name__ == "__main__":
    main()
